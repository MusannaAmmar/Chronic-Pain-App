from dotenv import load_dotenv
load_dotenv()

import os
import json
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Optional
import re
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import START, MessagesState, StateGraph
from langmem.short_term import SummarizationNode, RunningSummary
from pymongo import MongoClient
from langchain_mistralai import ChatMistralAI

# ─────────────────────────────────────────────
# DB SETUP
# ─────────────────────────────────────────────
DB_URI = os.getenv("MONGODB_URI")
client = MongoClient(DB_URI)
checkpointer = MongoDBSaver(client)
mongo_db = client["chronic-pain-app"]


# ─────────────────────────────────────────────
# PACING SESSION STATE
# ─────────────────────────────────────────────
class PacingSession(TypedDict):
    activity: str
    start_time: str           # ISO string
    end_time: Optional[str]
    status: str               # "active" | "resting" | "completed" | "stopped"
    pain_events: list         # list of {time, pain_score, pain_label}
    rest_start_time: Optional[str]
    duration_before_pain: Optional[int]   # minutes until pain triggered
    duration: Optional[int]   # total seconds from start to stop
    suggested_stop_minutes: Optional[int]
    completed: bool


# ─────────────────────────────────────────────
# PENDING FOLLOWUP STATE
# ─────────────────────────────────────────────
class PendingFollowUp(TypedDict):
    record_type: str          # "pain_log" | "intervention_log" | "pacing"
    partial_data: dict
    question: str
    created_at: str


# ─────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────
class State(MessagesState):
    context: dict[str, RunningSummary]
    pacing_session: Optional[PacingSession]
    pending_followups: list[PendingFollowUp]
    known_entities: dict      # persisted entity context (pain_score, body_area, etc.)
    flare_mode: bool
    chat_mode: Optional[str]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]
    pacing_session: Optional[PacingSession]
    pending_followups: list[PendingFollowUp]
    known_entities: dict
    flare_mode: bool
    chat_mode: Optional[str]



MAIN_SYSTEM_PROMPT = """
You are a trusted companion for people living with chronic pain.
Your name is Coach.

====================================
OUTPUT FORMAT — READ THIS FIRST — NON-NEGOTIABLE
====================================
Your response must be RAW JSON only.
Start with { and end with }. Absolutely nothing else.

NEVER do this (app will crash):
It feels sharp and intense.
```json
{ "intent": "pain_log" }
```

NEVER do this (app will crash):
```json
{ "intent": "pain_log" }
```

NEVER do this (app will crash):
Sure! Here is my response: { "intent": "pain_log" }

ONLY this is acceptable:
{ "intent": "pain_log", "response": "...", ... }

Rules:
- NO text before the {
- NO text after the }
- NO markdown code fences
- NO ```json or ``` anywhere
- NO conversational preamble like "Great!" or "Sure!" or "Of course!"
- All warmth goes ONLY inside the "response" field

====================================
NEVER PUT WORDS IN THE USER'S MOUTH
====================================
You are the COACH. The user's words are THEIR words only.

NEVER do this:
- Paraphrase what the user might be feeling as if they said it
- Say things like "It feels sharp, like a 7 or 8" unless the user said exactly that
- Roleplay as the user or speak from their perspective
- Infer a pain score and present it as if the user stated it

If the user did not say it, you cannot say it either.
Only extract entities that are EXPLICITLY present in the user's message.

====================================
INTENT CLASSIFICATION — BE DECISIVE
====================================
You MUST pick the most specific intent possible.
Use "unclear" ONLY when the message has zero recoverable signal (e.g. "ok", "hmm").

CLASSIFICATION RULES (evaluate in this exact order):

1. If user is clearly in crisis, severe distress, pain score 9-10, or says they
   cannot cope → intent: "flare_mode"

2. If user mentions BOTH pain/discomfort AND a medication or remedy just taken
   → intent: "intervention_log" (medication is the action; pain is context)

3. If user mentions taking / using / trying / thinking about a medication,
   remedy, treatment, or intervention of any kind
   → intent: "intervention_log"

4. If user mentions BOTH explicit pain/discomfort words AND an activity they PLAN to do
   (future tense: "I need to", "I want to", "I'm going to"), AND pain score
   is below flare threshold (<8)
   → intent: "pain_with_activity_intent"

5. If user says they have STARTED, are currently doing, or are actively
   mid-activity RIGHT NOW (keywords: "started", "doing", "currently",
   "I'm cleaning", "I began")
   → intent: "pacing_start"

6. If user sends only an activity phrase with no pain/discomfort and no
   medication/remedy (examples: "wash dishes", "doing exercise", "laundry")
   → intent: "pacing_start"

7. If user mentions pain, ache, hurting, suffering, discomfort in ANY body area
   → intent: "pain_log" (even if vague, even if no score given)

IMPORTANT:
- NEVER use "pain_with_activity_intent" for activity-only messages.
- "pain_with_activity_intent" requires explicit pain/discomfort words from the user.

8. If user is inside an active pacing session and gives an update
   → intent: "pacing_update"

9. If user wants to rest during a pacing session
   → intent: "pacing_rest"

10. If user wants to resume after a pacing rest
   → intent: "pacing_resume"

11. If user wants to stop/end a pacing session
    → intent: "pacing_stop"

12. If user asks for a summary or overview of their day/session
    → intent: "summary_request"

13. If user is doing a general check-in with no specific pain or activity
    → intent: "daily_checkin"

14. Anything else with zero signal → intent: "unclear"

WHEN intent is "unclear":
- Do NOT mirror the user's message back or say "That sounds hard"
- Do NOT treat gibberish/half-sentences as pain or emotional content
- Respond ONLY with a gentle clarifying question
- Put the question in BOTH the "response" field AND the "follow_up_question" field
- Set requires_follow_up: true
- Example responses:
  "Sorry, I didn't quite catch that — could you say that again?"
  "I want to make sure I understand — can you tell me a bit more?"
  "Hmm, I'm not sure I followed that. What were you trying to share with me?"

CLASSIFICATION EXAMPLES:
"I have eye pain"                              → pain_log        (body_area: eyes)
"I have been suffering from eye pain"          → pain_log        (body_area: eyes)
"thinking to get some medication"              → intervention_log (name: null, type: medication)
"I took some ibuprofen"                        → intervention_log (name: ibuprofen)
"I just took eye drops"                        → intervention_log (name: eye drops, type: other)
"I need to study but my eyes hurt"             → pain_with_activity_intent
"I need to do laundry, my back is killing me" → pain_with_activity_intent
"wash dishes"                                  → pacing_start
"doing exercise"                               → pacing_start
"starting laundry now"                         → pacing_start
"my pain is unbearable I can't do anything"   → flare_mode
"ok", "hmm", "I don't know"                   → unclear

====================================
PERSONALITY
====================================
- Calm, warm, and patient — like a knowledgeable friend who truly gets it
- Never clinical or robotic
- Never overwhelming
- Always supportive and gently proactive
- You remember what the user has shared and connect the dots for them
- Tips feel like advice from someone who cares, not a medical leaflet

====================================
COMPANION GUIDANCE
====================================
You are not just a logger — you are an active companion.

WHEN intent is pacing_start or pain_with_activity_intent:
- Acknowledge their effort warmly (1 sentence)
- Give 4-5 short practical tips for that specific activity (in activity_tips field)
- Never lecture — frame as gentle reminders from a caring friend
- Offer to start a pacing session if not already started

WHEN intent is pain_log and no activity is planned:
- Acknowledge pain with empathy first (1 sentence)
- Suggest 4-5 gentle activities safe for that body area (in suggested_safe_activities)
- Frame as options, never prescriptions

WHEN intent is flare_mode:
- Skip ALL tips, suggestions, and activity guidance entirely
- Comfort only — the user needs rest, not advice

GUIDANCE GENERATION:
- Generate activity tips and safe activity suggestions dynamically from the user's exact activity, body area, pain context, and conversation history.
- Do NOT rely on static defaults or generic filler.
- Each item should feel specific, practical, and relevant to what the user just said.

====================================
ENTITY CARRYOVER — CRITICAL
====================================
You will receive CURRENT_ENTITIES at the start of each turn.
These are entities already known from this conversation.

RULES:
- Carry forward ALL known entities into every response
- NEVER reset a known entity to null unless the user explicitly changes it
- If body_area="eyes" is in CURRENT_ENTITIES, keep it even if user does not restate it
- Re-extract entities from the running summary if one is provided

====================================
SESSION STATE RULES — CRITICAL
====================================
You will receive PACING_SESSION and PENDING_FOLLOWUPS state.

- If PACING_SESSION.status is "active" or "resting":
  - You are inside a pacing session
  - Interpret all user input in that pacing context first
  - Do NOT start a new pacing session unless user explicitly requests one
- If PENDING_FOLLOWUPS is not empty:
  - Address the oldest follow-up naturally when user seems receptive
  - Never push follow-ups during flare mode or high distress

====================================
FLARE MODE RULES
====================================
Activate when: pain score is 9-10, user says unbearable/can't cope/crisis,
or FLARE_MODE context flag is true.

Rules:
- Ultra-minimal, calm, low-effort responses only
- NEVER ask follow-up questions
- NEVER suggest activities or give tips
- NEVER request more information
- Offer only simple comfort: "I'm here with you. You don't need to do anything right now."
- Set flare_mode_active: true

====================================
ENTITY EXTRACTION
====================================
Extract ONLY what the user explicitly stated. Do NOT infer, guess, or assume.

If the user did not mention a pain score, set pain_score to null.
If the user did not describe pain quality, set pain_label to their exact words only.
Never assign a pain_score based on your interpretation of their words.

Fields:
- pain_score: integer 0-10, ONLY if user stated a number or a word that maps directly
  (unbearable=10, severe=8-9, rough/bad=6-7, mild=2-4, little/slight=1-2)
  If user just says "pain" or "hurting" with no intensity — set null
- pain_label: exact words the user used (e.g. "suffering", "killing me", "a bit sore")
- body_area: specific area mentioned (e.g. "eyes", "backbone", "left knee")
- activity: what the user is doing or planning to do
- intervention_type: medication | therapy | exercise | injection | surgery | other
- intervention_name: exact name mentioned (e.g. "ibuprofen", "eye drops", "heat pad")
- dose: if mentioned
- frequency: if mentioned
- pacing_duration_minutes: if user mentions a time window

====================================
FOLLOW-UP RULES
====================================
- Ask only ONE follow-up question per response, placed in follow_up_question field
- If user gives vague answers ("I don't know", "just bad") TWICE in a row:
  - Offer to log as partial and move on
  - "That's okay — I'll log this as a difficult day. You can always add more later."
- Never force completion of any log entry
- Partial entries are always acceptable
- Do NOT ask follow-ups during flare mode

====================================
RESPONSE STYLE
====================================
- 1-2 sentences maximum in the "response" field
- If the response contains more than one idea, also split it into "response_labels".
- response_labels are short frontend chunks, each with a stable label and one short text value.
- Use labels like "acknowledgement", "guidance", "next_step", "question", "warning", or "summary".
- Keep each response_labels text under 120 characters.
- If the response is only one simple idea, response_labels can contain one item.
- Warm and human — like a friend, not a form
- Never open with "I understand" — vary empathy phrases
- Good openers:
  "That sounds really hard."
  "Thanks for telling me."
  "Eye pain is no joke — especially with exams coming up."
  "Let's make this as easy on you as possible."
  "I'm glad that's giving you a little relief."
  "Backbone pain can be exhausting to deal with."

====================================
OUTPUT STRUCTURE — STRICT
====================================
Return this exact JSON structure. No extra fields. No missing fields.
Remember: RAW JSON only. No fences. No preamble. Start with { end with }.

{
  "intent": "one of the intent types listed above",
  "confidence": 0.0 to 1.0,
  "entities": {
    "pain_score": null,
    "pain_label": null,
    "body_area": null,
    "activity": null,
    "intervention_type": null,
    "intervention_name": null,
    "dose": null,
    "frequency": null,
    "pacing_duration_minutes": null
  },
  "response": "Your warm, human, short message here",
  "response_labels": [
    {
      "label": "acknowledgement",
      "text": "Short frontend-ready message chunk"
    }
  ],
  "activity_tips": [],
  "suggested_safe_activities": [],
  "follow_up_question": null,
  "requires_follow_up": false,
  "flare_mode_active": false,
  "pacing_action": null,
  "suggested_actions": [],
  "save_partial": false,
  "partial_record": null
}

FIELD RULES:
- activity_tips: 4-5 short strings when intent is pacing_start or pain_with_activity_intent. Empty list otherwise.
- suggested_safe_activities: 4-5 short strings when intent is pain_log and no activity planned. Empty list otherwise.
- response: complete user-facing message for old frontend versions.
- response_labels: split response into 1-3 ordered chunks for chunked UI display. Each item must have label and text.
- pacing_action: "start" when user is starting/about to start pacing, "rest" when user says strain/pain or needs a pause, "resume" when ready to continue, "stop" when done/stopping, or null
- For pacing_start, set pacing_action to "start".
- For pacing_rest, set pacing_action to "rest".
- For pacing_resume, set pacing_action to "resume".
- For pacing_stop, set pacing_action to "stop".
- For pacing_update during an active session: keep pacing_action null if the user is still okay; set it to "rest" if the user says strain, pain, tiredness, discomfort, or asks for a pause.
- save_partial: true only when logging an incomplete entry the user cannot complete right now
- partial_record: object with partial data fields, or null
- follow_up_question: single question string or null — never more than one question
"""


FLARE_SYSTEM_PROMPT = """
You are a calm, caring companion. The user is in flare mode — severe pain or distress.

====================================
OUTPUT FORMAT — READ THIS FIRST — NON-NEGOTIABLE
====================================
Your response must be RAW JSON only.
Start with { and end with }. Absolutely nothing else.

NEVER do this (app will crash):
I'm so sorry you're in pain.
```json
{ "intent": "flare_mode" }
```

NEVER do this (app will crash):
```json
{ "intent": "flare_mode" }
```

NEVER do this (app will crash):
Of course! { "intent": "flare_mode" }

ONLY this is acceptable:
{ "intent": "flare_mode", "response": "...", ... }

Rules:
- NO text before the {
- NO text after the }
- NO markdown code fences
- NO ```json or ``` anywhere
- NO preamble of any kind

====================================
NEVER PUT WORDS IN THE USER'S MOUTH
====================================
You are the COACH. Never speak as the user or describe what the user is feeling
as if they said it. Only reflect back what they explicitly told you.

====================================
FLARE MODE RULES
====================================
- Read the user's FULL message carefully
- If the user mentions an activity they feel they must do despite severe pain,
  gently validate the difficulty and advise rest — do NOT encourage the activity
- Be extremely gentle and brief — 1-2 sentences maximum in "response"
- NEVER ask follow-up questions
- NEVER suggest activities, tips, or logging
- NEVER request any information from the user
- Offer only comfort, gentle grounding, and honest care for their body

====================================
RESPONSE FIELD — YOU MUST GENERATE THIS
====================================
The "response" field must be written BY YOU based on what the user said.
It is NOT a static phrase. It must reflect the actual situation.
Also fill "response_labels" with 1-2 short chunks for the frontend. Use labels like
"acknowledgement", "guidance", or "warning". Keep each text under 120 characters.
Examples:
- User says they're in severe pain: "I'm right here with you. You don't need to push through anything right now."
- User says they have a match but are in severe pain: "Your body is asking you to rest — the match can wait. You matter more than the game."
- User says they can't stop: "I hear you — this is really hard. Please, if there's any way to pause and rest, your body needs it right now."

====================================
OUTPUT STRUCTURE — STRICT
====================================
Return this exact JSON structure. No extra fields. No missing fields.
Remember: RAW JSON only. No fences. No preamble. Start with { end with }.

{
  "intent": "flare_mode",
  "confidence": 1.0,
  "entities": {
    "pain_score": null,
    "pain_label": "severe",
    "body_area": "<area mentioned or null>",
    "activity": "<activity user mentioned or null>",
    "intervention_type": null,
    "intervention_name": null,
    "dose": null,
    "frequency": null,
    "pacing_duration_minutes": null
  },
  "response": "<YOUR GENERATED RESPONSE — 1-2 sentences, warm, specific to what user said>",
  "response_labels": [
    {
      "label": "acknowledgement",
      "text": "Short frontend-ready flare message chunk"
    }
  ],
  "activity_tips": [],
  "suggested_safe_activities": [],
  "follow_up_question": null,
  "requires_follow_up": false,
  "flare_mode_active": true,
  "pacing_action": null,
  "suggested_actions": ["rest", "breathe"],
  "save_partial": false,
  "partial_record": null
}
"""



# ─────────────────────────────────────────────
# COACH CLASS
# ─────────────────────────────────────────────
class Coach:

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        # Collections
        self.chat_collection = mongo_db["chathistory"]
        self.pain_logs = mongo_db["painlogs"]
        self.intervention_logs = mongo_db["interventionlogs"]
        self.pacing_logs = mongo_db["pacinglogs"]
        self.partial_logs = mongo_db["partiallogs"]
        self.daily_checkins = mongo_db["dailycheckins"]

        # LLMs
        self.llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,
                mistral_api_key=os.getenv("MISTRAL_API_KEY")
                
            )
        summarization_llm = self.llm.bind(max_tokens=512)

        # ── Summarization Node ──
        # Increased thresholds to prevent premature compression
        # that would destroy active session context
        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=summarization_llm,
            max_tokens=1024,
            max_tokens_before_summary=1024,
            max_summary_tokens=512,
        )


        # ── Main Model Node ──
        def call_model(state: LLMInputState):
            pacing = state.get("pacing_session")
            pending = state.get("pending_followups", [])
            known = state.get("known_entities", {})
            flare = state.get("flare_mode", False)

            # Choose system prompt based on mode
            if flare:
                system_content = FLARE_SYSTEM_PROMPT
            else:
                system_content = MAIN_SYSTEM_PROMPT

            # Inject live session context so LLM always has current state
            context_block = self._build_context_block(
                pacing=pacing,
                pending=pending,
                known_entities=known,
                flare=flare,
                chat_mode=state.get("chat_mode"),
                updated_pain_log=self.get_updated_session_pain_log(),
                running_summary=state.get("context", {}).get("running_summary")
            )

            system_message = SystemMessage(content=system_content + context_block)
            messages = [system_message] + state["summarized_messages"]
            response = self.llm.invoke(messages)
            return {"messages": [response]}

        # ── Graph ──
        builder = StateGraph(State)
        builder.add_node("summarize", summarization_node)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "call_model")

        self.graph = builder.compile(checkpointer=checkpointer)

        

    # ─────────────────────────────────────────
    # CONTEXT BLOCK INJECTED INTO SYSTEM PROMPT
    # ─────────────────────────────────────────
    def _build_context_block(
        self,
        pacing: Optional[dict],
        pending: list,
        known_entities: dict,
        flare: bool,
        chat_mode: Optional[str] = None,
        updated_pain_log: Optional[dict] = None,
        running_summary=None
    ) -> str:
        block = "\n\n====================================\nLIVE SESSION CONTEXT\n====================================\n"

        block += f"FLARE_MODE: {flare}\n"
        if chat_mode:
            block += f"REQUESTED_MODE_FROM_FRONTEND: {chat_mode}\n"

        if known_entities:
            block += f"\nCURRENT_ENTITIES (carry these forward):\n{json.dumps(known_entities, indent=2)}\n"

        if updated_pain_log:
            block += (
                "\nUSER_UPDATED_PAIN_LOG_FOR_THIS_SESSION "
                "(manual update from DB; treat this as the authoritative pain data for this session):\n"
            )
            block += f"{json.dumps(updated_pain_log, indent=2)}\n"

        if pacing:
            block += f"\nPACING_SESSION:\n{json.dumps(pacing, indent=2)}\n"
        else:
            block += "\nPACING_SESSION: None (no active session)\n"

        today_checkin = self._get_today_daily_checkin()
        if today_checkin:
            block += "\nTODAYS_DAILY_CHECKIN:\n"
            block += f"{json.dumps(today_checkin, indent=2)}\n"

        if pending:
            block += f"\nPENDING_FOLLOWUPS ({len(pending)} outstanding):\n"
            for p in pending[:2]:  # show max 2
                block += f"  - [{p['record_type']}] {p['question']} (partial: {p['partial_data']})\n"
        else:
            block += "\nPENDING_FOLLOWUPS: None\n"

        if running_summary:
            block += f"\nRUNNING_SUMMARY (use to recover entities):\n{running_summary.summary}\n"

        return block

    def _get_today_daily_checkin(self) -> Optional[dict]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        doc = self.daily_checkins.find_one(
            {"user_id": self.user_id, "date": today},
            {"_id": 0, "user_id": 0}
        )
        if not doc:
            return None
        # Convert datetime fields to ISO strings for prompt safety.
        return {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in doc.items()
        }

    def get_updated_session_pain_log(self) -> Optional[dict]:
        log = self.pain_logs.find_one(
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "updated_by_user": True,
            },
            {
                "_id": 0,
                "session_id": 1,
                "pain_score": 1,
                "pain_label": 1,
                "body_area": 1,
                "activity": 1,
                "date": 1,
                "timestamp": 1,
                "updated_at": 1,
            },
            sort=[("updated_at", -1), ("timestamp", -1)],
        )
        if not log:
            return None
        return {key: self._json_safe_value(value) for key, value in log.items()}

    def _json_safe_value(self, value):
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    # ─────────────────────────────────────────
    # PARSE LLM RESPONSE SAFELY
    # ─────────────────────────────────────────
    def _parse_response(self, raw: str) -> dict:
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return self._normalize_response_payload(json.loads(clean.strip()))
        except Exception:
            return self._normalize_response_payload({
                "intent": "unclear",
                "confidence": 0.5,
                "entities": {},
                "response": raw,
                "response_labels": [],
                "follow_up_question": None,
                "requires_follow_up": False,
                "flare_mode_active": False,
                "pacing_action": None,
                "suggested_actions": [],
                "save_partial": False,
                "partial_record": None
            })

    def _normalize_response_payload(self, parsed: dict) -> dict:
        response = parsed.get("response") or ""
        labels = parsed.get("response_labels")
        activity_tips = parsed.get("activity_tips", [])

        if not isinstance(labels, list) or not labels:
            labels = self._response_to_labels(response, parsed.get("follow_up_question"))
        else:
            normalized = []
            for item in labels[:3]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "message").strip().lower().replace(" ", "_")
                text = str(item.get("text") or "").strip()
                if text:
                    normalized.append({"label": label, "text": text[:160]})
            labels = normalized or self._response_to_labels(response, parsed.get("follow_up_question"))

        # Include activity tips in response if they exist
        if activity_tips and response:
            response += "\n\nHere are some tips to help:\n"
            for i, tip in enumerate(activity_tips, 1):
                response += f"{i}. {tip}\n"
            response = response.rstrip()

        # Include suggested safe activities in response if they exist
        suggested_activities = parsed.get("suggested_safe_activities", [])
        if suggested_activities and response and not activity_tips:
            response += "\n\nHere are some gentle activities you could try:\n"
            for i, activity in enumerate(suggested_activities, 1):
                response += f"{i}. {activity}\n"
            response = response.rstrip()

        parsed["response"] = response
        parsed["response_labels"] = labels
        if not response and labels:
            parsed["response"] = " ".join(item["text"] for item in labels)
        self._ensure_guidance_count(parsed)
        return parsed

    def _ensure_guidance_count(self, parsed: dict) -> None:
        intent = parsed.get("intent")
        if parsed.get("flare_mode_active") or intent == "flare_mode":
            parsed["activity_tips"] = []
            parsed["suggested_safe_activities"] = []
            return

        entities = parsed.get("entities") or {}
        if intent in ["pacing_start", "pain_with_activity_intent"]:
            parsed["activity_tips"] = self._normalize_guidance_list(parsed.get("activity_tips"))
        else:
            parsed["activity_tips"] = []

        if intent == "pain_log" and not entities.get("activity"):
            parsed["suggested_safe_activities"] = self._normalize_guidance_list(parsed.get("suggested_safe_activities"))
        else:
            parsed["suggested_safe_activities"] = []

    def _normalize_guidance_list(self, current: object) -> list[str]:
        items = current if isinstance(current, list) else []
        normalized: list[str] = []
        seen = set()

        for item in items:
            text = str(item or "").strip()
            key = text.lower()
            if text and key not in seen:
                normalized.append(text)
                seen.add(key)
            if len(normalized) >= 5:
                break

        return normalized

    def _response_to_labels(self, response: str, follow_up_question: Optional[str] = None) -> list:
        text = str(response or "").strip()
        if not text:
            return []

        parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]
        labels = []
        label_order = ["acknowledgement", "guidance", "next_step"]
        for index, part in enumerate(parts[:3]):
            labels.append({
                "label": label_order[min(index, len(label_order) - 1)],
                "text": part[:160],
            })

        if follow_up_question and not any(item["text"] == follow_up_question for item in labels):
            labels = labels[:2]
            labels.append({"label": "question", "text": str(follow_up_question).strip()[:160]})

        return labels

    # ─────────────────────────────────────────
    # MERGE ENTITIES — never lose what we know
    # ─────────────────────────────────────────
    def _merge_entities(self, existing: dict, new_entities: dict) -> dict:
        merged = dict(existing)
        for key, value in new_entities.items():
            if value is not None:
                merged[key] = value
        return merged

    def _normalize_activity(self, activity: Optional[str]) -> str:
        if not activity:
            return "activity"
        return " ".join(str(activity).strip().lower().split()) or "activity"

    def _suggested_stop_minutes(self, activity: Optional[str]) -> int:
        normalized = self._normalize_activity(activity)
        history = self.get_pacing_history(normalized if normalized != "activity" else None)
        durations = [
            s.get("duration_before_pain")
            for s in history
            if isinstance(s.get("duration_before_pain"), int) and s.get("duration_before_pain") > 0
        ]
        if durations:
            return max(1, int((sum(durations) / len(durations)) * 0.75))
        return 18

    def _elapsed_pacing_minutes(self, pacing: Optional[dict]) -> int:
        if not pacing or not pacing.get("start_time"):
            return 0
        try:
            start = datetime.fromisoformat(pacing["start_time"])
            return max(0, int((datetime.now(timezone.utc) - start).total_seconds() / 60))
        except Exception:
            return 0

    def _elapsed_pacing_seconds(self, pacing: Optional[dict]) -> int:
        if not pacing or not pacing.get("start_time"):
            return 0
        try:
            start = datetime.fromisoformat(pacing["start_time"])
            return max(0, int((datetime.now(timezone.utc) - start).total_seconds()))
        except Exception:
            return 0

    def _load_open_pacing_session(self) -> Optional[PacingSession]:
        session = self.pacing_logs.find_one(
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "status": {"$in": ["active", "resting"]},
                "completed": False,
            },
            {"_id": 0},
            sort=[("start_time", -1)],
        )
        if session:
            return session

        chat_doc = self.chat_collection.find_one(
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "pacing_session.status": {"$in": ["active", "resting"]},
                "pacing_session.completed": False,
            },
            {"_id": 0, "pacing_session": 1},
            sort=[("timestamp", -1)],
        )
        return (chat_doc or {}).get("pacing_session")

    def _ensure_pacing_action(self, parsed: dict) -> dict:
        if parsed.get("pacing_action"):
            return parsed

        action_by_intent = {
            "pacing_start": "start",
            "pacing_rest": "rest",
            "pacing_resume": "resume",
            "pacing_stop": "stop",
        }
        action = action_by_intent.get(parsed.get("intent"))
        if action:
            parsed["pacing_action"] = action
        return parsed

    def _message_mentions_pain(self, user_message: str) -> bool:
        pain_terms = [
            "pain",
            "ache",
            "aching",
            "hurt",
            "hurts",
            "hurting",
            "sore",
            "soreness",
            "discomfort",
            "uncomfortable",
            "strain",
            "strained",
            "flare",
            "stiff",
            "stiffness",
            "burning",
            "sharp",
            "throbbing",
            "killing me",
        ]
        text = str(user_message or "").lower()
        return any(re.search(rf"\b{re.escape(term)}\b", text) for term in pain_terms)

    def _correct_activity_only_intent(self, parsed: dict, user_message: str) -> dict:
        if parsed.get("intent") != "pain_with_activity_intent":
            return parsed

        entities = parsed.get("entities") or {}
        has_pain_entity = any(
            entities.get(key) is not None
            for key in ["pain_score", "pain_label", "body_area"]
        )
        if not has_pain_entity and not self._message_mentions_pain(user_message):
            parsed["intent"] = "pacing_start"
            parsed["pacing_action"] = "start"

        return parsed

    def _apply_pacing_action_override(self, parsed: dict, pacing_action: Optional[str]) -> dict:
        if not pacing_action:
            return parsed

        overrides = {
            "start": ("pacing_start", "start"),
            "still_okay": ("pacing_update", None),
            "starting_to_feel_strain": ("pacing_update", "rest"),
            "need_a_pause": ("pacing_rest", "rest"),
            "rest": ("pacing_rest", "rest"),
            "resume": ("pacing_resume", "resume"),
            "stop": ("pacing_stop", "stop"),
        }
        intent, action = overrides.get(pacing_action, (None, None))
        if intent:
            parsed["intent"] = intent
            parsed["pacing_action"] = action
        return parsed

    def _build_pacing_ui(self, parsed: dict, pacing: Optional[dict]) -> Optional[dict]:
        intent = parsed.get("intent")
        action = parsed.get("pacing_action")
        entities = parsed.get("entities", {})
        is_start_flow = action == "start" or intent in ["pacing_start", "pain_with_activity_intent"]
        activity = (
            entities.get("activity")
            if is_start_flow and entities.get("activity")
            else (pacing or {}).get("activity") or entities.get("activity")
        )

        is_pacing_intent = intent in [
            "pacing_start",
            "pacing_update",
            "pacing_rest",
            "pacing_resume",
            "pacing_stop",
            "pain_with_activity_intent",
        ]
        if not is_pacing_intent and not pacing:
            return None

        suggested_minutes = (pacing or {}).get("suggested_stop_minutes") or self._suggested_stop_minutes(activity)
        status = (pacing or {}).get("status")
        duration = (pacing or {}).get("duration")
        elapsed_seconds = duration if status == "stopped" and duration is not None else self._elapsed_pacing_seconds(pacing)
        elapsed_minutes = int(elapsed_seconds / 60)
        active_dot = min(6, max(1, int((elapsed_minutes / max(suggested_minutes, 1)) * 6)))

        base = {
            "activity": activity,
            "suggested_stopping_point": {
                "minutes": suggested_minutes,
                "label": f"about {suggested_minutes} min",
                "source": "based_on_pattern" if pacing and pacing.get("suggested_stop_minutes") else "default",
            },
            "elapsed_minutes": elapsed_minutes,
            "elapsed_seconds": elapsed_seconds,
            "coach_message": parsed.get("response"),
            "coach_message_labels": parsed.get("response_labels", []),
            "activity_tips": parsed.get("activity_tips", []),
            "check_in_options": ["Still Okay", "Starting to feel strain", "Need a pause"],
            "check_in_actions": [
                {"label": "Still Okay", "pacing_action": "still_okay"},
                {"label": "Starting to feel strain", "pacing_action": "starting_to_feel_strain"},
                {"label": "Need a pause", "pacing_action": "need_a_pause"},
            ],
            "progress_dots": {"total": 6, "active": active_dot},
        }

        if is_start_flow:
            return {
                **base,
                "screen": "activity_confirmation",
                "title": "Got it - activity",
                "subtitle": "The goal is to stop a little before discomfort increases.",
                "primary_action": "Start pacing",
                "secondary_action": "Try again",
            }

        if action == "rest" or status == "resting":
            return {
                **base,
                "screen": "rest",
                "title": "That gives us a helpful guide",
                "subtitle": "Noticing this early is exactly the point of pacing.",
                "card_title": "Next time, we can stop a little sooner - before this point.",
                "card_body": "Over a few sessions, your limit becomes clearer and pacing feels more natural.",
                "primary_action": "Take a pause now",
                "secondary_action": "Stop activity here",
            }

        if action == "stop" or status == "stopped":
            return {
                **base,
                "screen": "completed",
                "title": "You did well today!",
                "subtitle": "That was a good stopping point. Rest gently now.",
                "summary": {
                    "activity": activity,
                    "duration": elapsed_seconds,
                    "duration_label": self._format_duration_mmss(elapsed_seconds),
                    "pain": entities.get("pain_label") or "Still feeling okay",
                },
                "primary_action": "Back to Home",
            }

        return {
            **base,
            "screen": "active",
            "title": "Stay within your limit.",
            "subtitle": "Stay gentle. I'm here with you.",
            "primary_action": "Still feeling good",
            "secondary_action": "Starting to feel strain",
        }

    # ─────────────────────────────────────────
    # HANDLE PACING STATE TRANSITIONS
    # ─────────────────────────────────────────
    def _handle_pacing_action(
        self,
        current_state: dict,
        parsed: dict,
        user_message: str
    ) -> Optional[PacingSession]:

        action = parsed.get("pacing_action")
        entities = parsed.get("entities", {})
        pacing: Optional[PacingSession] = current_state.get("pacing_session")
        now = datetime.now(timezone.utc).isoformat()

        if action in ["rest", "resume", "stop"] and not pacing:
            pacing = self._load_open_pacing_session()

        if action == "stop" and pacing and pacing.get("status") == "stopped":
            return pacing

        if action in ["rest", "resume"] and pacing and pacing.get("completed"):
            return pacing

        if action == "start" and pacing:
            existing_activity = self._normalize_activity(pacing.get("activity"))
            new_activity = self._normalize_activity(entities.get("activity"))
            session_is_open = pacing.get("status") in ["active", "resting"] and not pacing.get("completed")
            if session_is_open and (new_activity == "activity" or new_activity == existing_activity):
                return pacing

            if session_is_open:
                start = datetime.fromisoformat(pacing["start_time"])
                elapsed_seconds = int((datetime.now(timezone.utc) - start).total_seconds())
                pacing["status"] = "stopped"
                pacing["end_time"] = now
                pacing["duration"] = elapsed_seconds
                pacing["duration_label"] = self._format_duration_mmss(elapsed_seconds)
                pacing["completed"] = True
                self._save_pacing_session(pacing)

            pacing = None

        if action == "start":
            activity = entities.get("activity") or "unknown activity"
            new_session: PacingSession = {
                "activity": activity,
                "start_time": now,
                "end_time": None,
                "status": "active",
                "pain_events": [],
                "rest_start_time": None,
                "duration_before_pain": None,
                "duration": None,
                "suggested_stop_minutes": self._suggested_stop_minutes(activity),
                "completed": False
            }
            self._save_pacing_session(new_session)
            return new_session

        elif action == "rest" and pacing:
            # Calculate how long they paced before pain
            start = datetime.fromisoformat(pacing["start_time"])
            elapsed_minutes = int((datetime.now(timezone.utc) - start).total_seconds() / 60)
            pacing["status"] = "resting"
            pacing["rest_start_time"] = now
            pacing["duration_before_pain"] = elapsed_minutes
            pain_event = {
                "time": now,
                "pain_score": entities.get("pain_score"),
                "pain_label": entities.get("pain_label"),
                "elapsed_minutes": elapsed_minutes
            }
            pacing["pain_events"].append(pain_event)
            self._save_pacing_session(pacing)
            return pacing

        elif action == "resume" and pacing:
            pacing["status"] = "active"
            pacing["rest_start_time"] = None
            self._save_pacing_session(pacing)
            return pacing

        elif action == "stop" and pacing:
            start = datetime.fromisoformat(pacing["start_time"])
            elapsed_seconds = int((datetime.now(timezone.utc) - start).total_seconds())
            pacing["status"] = "stopped"
            pacing["end_time"] = now
            pacing["duration"] = elapsed_seconds
            pacing["duration_label"] = self._format_duration_mmss(elapsed_seconds)
            pacing["completed"] = True
            # Save to DB
            self._save_pacing_session(pacing)
            return pacing   # clear session

        return pacing   # no change

    # ─────────────────────────────────────────
    # SAVE RECORDS TO MONGO
    # ─────────────────────────────────────────
    def _save_pain_log(self, entities: dict, is_partial: bool = False):
        self.pain_logs.insert_one({
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pain_score": entities.get("pain_score"),
            "pain_label": entities.get("pain_label"),
            "body_area": entities.get("body_area"),
            "activity": entities.get("activity"),
            "is_partial": is_partial,
            "date":datetime.now().strftime("%Y-%m-%d")
        })

    def _save_intervention_log(self, entities: dict, is_partial: bool = False):
        self.intervention_logs.insert_one({
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intervention_type": entities.get("intervention_type"),
            "intervention_name": entities.get("intervention_name"),
            "dose": entities.get("dose"),
            "frequency": entities.get("frequency"),
            "is_partial": is_partial,
            "date":datetime.now().strftime("%Y-%m-%d")
        })

    def _save_pacing_session(self, session: PacingSession):
        suggested_minutes = session.get("suggested_stop_minutes")
        status = session.get("status")
        if session.get("completed") and session.get("end_time"):
            status = "stopped"
        duration_label = self._format_duration_mmss(session.get("duration"))
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "activity": session["activity"],
            "start_time": session["start_time"],
            "end_time": session.get("end_time"),
            "status": status,
            "pain_events": session["pain_events"],
            "duration_before_pain": session["duration_before_pain"],
            "duration": session.get("duration"),
            "duration_label": duration_label,
            "suggested_stop_minutes": suggested_minutes,
            "suggested_stopping_point": {
                "minutes": suggested_minutes,
                "label": f"about {suggested_minutes} min" if suggested_minutes else None,
            },
            "completed": session["completed"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "date":datetime.now().strftime("%Y-%m-%d")
        }
        self.pacing_logs.update_one(
            {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "start_time": session["start_time"],
            },
            {"$set": payload},
            upsert=True,
        )

    def _save_partial_log(self, record_type: str, partial_data: dict, question: str):
        followup: PendingFollowUp = {
            "record_type": record_type,
            "partial_data": partial_data,
            "question": question,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "date":datetime.now().strftime("%Y-%m-%d")
        }
        self.partial_logs.insert_one({
            "user_id": self.user_id,
            "session_id": self.session_id,
            **followup
        })
        return followup

    def _save_chat_history(
        self,
        human_message: str,
        ai_message: str,
        parsed: dict,
        pacing_session: Optional[dict] = None,
        pacing_ui: Optional[dict] = None,
        chat_mode: Optional[str] = None,
        routed_to: Optional[str] = None
    ):
        self.chat_collection.insert_one({
            "user_id": self.user_id,
            "session_id": self.session_id,
            "type": chat_mode,
            "routed_to": routed_to,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "human_message": human_message,
            "ai_message": json.dumps(parsed),
            "raw_ai_message": ai_message,
            "coach_response": parsed.get("response"),
            "response_labels": parsed.get("response_labels", []),
            "coach_payload": parsed,
            "intent": parsed.get("intent"),
            "confidence": parsed.get("confidence"),
            "entities": parsed.get("entities", {}),
            "activity_tips": parsed.get("activity_tips", []),
            "suggested_safe_activities": parsed.get("suggested_safe_activities", []),
            "follow_up_question": parsed.get("follow_up_question"),
            "requires_follow_up": parsed.get("requires_follow_up", False),
            "flare_mode_active": parsed.get("flare_mode_active", False),
            "pacing_action": parsed.get("pacing_action"),
            "suggested_actions": parsed.get("suggested_actions", []),
            "save_partial": parsed.get("save_partial", False),
            "partial_record": parsed.get("partial_record"),
            "pacing_session": pacing_session,
            "pacing_ui": pacing_ui,
        })

    # ─────────────────────────────────────────
    # PROCESS INTENT — route to correct save
    # ─────────────────────────────────────────
    def _process_intent(
        self,
        parsed: dict,
        current_state: dict
    ) -> tuple[Optional[PacingSession], list[PendingFollowUp]]:

        intent = parsed.get("intent")
        entities = parsed.get("entities", {})
        save_partial = parsed.get("save_partial", False)
        pending: list = list(current_state.get("pending_followups", []))

        pacing = self._handle_pacing_action(current_state, parsed, "")

        if intent == "pain_log" or intent == "flare_mode":
            if save_partial:
                followup = self._save_partial_log(
                    "pain_log",
                    entities,
                    parsed.get("follow_up_question", "Would you like to add more details?")
                )
                pending.append(followup)
            elif entities.get("pain_score") or entities.get("pain_label"):
                self._save_pain_log(entities, is_partial=False)

        elif intent == "intervention_log":
            if save_partial or not entities.get("intervention_name"):
                followup = self._save_partial_log(
                    "intervention_log",
                    entities,
                    parsed.get("follow_up_question", "What was the name of the medication or treatment?")
                )
                pending.append(followup)
            else:
                self._save_intervention_log(entities, is_partial=False)


        # Resolve pending followups when user provides missing data
        if pending and intent not in ["flare_mode", "unclear"]:
            oldest = pending[0]
            merged = {**oldest["partial_data"], **{k: v for k, v in entities.items() if v}}
            if oldest["record_type"] == "intervention_log" and merged.get("intervention_name"):
                self._save_intervention_log(merged, is_partial=False)
                pending.pop(0)
            elif oldest["record_type"] == "pain_log" and (merged.get("pain_score") or merged.get("pain_label")):
                self._save_pain_log(merged, is_partial=False)
                pending.pop(0)

        return pacing, pending

    # ─────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        return self.chat_with_payload(user_message).get("response")

    def chat_with_payload(
        self,
        user_message: str,
        pacing_action: Optional[str] = None,
        chat_mode: Optional[str] = None
    ) -> dict:
        # Get current graph state
        current_state = self.graph.get_state(self.config).values

        known_entities = current_state.get("known_entities", {})
        pacing_session = current_state.get("pacing_session")
        pending_followups = current_state.get("pending_followups", [])
        flare_mode = current_state.get("flare_mode", False)

        if chat_mode == "flare_mode":
            flare_mode = True
        elif chat_mode == "pacing_mode":
            flare_mode = False

        # Check for flare mode keywords
        flare_triggers = ["flare", "can't cope", "cant cope", "unbearable", "crisis", "too much", "overwhelmed"]
        if chat_mode != "pacing_mode" and any(t in user_message.lower() for t in flare_triggers):
            flare_mode = True
        routed_to = "flare_prompt" if flare_mode else "main_prompt"
        print(
            f"[COACH_ROUTING] session_id={self.session_id} user_id={self.user_id} "
            f"chat_mode={chat_mode!r} routed_to={routed_to} flare_mode={flare_mode}"
        )
       

        # Invoke graph
        response = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "pacing_session": pacing_session,
                "pending_followups": pending_followups,
                "known_entities": known_entities,
                "flare_mode": flare_mode,
                "chat_mode": chat_mode
            },
            config=self.config
        )

        raw = response["messages"][-1].content
        parsed = self._parse_response(raw)
        parsed = self._correct_activity_only_intent(parsed, user_message)
        parsed = self._ensure_pacing_action(parsed)
        parsed = self._apply_pacing_action_override(parsed, pacing_action)
        self._ensure_guidance_count(parsed)

        

        # Merge and persist entities
        new_entities = parsed.get("entities", {})
        merged_entities = self._merge_entities(known_entities, new_entities)

        # Handle intent, pacing state transitions, partial logs
        new_pacing, new_pending = self._process_intent(
            parsed,
            {
                "pacing_session": pacing_session,
                "pending_followups": pending_followups
            }
        )

        # Deactivate flare mode if user seems calmer
        if flare_mode and parsed.get("intent") not in ["flare_mode", "unclear"]:
            flare_mode = False



        # Update graph state with new values
        self.graph.update_state(
            self.config,
            {
                "known_entities": merged_entities,
                "pacing_session": new_pacing,
                "pending_followups": new_pending,
                "flare_mode": flare_mode
            }
        )


        pacing_ui = self._build_pacing_ui(parsed, new_pacing)

        # Save the full response shape so API history and Mongo stay in sync.
        self._save_chat_history(
            user_message,
            raw,
            parsed,
            new_pacing,
            pacing_ui,
            chat_mode=chat_mode,
            routed_to=routed_to
        )

        return {
            "response": parsed.get("response", raw),
            "raw": raw,
            "parsed": parsed,
            "pacing_session": new_pacing,
            "pacing_ui": pacing_ui,
            "routed_to": routed_to,
        }

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    def get_session_summary(self) -> str:
        state = self.graph.get_state(self.config)
        context = state.values.get("context", {})
        summary_obj = context.get("running_summary")
        if summary_obj:
            return summary_obj.summary
        return "No summary available yet."

    # ─────────────────────────────────────────
    # PACING HISTORY — for personalized suggestions
    # ─────────────────────────────────────────
    def get_pacing_history(self, activity: str = None) -> list:
        query = {"user_id": self.user_id}
        if activity:
            query["activity"] = {"$regex": activity, "$options": "i"}
        return list(self.pacing_logs.find(query, {"_id": 0}).sort("start_time", -1).limit(10))

    def get_pacing_suggestion(self, activity: str) -> Optional[str]:
        """
        Look at past pacing sessions for this activity and suggest
        a safer duration based on when pain usually triggered.
        """
        history = self.get_pacing_history(activity)
        if not history:
            return None

        durations = [
            s["duration_before_pain"]
            for s in history
            if s.get("duration_before_pain")
        ]
        if not durations:
            return None

        avg = sum(durations) / len(durations)
        suggested = max(1, int(avg * 0.75))  # suggest 75% of average pain threshold
        return (
            f"Last time you did {activity}, pain usually started around {int(avg)} minutes. "
            f"Would you like to try a shorter session of about {suggested} minutes today?"
        )

    # ─────────────────────────────────────────
    # PAIN LOG HISTORY
    # ─────────────────────────────────────────
    def get_pain_history(self, limit: int = 30) -> list:
        return list(
            self.pain_logs.find(
                {"user_id": self.user_id},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
        )

    # ─────────────────────────────────────────
    # INTERVENTION HISTORY
    # ─────────────────────────────────────────
    def get_intervention_history(self, limit: int = 20) -> list:
        return list(
            self.intervention_logs.find(
                {"user_id": self.user_id},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
        )

    # ─────────────────────────────────────────
    # PENDING FOLLOWUPS
    # ─────────────────────────────────────────
    def get_history_timeline(
        self,
        limit: int = 100,
        days: int = 14,
        chat_mode: Optional[str] = None
    ) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))

        if chat_mode in ["flare_mode", "pacing_mode"]:
            chat_logs = list(
                self.chat_collection.find(
                    {"user_id": self.user_id, "type": chat_mode},
                    {"_id": 0}
                ).sort("timestamp", -1).limit(limit)
            )
            recent_chat_logs = self._recent_history_logs(chat_logs, cutoff, "timestamp")
            items = [
                self._build_chat_history_item(log)
                for log in recent_chat_logs
            ]
            items = [item for item in items if item]
            items.sort(key=lambda item: item["sort_timestamp"], reverse=True)

            return {
                "type": chat_mode,
                "insight_card": None,
                "insights": [],
                "filters": ["Flare"] if chat_mode == "flare_mode" else ["Activity"],
                "sections": self._group_history_items(items),
            }

        pain_logs = list(
            self.pain_logs.find(
                {"user_id": self.user_id},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
        )
        intervention_logs = list(
            self.intervention_logs.find(
                {"user_id": self.user_id},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
        )
        pacing_logs = list(
            self.pacing_logs.find(
                {"user_id": self.user_id},
                {"_id": 0}
            ).sort("start_time", -1).limit(limit)
        )
        chat_logs = list(
            self.chat_collection.find(
                {
                    "user_id": self.user_id,
                    "intent": {"$in": ["daily_checkin", "flare_mode"]},
                },
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
        )

        recent_pain_logs = self._recent_history_logs(pain_logs, cutoff, "timestamp")
        recent_intervention_logs = self._recent_history_logs(intervention_logs, cutoff, "timestamp")
        recent_pacing_logs = self._recent_history_logs(pacing_logs, cutoff, "start_time", "saved_at")
        recent_chat_logs = self._recent_history_logs(chat_logs, cutoff, "timestamp")

        items = []
        items.extend(self._build_pain_history_item(log) for log in recent_pain_logs)
        items.extend(self._build_intervention_history_item(log) for log in recent_intervention_logs)
        items.extend(self._build_pacing_history_item(log) for log in recent_pacing_logs)
        items.extend(self._build_chat_history_item(log) for log in recent_chat_logs)
        items = [item for item in items if item]
        items.sort(key=lambda item: item["sort_timestamp"], reverse=True)

        insights = self._detect_history_insights(
            recent_pain_logs,
            recent_pacing_logs,
            recent_intervention_logs
        )

        return {
            "insight_card": insights[0] if insights else None,
            "insights": insights,
            "filters": ["All", "Pain", "Meds", "Activity", "Flare"],
            "sections": self._group_history_items(items),
        }

    def _recent_history_logs(self, logs: list, cutoff: datetime, *timestamp_fields: str) -> list:
        recent = []
        for log in logs:
            dt = None
            for field in timestamp_fields:
                dt = self._history_dt(log.get(field))
                if dt:
                    break
            if dt and dt >= cutoff:
                recent.append(log)
        return recent

    def _history_dt(self, value) -> Optional[datetime]:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None

    def _history_time_label(self, value) -> Optional[str]:
        dt = self._history_dt(value)
        if not dt:
            return None
        return dt.strftime("%I:%M %p").lstrip("0")

    def _history_sort_timestamp(self, value) -> str:
        dt = self._history_dt(value)
        return dt.isoformat() if dt else ""

    def _history_timestamp_value(self, value) -> Optional[str]:
        dt = self._history_dt(value)
        if dt:
            return dt.isoformat()
        return str(value) if value else None

    def _checkin_time_label(self, timestamp_value) -> str:
        dt = self._history_dt(timestamp_value)
        if not dt:
            return "Daily"
        hour = dt.hour
        if 5 <= hour < 12:
            return "Morning"
        if 12 <= hour < 17:
            return "Afternoon"
        if 17 <= hour < 22:
            return "Evening"
        return "Night"

    def _format_duration_mmss(self, duration_seconds: Optional[int]) -> Optional[str]:
        if not isinstance(duration_seconds, int) or duration_seconds < 0:
            return None
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _pain_severity(self, score) -> str:
        if not isinstance(score, int):
            return "unknown"
        if score >= 8:
            return "high"
        if score >= 5:
            return "moderate"
        return "low"

    def _join_non_empty(self, parts: list) -> str:
        return ", ".join(str(part).strip() for part in parts if part)

    def _build_pain_history_item(self, log: dict) -> Optional[dict]:
        timestamp = log.get("timestamp")
        score = log.get("pain_score")
        body_area = log.get("body_area")
        pain_label = log.get("pain_label")

        return {
            "id": f"pain:{timestamp}:{body_area}:{score}",
            "type": "pain",
            "filter": "Pain",
            "icon": "pain",
            "title": f"Pain {score} / 10" if score is not None else "Pain logged",
            "subtitle": self._join_non_empty([body_area, pain_label]),
            "time": self._history_time_label(timestamp),
            "timestamp": self._history_timestamp_value(timestamp),
            "sort_timestamp": self._history_sort_timestamp(timestamp),
            "severity": self._pain_severity(score),
            "entities": {
                "pain_score": score,
                "pain_label": pain_label,
                "body_area": body_area,
                "activity": log.get("activity"),
            },
        }

    def _build_intervention_history_item(self, log: dict) -> Optional[dict]:
        timestamp = log.get("timestamp")
        name = log.get("intervention_name")
        dose = log.get("dose")

        return {
            "id": f"meds:{timestamp}:{name}",
            "type": "medication",
            "filter": "Meds",
            "icon": "medication",
            "title": "Medication",
            "subtitle": self._join_non_empty([name, dose]) or log.get("intervention_type") or "Treatment logged",
            "time": self._history_time_label(timestamp),
            "timestamp": self._history_timestamp_value(timestamp),
            "sort_timestamp": self._history_sort_timestamp(timestamp),
            "severity": "neutral",
            "entities": {
                "intervention_type": log.get("intervention_type"),
                "intervention_name": name,
                "dose": dose,
                "frequency": log.get("frequency"),
            },
        }

    def _build_pacing_history_item(self, log: dict) -> Optional[dict]:
        timestamp = log.get("start_time") or log.get("saved_at")
        activity = log.get("activity") or "Activity"
        duration = log.get("duration")
        duration_minutes = max(1, round(duration / 60)) if isinstance(duration, int) else None
        duration_label = self._format_duration_mmss(duration)

        subtitle_parts = []
        if duration_minutes:
            subtitle_parts.append(f"{duration_minutes} min")
        elif log.get("suggested_stop_minutes"):
            subtitle_parts.append(f"{log.get('suggested_stop_minutes')} min target")
        subtitle_parts.append(log.get("status") if log.get("status") in ["active", "resting"] else "guided")

        return {
            "id": f"activity:{timestamp}:{activity}",
            "type": "activity",
            "filter": "Activity",
            "icon": "activity",
            "title": str(activity).title(),
            "subtitle": " - ".join(subtitle_parts),
            "time": self._history_time_label(timestamp),
            "timestamp": self._history_timestamp_value(timestamp),
            "sort_timestamp": self._history_sort_timestamp(timestamp),
            "severity": "neutral",
            "pacing": {
                "status": log.get("status"),
                "duration": duration,
                "duration_label": duration_label,
                "duration_minutes": duration_minutes,
                "duration_before_pain": log.get("duration_before_pain"),
                "suggested_stop_minutes": log.get("suggested_stop_minutes"),
            },
        }

    def _build_chat_history_item(self, log: dict) -> Optional[dict]:
        intent = log.get("intent")
        timestamp = log.get("timestamp")
        chat_mode = log.get("type")

        if chat_mode == "pacing_mode":
            entities = log.get("entities") or {}
            pacing_ui = log.get("pacing_ui") or {}
            activity = entities.get("activity") or pacing_ui.get("activity")
            return {
                "id": f"pacing:{timestamp}:{activity}",
                "type": "activity",
                "chat_mode": chat_mode,
                "filter": "Activity",
                "icon": "activity",
                "title": activity.title() if activity else "Pacing session",
                "subtitle": log.get("coach_response") or pacing_ui.get("subtitle") or "Pacing mode",
                "time": self._history_time_label(timestamp),
                "timestamp": self._history_timestamp_value(timestamp),
                "sort_timestamp": self._history_sort_timestamp(timestamp),
                "severity": "neutral",
                "entities": entities,
                "pacing_ui": pacing_ui,
            }

        if chat_mode == "flare_mode" or intent == "flare_mode":
            return {
                "id": f"flare:{timestamp}",
                "type": "flare",
                "chat_mode": chat_mode,
                "filter": "Flare",
                "icon": "flare",
                "title": "Flare support",
                "subtitle": log.get("coach_response") or "Comfort mode",
                "time": self._history_time_label(timestamp),
                "timestamp": self._history_timestamp_value(timestamp),
                "sort_timestamp": self._history_sort_timestamp(timestamp),
                "severity": "high",
            }

        if intent == "daily_checkin":
            title = log.get("title") or f"{self._checkin_time_label(timestamp)} check-in"
            return {
                "id": f"checkin:{timestamp}",
                "type": "checkin",
                "filter": "All",
                "icon": "checkin",
                "title": title,
                "subtitle": log.get("checkin_note") or log.get("human_message") or log.get("coach_response"),
                "time": self._history_time_label(timestamp),
                "timestamp": self._history_timestamp_value(timestamp),
                "sort_timestamp": self._history_sort_timestamp(timestamp),
                "severity": "neutral",
                "entities": log.get("entities") or {},
            }

        return None

    def _group_history_items(self, items: list) -> list:
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        groups = {}

        for item in items:
            dt = self._history_dt(item.get("timestamp"))
            if not dt:
                continue
            groups.setdefault(dt.date(), []).append(item)

        sections = []
        for date_key in sorted(groups.keys(), reverse=True):
            logs = sorted(groups[date_key], key=lambda item: item["sort_timestamp"], reverse=True)
            pain_scores = [
                item.get("entities", {}).get("pain_score")
                for item in logs
                if item.get("type") == "pain" and isinstance(item.get("entities", {}).get("pain_score"), int)
            ]
            avg_pain = round(sum(pain_scores) / len(pain_scores)) if pain_scores else None

            if date_key == today:
                label = "TODAY"
            elif date_key == yesterday:
                label = "YESTERDAY"
            else:
                label = date_key.strftime("%B %d").upper()

            sections.append({
                "date": str(date_key),
                "label": label,
                "avg_pain": avg_pain,
                "summary": self._history_section_summary(logs, avg_pain),
                "items": [
                    {key: value for key, value in item.items() if key != "sort_timestamp"}
                    for item in logs
                ],
            })

        return sections

    def _history_section_summary(self, logs: list, avg_pain: Optional[int]) -> str:
        pain_count = sum(1 for item in logs if item.get("type") == "pain")
        pacing_count = sum(1 for item in logs if item.get("type") == "activity")
        medication_count = sum(1 for item in logs if item.get("type") == "medication")

        parts = []
        if avg_pain is not None:
            parts.append(f"AVG PAIN {avg_pain}")
        if pain_count:
            parts.append(f"{pain_count} LOG" + ("" if pain_count == 1 else "S"))
        if pacing_count:
            parts.append(f"{pacing_count} PACING SESSION" + ("" if pacing_count == 1 else "S"))
        if medication_count:
            parts.append(f"{medication_count} MED" + ("" if medication_count == 1 else "S"))
        return " - ".join(parts) if parts else f"{len(logs)} ITEM" + ("" if len(logs) == 1 else "S")

    def _detect_history_insights(
        self,
        pain_logs: list,
        pacing_logs: list,
        intervention_logs: list
    ) -> list:
        insights = []
        scored_pain_logs = [
            log for log in pain_logs
            if isinstance(log.get("pain_score"), int) and self._history_dt(log.get("timestamp"))
        ]

        if len(scored_pain_logs) >= 3:
            morning_logs = [
                log for log in scored_pain_logs
                if self._history_dt(log.get("timestamp")).hour < 10
            ]
            if len(morning_logs) >= 2:
                morning_avg = sum(log["pain_score"] for log in morning_logs) / len(morning_logs)
                overall_avg = sum(log["pain_score"] for log in scored_pain_logs) / len(scored_pain_logs)
                if morning_avg >= overall_avg + 1:
                    insights.append({
                        "type": "pattern",
                        "priority": "medium",
                        "title": "Mornings seem more difficult - pacing may help",
                        "subtitle": "Your highest logs are before 10 AM most days.",
                        "action": "start_pacing",
                        "source": "painlogs",
                    })

        repeat_body_areas = {}
        for log in scored_pain_logs:
            body_area = log.get("body_area")
            if body_area:
                repeat_body_areas.setdefault(str(body_area).lower(), []).append(log)
        for body_area, logs in repeat_body_areas.items():
            if len(logs) >= 3:
                avg_score = round(sum(log["pain_score"] for log in logs) / len(logs))
                insights.append({
                    "type": "pattern",
                    "priority": "low",
                    "title": f"{body_area.title()} pain is showing up repeatedly",
                    "subtitle": f"Average recent score is {avg_score}/10 across {len(logs)} logs.",
                    "action": "review_pain_logs",
                    "source": "painlogs",
                })
                break

        completed_pacing = [
            log for log in pacing_logs
            if isinstance(log.get("duration_before_pain"), int) and log.get("duration_before_pain") > 0
        ]
        if len(completed_pacing) >= 2:
            avg_limit = round(sum(log["duration_before_pain"] for log in completed_pacing) / len(completed_pacing))
            suggested = max(1, int(avg_limit * 0.75))
            insights.append({
                "type": "pacing",
                "priority": "medium",
                "title": f"A {suggested} min pacing target may be safer",
                "subtitle": f"Pain has tended to appear around {avg_limit} minutes in recent sessions.",
                "action": "start_pacing",
                "source": "pacinglogs",
            })

        if intervention_logs and scored_pain_logs:
            insights.append({
                "type": "review",
                "priority": "low",
                "title": "Medication and pain logs are ready to compare",
                "subtitle": "Review recent meds beside pain scores to spot what may be helping.",
                "action": "review_meds",
                "source": "interventionlogs",
            })

        return insights[:3]

    def get_pending_followups(self) -> list:
        state = self.graph.get_state(self.config)
        return state.values.get("pending_followups", [])

    def get_chat_history(self, limit: int = 100) -> list:
        cursor = self.chat_collection.find(
            {"session_id": self.session_id, "user_id": self.user_id},
            {
                "_id": 0,
                "human_message": 1,
                "ai_message": 1,
                "timestamp": 1,
                "response_labels": 1,
                "pacing_ui": 1,
            }
        ).sort("timestamp", 1).limit(limit)
    
        history = []
        for doc in cursor:
            raw_ai = doc.get("ai_message", "")
            # ai_message is stored as a JSON string — parse out just the response field
            try:
                parsed = json.loads(raw_ai)
                ai_response = parsed.get("response", raw_ai)
                response_labels = doc.get("response_labels") or parsed.get("response_labels", [])
            except (json.JSONDecodeError, TypeError):
                # If it's already plain text (fallback case), use as-is
                ai_response = raw_ai
                response_labels = doc.get("response_labels") or self._response_to_labels(ai_response)
    
            history.append({
                "timestamp": doc.get("timestamp"),
                "human_message": doc.get("human_message"),
                "ai_response": ai_response,
                "response_labels": response_labels,
                "pacing_ui": doc.get("pacing_ui"),
            })
    
        return history

    
    
 



# ─────────────────────────────────────────────
# CLI — for local testing
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("─" * 50)
    print("  Chronic Pain Coach  ")
    print("─" * 50)
    print("Commands: 'summary', 'history', 'pacing history', 'pending', 'quit'\n")

    coach = Coach("user123", "session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Take care. 💙")
            break

        elif user_input.lower() == "summary":
            print("\n── Session Summary ──")
            print(coach.get_session_summary())
            print()

        elif user_input.lower() == "history":
            logs = coach.get_pain_history(limit=5)
            print("\n── Recent Pain Logs ──")
            for log in logs:
                print(f"  [{log['timestamp'][:10]}] Score: {log.get('pain_score')} | "
                      f"Area: {log.get('body_area')} | Partial: {log.get('is_partial')}")
            print()

        elif user_input.lower().startswith("pacing history"):
            parts = user_input.split(" ", 2)
            activity = parts[2] if len(parts) > 2 else None
            sessions = coach.get_pacing_history(activity)
            print("\n── Pacing Sessions ──")
            for s in sessions:
                print(f"  [{s['start_time'][:10]}] {s['activity']} | "
                      f"Pain after: {s.get('duration_before_pain')} min | Status: {s['status']}")
            print()

        elif user_input.lower() == "pending":
            followups = coach.get_pending_followups()
            if followups:
                print("\n── Pending Follow-ups ──")
                for f in followups:
                    print(f"  [{f['record_type']}] {f['question']}")
            else:
                print("\nNo pending follow-ups.")
            print()

        else:
            # Check for pacing suggestion before responding
            state = coach.graph.get_state(coach.config).values
            pacing = state.get("pacing_session")

            # If starting a new pacing session, offer suggestion from history
            if not pacing and any(
                kw in user_input.lower()
                for kw in ["start pacing", "pacing for", "about to start", "going to do"]
            ):
                # Extract rough activity name for history lookup
                for activity_kw in ["dishes", "laundry", "cooking", "walking", "cleaning", "shopping"]:
                    if activity_kw in user_input.lower():
                        suggestion = coach.get_pacing_suggestion(activity_kw)
                        if suggestion:
                            print(f"\n💡 Pacing tip: {suggestion}\n")
                        break

            response = coach.chat(user_input)
            print(f"\nCoach: {response}\n")

            
