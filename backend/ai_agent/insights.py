from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
import os
import json
import random
import time

load_dotenv()

# =========================
# DATABASE SETUP
# =========================

DB_URI = os.getenv("MONGODB_URI")

client = MongoClient(DB_URI)

db = client["chronic-pain-app"]

INSIGHTS_MODEL = os.getenv("MISTRAL_INSIGHTS_MODEL", "mistral-small-latest")
INSIGHTS_REQUEST_DELAY_SECONDS = float(os.getenv("INSIGHTS_REQUEST_DELAY_SECONDS", "2"))
INSIGHTS_MAX_RETRIES = int(os.getenv("INSIGHTS_MAX_RETRIES", "4"))
INSIGHTS_INITIAL_BACKOFF_SECONDS = float(os.getenv("INSIGHTS_INITIAL_BACKOFF_SECONDS", "5"))

# =========================
# AI PROMPT
# =========================

INSIGHTS_PROMPT = """
You are a compassionate chronic pain companion.

Your job is to generate SHORT, gentle, emotionally supportive observations
based on the user's recent activity, pacing, pain, and intervention data.

USER DATA:
{insights_data}

STYLE GUIDE:
- Write like a calm supportive wellness companion
- Keep each insight SHORT (1 sentence)
- Use soft language
- Sound human and warm
- Avoid medical or robotic wording
- Avoid repeating the word "pain" too much
- Avoid sounding analytical or diagnostic
- Insights should feel comforting and reflective

GOOD EXAMPLES:
- "Pain tends to increase after doing dishes (~12 minutes)."
- "You feel better on days with longer rest periods."
- "Medication seems to reduce discomfort slightly over 2 days."
- "Mornings are usually your hardest time of day."
- "Household activities seem to take more energy lately."
- "Short breaks appear to help you recover more comfortably."

BAD EXAMPLES:
- "There seems to be a pattern of needing to stop activities to manage severe pain with medication."
- "Pain in the lower back appears frequently during household activities."
- "The patient demonstrates worsening symptoms."

RULES:
- NEVER give medical advice
- NEVER diagnose
- NEVER mention exact timestamps
- NEVER mention UTC times
- Keep insights under 18 words when possible
- Make observations feel emotionally safe

OUTPUT:
Return ONLY valid JSON.

{{
  "insights": [
    {{
      "text": "Pain tends to increase after doing dishes (~12 minutes).",
      "category": "activity"
    }}
  ]
}}

"""

# =========================
# HELPERS
# =========================

def _get_time_of_day_label(timestamp_str: str) -> str:
    """
    Convert timestamp into safe time-of-day label.
    """

    if not timestamp_str:
        return "unknown"

    try:
        dt = datetime.fromisoformat(timestamp_str)

        hour = dt.hour

        if 5 <= hour < 12:
            return "morning"

        elif 12 <= hour < 17:
            return "afternoon"

        elif 17 <= hour < 21:
            return "evening"

        else:
            return "night"

    except Exception:
        return "unknown"


def _calculate_duration(start_time_str, end_time_str):
    """
    Calculate duration between two ISO timestamps.
    """

    if not start_time_str or not end_time_str:
        return None

    try:
        start_time = datetime.fromisoformat(start_time_str)
        end_time = datetime.fromisoformat(end_time_str)

        duration = end_time - start_time

        total_seconds = int(duration.total_seconds())

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return {
            "total_seconds": total_seconds,
            "minutes": round(total_seconds / 60, 2),
            "formatted": f"{hours}h {minutes}m {seconds}s"
        }

    except Exception:
        return None


# =========================
# INSIGHTS CLASS
# =========================

class Insights:

    def __init__(self, user_id: str):

        self.user_id = user_id

        self.intervention_logs = db["interventionlogs"]
        self.pain_logs = db["painlogs"]
        self.partial_logs = db["partiallogs"]
        self.chat_history = db["chathistory"]
        self.pacing_logs = db["pacinglogs"]
        self.insights_collection = db["insights"]

        self.llm = ChatMistralAI(
            model=INSIGHTS_MODEL,
            temperature=0.3
        )

    # =========================
    # PACING DATA
    # =========================

    def get_recent_pacing_logs(self, limit=5):

        pacing_logs = list(
            self.pacing_logs.find(
                {"user_id": self.user_id}
            ).sort("saved_at", -1).limit(limit)
        )

        cleaned_logs = []

        for log in pacing_logs:

            start_time = log.get("start_time")
            end_time = log.get("saved_at")

            duration_data = _calculate_duration(
                start_time,
                end_time
            )

            cleaned_logs.append({
                "activity": log.get("activity"),
                "status": log.get("status"),
                "completed": log.get("completed"),
                "pain_events": log.get("pain_events", []),
                "duration": duration_data,
                "time_of_day": _get_time_of_day_label(start_time),
                "start_time": start_time,
                "end_time": end_time,
                "raw_data": log
            })

        return cleaned_logs

    # =========================
    # ALL RECENT DATA
    # =========================

    def get_recent_insights_data(self):

        recent_interventions = list(
            self.intervention_logs.find(
                {"user_id": self.user_id}
            ).sort("timestamp", -1).limit(5)
        )

        recent_pain_logs = list(
            self.pain_logs.find(
                {"user_id": self.user_id}
            ).sort("timestamp", -1).limit(5)
        )

        recent_partial_logs = list(
            self.partial_logs.find(
                {"user_id": self.user_id}
            ).sort("timestamp", -1).limit(5)
        )

        recent_chat_logs = list(
            self.chat_history.find(
                {"user_id": self.user_id}
            ).sort("timestamp", -1).limit(5)
        )

        recent_pacing_logs = self.get_recent_pacing_logs()

        insights_summary = {

            "recent_interventions": [
                {
                    "intervention_name": log.get("intervention_name"),
                    "intervention_type": log.get("intervention_type"),
                    "dose": log.get("dose"),
                    "frequency": log.get("frequency")
                }
                for log in recent_interventions
            ],

            "recent_pain_levels": [
                {
                    "pain_score": log.get("pain_score"),
                    "pain_label": log.get("pain_label"),
                    "body_area": log.get("body_area"),
                    "activity": log.get("activity")
                }
                for log in recent_pain_logs
            ],

            "recent_partial_entries": [
                {
                    "record_type": log.get("record_type"),
                    "partial_data": log.get("partial_data")
                }
                for log in recent_partial_logs
            ],

            "recent_chat_messages": [
                {
                    "human_message": log.get("human_message"),
                    "intent": log.get("intent"),
                    "entities": log.get("entities", {}),
                    "time_of_day": _get_time_of_day_label(
                        log.get("timestamp")
                    )
                }
                for log in recent_chat_logs
            ],

            "recent_pacing_logs": [
                {
                    "activity": log.get("activity"),
                    "status": log.get("status"),
                    "completed": log.get("completed"),
                    "duration": log.get("duration"),
                    "pain_events": log.get("pain_events"),
                    "time_of_day": log.get("time_of_day")
                }
                for log in recent_pacing_logs
            ]
        }

        return insights_summary

    def _is_rate_limit_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return "429" in text or "rate limit" in text or "rate_limited" in text

    def _invoke_llm_with_retries(self, prompt: str):
        last_error = None

        for attempt in range(INSIGHTS_MAX_RETRIES + 1):
            try:
                return self.llm.invoke(prompt)
            except Exception as e:
                last_error = e
                if not self._is_rate_limit_error(e) or attempt >= INSIGHTS_MAX_RETRIES:
                    raise

                backoff = INSIGHTS_INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                jitter = random.uniform(0, 2)
                sleep_seconds = backoff + jitter
                print(
                    f"[RATE LIMIT] User {self.user_id} - waiting {sleep_seconds:.1f}s "
                    f"before retry {attempt + 1}/{INSIGHTS_MAX_RETRIES}"
                )
                time.sleep(sleep_seconds)

        raise last_error

    def _generate_rule_based_insights(self, raw_data: dict) -> dict:
        insights = []

        pain_logs = raw_data.get("recent_pain_levels", [])
        scored_logs = [
            log for log in pain_logs
            if isinstance(log.get("pain_score"), int)
        ]

        if scored_logs:
            body_area_counts = {}
            for log in scored_logs:
                body_area = log.get("body_area")
                if body_area:
                    body_area_counts[body_area] = body_area_counts.get(body_area, 0) + 1

            if body_area_counts:
                body_area = max(body_area_counts, key=body_area_counts.get)
                insights.append({
                    "text": f"{body_area.title()} discomfort has shown up more than once recently.",
                    "category": "pain"
                })

            avg_score = round(
                sum(log["pain_score"] for log in scored_logs) / len(scored_logs),
                1
            )
            if avg_score >= 6:
                insights.append({
                    "text": "Recent logs suggest gentler pacing may be helpful right now.",
                    "category": "pacing"
                })
            elif avg_score <= 4:
                insights.append({
                    "text": "Your recent entries look a little steadier.",
                    "category": "pain"
                })

        pacing_logs = raw_data.get("recent_pacing_logs", [])
        pacing_with_events = [
            log for log in pacing_logs
            if log.get("pain_events")
        ]
        if pacing_with_events:
            activity = pacing_with_events[0].get("activity") or "activity"
            insights.append({
                "text": f"{str(activity).title()} may need shorter, softer pacing.",
                "category": "activity"
            })

        interventions = raw_data.get("recent_interventions", [])
        if interventions and scored_logs:
            insights.append({
                "text": "Medication and comfort logs are ready to compare gently.",
                "category": "medication"
            })

        if not insights:
            insights.append({
                "text": "A few more logs will help clearer patterns appear.",
                "category": "general"
            })

        return {"insights": insights[:3], "fallback": True}

    # =========================
    # GENERATE AI INSIGHTS
    # =========================

    # def generate_insights(self):

    #     raw_data = self.get_recent_insights_data()

    #     prompt = INSIGHTS_PROMPT.format(
    #         insights_data=json.dumps(raw_data, default=str)
    #     )

    #     response = self.llm.invoke(prompt)

    #     try:

    #         result = json.loads(response.content)

    #     except json.JSONDecodeError:

    #         cleaned_response = (
    #             response.content
    #             .replace("```json", "")
    #             .replace("```", "")
    #             .strip()
    #         )

    #         result = json.loads(cleaned_response)

    #     # Save generated insights
    #     self.insights_collection.insert_one({
    #         "user_id": self.user_id,
    #         "insights": result.get("insights", []),
    #         "created_at": datetime.now(timezone.utc),
    #     })

    #     return result


    def generate_insights(self, session_id: str = None, allow_fallback: bool = True):

        # Skip if insights already generated for this session
        if session_id:
            existing = self.insights_collection.find_one({
                "user_id": self.user_id,
                "session_id": session_id
            })
            if existing:
                print(f"[SKIP] Insights already generated for session {session_id}")
                return {"insights": existing.get("insights", []), "skipped": True}

        raw_data = self.get_recent_insights_data()

        prompt = INSIGHTS_PROMPT.format(
            insights_data=json.dumps(raw_data, default=str)
        )

        try:
            response = self._invoke_llm_with_retries(prompt)

            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                cleaned_response = (
                    response.content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                result = json.loads(cleaned_response)
        except Exception as e:
            if not allow_fallback:
                raise
            print(f"[FALLBACK] User {self.user_id} session {session_id} - {str(e)}")
            result = self._generate_rule_based_insights(raw_data)

        # Save with session_id
        self.insights_collection.insert_one({
            "user_id": self.user_id,
            "session_id": session_id,
            "insights": result.get("insights", []),
            "fallback": result.get("fallback", False),
            "created_at": datetime.now(timezone.utc),
        })

        return result






# =========================
# BULK INSIGHTS GENERATION
# =========================

def generate_insights_for_all_users(
    max_sessions_per_user: int = None,
    delay_seconds: float = INSIGHTS_REQUEST_DELAY_SECONDS
):

    users_collection = db["users"]
    chat_collection = db["chathistory"]

    user_ids = users_collection.distinct("_id")

    if not user_ids:
        print("No users found.")
        return {"total": 0, "success": 0, "failed": 0, "errors": []}

    total = len(user_ids)
    success = 0
    failed = 0
    skipped = 0
    errors = []

    print(f"Generating insights for {total} users...")

    for user_id in user_ids:

        user_id_str = str(user_id)

        # Get all unique session IDs for this user from chat history
        session_ids = chat_collection.distinct(
            "session_id",
            {"user_id": user_id_str}
        )

        if not session_ids:
            print(f"[SKIP] User {user_id_str} — no sessions found")
            skipped += 1
            continue

        existing_session_ids = set(
            doc.get("session_id")
            for doc in db["insights"].find(
                {
                    "user_id": user_id_str,
                    "session_id": {"$in": session_ids}
                },
                {"_id": 0, "session_id": 1}
            )
            if doc.get("session_id")
        )

        pending_session_ids = [
            session_id
            for session_id in session_ids
            if session_id not in existing_session_ids
        ]

        if max_sessions_per_user:
            pending_session_ids = pending_session_ids[:max_sessions_per_user]

        for session_id in existing_session_ids:
            print(f"[SKIP] User {user_id_str} session {session_id} - already processed")
        skipped += len(session_ids) - len(pending_session_ids)

        insights_obj = Insights(user_id=user_id_str)

        for session_id in pending_session_ids:
            try:
                result = insights_obj.generate_insights(session_id=session_id)

                if result.get("skipped"):
                    print(f"[SKIP] User {user_id_str} session {session_id} — already processed")
                    skipped += 1
                else:
                    print(f"[OK] User {user_id_str} session {session_id} — {len(result.get('insights', []))} insights")
                    success += 1
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

            except Exception as e:
                print(f"[FAIL] User {user_id_str} session {session_id} — {str(e)}")
                failed += 1
                errors.append({
                    "user_id": user_id_str,
                    "session_id": session_id,
                    "error": str(e)
                })

    summary = {
        "total_users": total,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "errors": errors
    }

    print(f"\nDone. {success} generated, {skipped} skipped, {failed} failed.")
    return summary



# =========================
# USAGE
# =========================

if __name__ == "__main__":

    insights_obj = Insights(
        user_id="69f1d88dbd8f44b631f61f85"
    )

    # Get pacing logs
    pacing_data = insights_obj.get_recent_pacing_logs()

    print("\n===== PACING DATA =====\n")
    print(json.dumps(pacing_data, indent=2, default=str))

    # Generate AI insights
    insights = insights_obj.generate_insights()

    print("\n===== AI INSIGHTS =====\n")
    print(json.dumps(insights, indent=2))
