import datetime
import json

from annotated_types import doc
from fastapi import APIRouter, UploadFile, File,Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pymongo import MongoClient
# from whisper import load_model
from mistralai.client import Mistral
from backend.ai_agent.coach import Coach
import uuid
import os
import tempfile
from utils import get_current_user
from pydantic import BaseModel
from bson import ObjectId
import io
from mistralai.client import models

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from bson import json_util  # ← This is the key


DB_URI = os.getenv("MONGODB_URI")
client = MongoClient(DB_URI)
db=client['chronic-pain-app']
insights=db["insights"]

pacinglogs=db['pacinglogs']



mistral_client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY")
)

def serialize_doc(doc):
        doc["_id"] = str(doc["_id"])
        return doc

router = APIRouter(tags=['Coach'])









class UserInput(BaseModel):
    message: str
    session_id: str
    pacing_action: str | None = None
    type: str | None = None

PACING_ACTION_MESSAGES = {
    "start": "I'm starting pacing now.",
    "still_okay": "I'm still okay and can continue this pacing session.",
    "starting_to_feel_strain": "I'm starting to feel strain during this pacing session.",
    "need_a_pause": "I need a pause from this pacing session.",
    "rest": "I need a pause from this pacing session.",
    "resume": "I'm ready to continue the activity.",
    "stop": "Stop pacing, I'm done with the activity.",
}

PACING_ACTION_ALIASES = {
    "still_ok": "still_okay",
    "okay": "still_okay",
    "starting_to_strain": "starting_to_feel_strain",
    "feel_strain": "starting_to_feel_strain",
    "strain": "starting_to_feel_strain",
    "pause": "need_a_pause",
}

def _get_pacing_state(coach: Coach) -> dict | None:
    state = coach.graph.get_state(coach.config).values
    return state.get("pacing_session")


def _normalize_pacing_action(action: str | None) -> str | None:
    if not action:
        return None
    normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
    return PACING_ACTION_ALIASES.get(normalized, normalized)


def _normalize_chat_mode(mode: str | None) -> str | None:
    if not mode:
        return None
    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"pacing_mode", "flare_mode"}:
        return normalized
    return None


@router.post('/coach-chat')
def coach_chat(user_input: UserInput, user: dict = Depends(get_current_user)):
    user_id = user['id']
    coach = Coach(user_id, user_input.session_id)

    message = user_input.message
    chat_mode = _normalize_chat_mode(user_input.type)
    print(f"[COACH_CHAT] frontend_type={user_input.type!r} normalized_mode={chat_mode!r}")

    pacing_action = _normalize_pacing_action(user_input.pacing_action)
    print(f"[COACH_CHAT] frontend_pacing_action={user_input.pacing_action!r} normalized_pacing_action={pacing_action!r}")

    if pacing_action and pacing_action in PACING_ACTION_MESSAGES:
        message = PACING_ACTION_MESSAGES[pacing_action]

    coach_payload = coach.chat_with_payload(
        message,
        pacing_action=pacing_action,
        chat_mode=chat_mode
    )
    response = coach_payload["response"]
    chat_history = coach.get_chat_history()
    pacing_session = coach_payload.get("pacing_session") or _get_pacing_state(coach)
    print(
        "[COACH_CHAT] routed_to="
        f"{coach_payload.get('routed_to')} intent={(coach_payload.get('parsed') or {}).get('intent')!r} "
        f"flare_mode_active={(coach_payload.get('parsed') or {}).get('flare_mode_active')!r} "
        f"has_pacing_session={pacing_session is not None}"
    )

    pacing_completed = pacing_session is not None and pacing_session.get("status") == "stopped"


    return JSONResponse(content={
        "coach_response": response,
        "response_labels": (coach_payload.get("parsed") or {}).get("response_labels", []),
        "coach_payload": coach_payload.get("parsed"),
        "chat_history": chat_history,
        "pacing_session": pacing_session,  # active state OR summary data
        "pacing_ui": coach_payload.get("pacing_ui"),
        "pacing_completed": pacing_completed,  # boolean flag for frontend
        "chat_mode": chat_mode,
    })





@router.post("/transcribe")
async def transcribe_audio(
    session_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    try:
        audio_data = await file.read()
        # print("1. audio_data read, size:", len(audio_data))

        file_obj = models.File(
            file_name=file.filename or "audio.ogg",
            content=audio_data,
        )
        # print("2. file_obj created:", file_obj)

        transcription_response = mistral_client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file=file_obj
        )
        # print("3. transcription_response:", type(transcription_response), dir(transcription_response))

        transcription = transcription_response.text
        print("4. transcription:", transcription)

        user_id = user["id"]
        coach = Coach(user_id, session_id)
        coach_payload = coach.chat_with_payload(transcription)
        response = coach_payload["response"]
        chat_history = coach.get_chat_history()
        pacing_session = coach_payload.get("pacing_session") or _get_pacing_state(coach)
        pacing_completed = (
            pacing_session is not None
            and pacing_session.get("status") == "stopped"
        )

        return JSONResponse(content={
            "transcription": transcription,
            "coach_response": response,
            "response_labels": (coach_payload.get("parsed") or {}).get("response_labels", []),
            "coach_payload": coach_payload.get("parsed"),
            "chat_history": chat_history,
            "pacing_session": pacing_session,
            "pacing_ui": coach_payload.get("pacing_ui"),
            "pacing_completed": pacing_completed
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )




@router.get('/get-insights')
def get_insights(user: dict = Depends(get_current_user)):
    try:
        user_id = user['id']
        
        insights_summary = list(
            insights.find(
                {"user_id": user_id}, 
                {"_id": 0, "user_id": 0}
            ).sort("created_at", -1)   # Optional: newest first
        )

        if not insights_summary:
            return JSONResponse(
                content={"message": "No insights found for this user."}, 
                status_code=404
            )

        # Convert MongoDB documents (including datetime) to JSON-safe format
        safe_insights = json_util.dumps(insights_summary)
        safe_insights_dict = json.loads(safe_insights)

        return JSONResponse(
            content=safe_insights_dict, 
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get('/session-id')
def get_session_id(user:dict=Depends(get_current_user)):
    return {"session_id": str(uuid.uuid4())}








@router.get('/chat-history')
def get_chat_history(user: dict = Depends(get_current_user)):
    try:
        
        user_id = user['id']
        daily_checkin_cursor = db["dailycheckins"].find(
            {"user_id": user_id},
            {
                "_id": 0,
                "user_id": 0,
            }
        ).sort("timestamp", -1)
        saved_daily_checkins = list(daily_checkin_cursor)
        
        cursor = db["chathistory"].find(
            {"user_id": user_id},
            {
                "_id": 0,
                "timestamp": 1,
                "intent": 1,
                "entities": 1,
                "activity_tips": 1,
                "checkin_note": 1,
                "ai_message": 1,
                "coach_response": 1,
                "response_labels": 1,
                "pacing_ui": 1,
                "suggested_safe_activities": 1,
            }
        ).sort("timestamp", -1)

        items = []
        for doc in cursor:
            intent = doc.get("intent")

            # Skip noise
            if intent in ["unclear", "summary_request", None]:
                continue

            # Daily check-ins now come from the dedicated dailycheckins collection.
            if intent == "daily_checkin":
                continue


            ai_message_raw = doc.get("ai_message")

            activity_tips = []
            suggested_safe_activities = []
            response_labels = doc.get("response_labels") or []

            if isinstance(ai_message_raw, str):
                try:
                    ai_message = json.loads(ai_message_raw)
                except Exception:
                    ai_message = {}
            elif isinstance(ai_message_raw, dict):
                ai_message = ai_message_raw
            else:
                ai_message = {}

            if isinstance(ai_message, dict):
                activity_tips = ai_message.get("activity_tips") or []
                suggested_safe_activities = ai_message.get("suggested_safe_activities") or []
                suggested_actions=ai_message.get("suggested_actions") or []
                response_labels = response_labels or ai_message.get("response_labels") or []

            print(f"Activity tips: {activity_tips}, Suggested safe activities: {suggested_safe_activities}")

            items.append({
                "timestamp": doc.get("timestamp"),
                "intent": intent,
                "type": _map_intent_to_type(intent),
                "title": _build_title(intent, doc.get("entities", {})),
                "subtitle": _build_subtitle(intent, doc.get("entities", {}), doc),
                "guidance":activity_tips + suggested_safe_activities+suggested_actions ,
                "ai_response": doc.get("coach_response") or ai_message.get("response"),
                "response_labels": response_labels,
                "pacing_ui": doc.get("pacing_ui"),
                "checkin_note": doc.get("checkin_note"),
                "entities": doc.get("entities", {}),
            })

        for checkin in saved_daily_checkins:
            checkin_label = checkin.get("title") or f"{_checkin_time_label(checkin.get('timestamp'))} check-in"
            items.append({
                "timestamp": checkin.get("timestamp"),
                "intent": "daily_checkin",
                "type": "checkin",
                "title": checkin_label,
                "subtitle": checkin.get("text") or "",
                "guidance": [],
                "ai_response": "",
                "response_labels": [],
                "pacing_ui": None,
                "checkin_note": checkin.get("text"),
                "entities": {},
            })

        grouped = _group_by_date(items)
        return JSONResponse(
            content=jsonable_encoder({
                "chat_history": grouped,
                "daily_checkins": json.loads(json_util.dumps(saved_daily_checkins)),
            }),
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Helpers ──

def _map_intent_to_type(intent: str) -> str:
    return {
        "pain_log": "pain",
        "pain_with_activity_intent": "pain",
        "flare_mode": "flare",
        "intervention_log": "medication",
        "pacing_start": "activity",
        "pacing_stop": "activity",
        "pacing_update": "activity",
        "daily_checkin": "checkin",
    }.get(intent, "general")




def _build_title(intent: str, entities: dict) -> str:
    if intent in ["pain_log", "pain_with_activity_intent", "flare_mode"]:
        score = entities.get("pain_score")
        return f"Pain {score}/10" if score else "Pain logged"

    if intent == "intervention_log":
        name = entities.get("intervention_name")
        return name.title() if name else "Medication"

    if intent in ["pacing_start", "pacing_update"]:
        activity = entities.get("activity")
        return activity.title() if activity else "Activity"

    if intent == "daily_checkin":
        return "Check-in"

    return "Log"


def _build_subtitle(intent: str, entities: dict, doc: dict) -> str:
    if intent in ["pain_log", "pain_with_activity_intent", "flare_mode"]:
        parts = []
        if entities.get("body_area"):
            parts.append(entities["body_area"])
        if entities.get("pain_label"):
            parts.append(entities["pain_label"])
        return ", ".join(parts)

    if intent == "intervention_log":
        parts = []
        if entities.get("intervention_name"):
            parts.append(entities["intervention_name"])
        if entities.get("dose"):
            parts.append(entities["dose"])
        return " · ".join(parts)

    if intent in ["pacing_start", "pacing_update"]:
        activity = entities.get("activity", "")
        return f"{activity} session"

    if intent == "daily_checkin":
        return doc.get("checkin_note") or doc.get("human_message", "")

    return ""


def _group_by_date(items: list) -> list:
    

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    groups = defaultdict(list)
    ordered_items = sorted(
        items,
        key=lambda item: _parse_graph_dt(item.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    for item in ordered_items:
        try:
            ts = _parse_graph_dt(item.get("timestamp")).date()
        except Exception:
            continue
        groups[ts].append(item)

    result = []
    for date, logs in sorted(groups.items(), reverse=True):
        if date == today:
            label = "TODAY"
        elif date == yesterday:
            label = "YESTERDAY"
        else:
            label = date.strftime("%B %d")

        pain_scores = [
            l["entities"].get("pain_score")
            for l in logs
            if l.get("intent") in ["pain_log", "pain_with_activity_intent", "flare_mode"]
            and l.get("entities", {}).get("pain_score")
        ]
        avg_pain = round(sum(pain_scores) / len(pain_scores)) if pain_scores else None

        result.append({
            "date": str(date),
            "label": label,
            "avg_pain": avg_pain,
            "log_count": len(logs),
            "items": logs,
        })

    return result


def _checkin_time_label(timestamp_value) -> str:
    dt = _parse_graph_dt(timestamp_value)
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


GRAPH_WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

PAIN_LABEL_SCORE_MAP = {
    "unbearable": 10,
    "excruciating": 10,
    "severe": 8,
    "intense": 7,
    "intense pain": 7,
    "bad": 6,
    "rough": 6,
    "moderate": 5,
    "moderate pain": 5,
    "pain": 5,
    "neck pain": 5,
    "mild": 3,
    "slight": 2,
    "little": 2,
}

def _parse_graph_dt(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _graph_date_value(log: dict):
    date_value = log.get("date")
    if date_value:
        try:
            return datetime.fromisoformat(str(date_value)).date()
        except Exception:
            pass

    timestamp = _parse_graph_dt(log.get("timestamp") or log.get("saved_at") or log.get("start_time"))
    return timestamp.date() if timestamp else None


def _pain_score_for_graph(log: dict) -> int | None:
    score = log.get("pain_score")
    if isinstance(score, bool):
        return None
    if isinstance(score, int) and 0 <= score <= 10:
        return score
    if isinstance(score, float) and 0 <= score <= 10:
        return round(score)

    label = str(log.get("pain_label") or "").strip().lower()
    if not label:
        return None
    if label in PAIN_LABEL_SCORE_MAP:
        return PAIN_LABEL_SCORE_MAP[label]

    for keyword, mapped_score in PAIN_LABEL_SCORE_MAP.items():
        if keyword in label:
            return mapped_score
    return None


def _title_case_value(value) -> str | None:
    if not value:
        return None
    return str(value).replace("_", " ").strip().title()


def _graph_time_label(log: dict) -> str | None:
    timestamp = _parse_graph_dt(log.get("timestamp") or log.get("saved_at") or log.get("start_time"))
    if not timestamp:
        return None
    return timestamp.strftime("%I:%M %p").lstrip("0")


def _graph_marker_type_from_intervention(log: dict) -> str:
    intervention_type = str(log.get("intervention_type") or "").strip().lower()
    if intervention_type in {"therapy", "exercise"}:
        return intervention_type
    return "medication"


def _graph_marker_detail(marker_type: str, log: dict, score: int | None = None) -> dict:
    if marker_type == "exercise":
        activity = _title_case_value(log.get("activity") or log.get("intervention_name")) or "Exercise"
        title = f"Logged {activity}"
        subtitle_parts = []
        if score is not None:
            subtitle_parts.append(f"Pain {score}/10")
        if log.get("pain_label"):
            subtitle_parts.append(str(log.get("pain_label")).title())
        if _graph_time_label(log):
            subtitle_parts.append(_graph_time_label(log))
    else:
        name = _title_case_value(log.get("intervention_name")) or marker_type.title()
        title = f"Started {name}"
        subtitle_parts = [
            part for part in [
                _graph_time_label(log),
                log.get("dose"),
                log.get("frequency"),
            ]
            if part
        ]

    return {
        "type": marker_type,
        "title": title,
        "subtitle": " - ".join(str(part) for part in subtitle_parts),
        "raw": log,
    }


def _build_graph_payload(pain_logs: list, intervention_logs: list, days: int = 7) -> dict:
    today = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=(today.weekday() + 1) % 7)
    graph_dates = [week_start + timedelta(days=index) for index in range(days)]
    date_keys = {date_value: index for index, date_value in enumerate(graph_dates)}

    pain_by_date = defaultdict(list)
    markers_by_date = defaultdict(dict)

    for log in pain_logs:
        date_value = _graph_date_value(log)
        if date_value not in date_keys:
            continue

        score = _pain_score_for_graph(log)
        if score is not None:
            pain_by_date[date_value].append(score)

        activity = str(log.get("activity") or "").strip().lower()
        if activity:
            markers_by_date[date_value]["exercise"] = _graph_marker_detail("exercise", log, score)

    for log in intervention_logs:
        date_value = _graph_date_value(log)
        if date_value not in date_keys:
            continue

        marker_type = _graph_marker_type_from_intervention(log)
        markers_by_date[date_value][marker_type] = _graph_marker_detail(marker_type, log)

    points = []
    all_markers = []
    for index, date_value in enumerate(graph_dates):
        scores = pain_by_date.get(date_value, [])
        avg_score = round(sum(scores) / len(scores), 1) if scores else None
        day_markers = list(markers_by_date.get(date_value, {}).values())
        all_markers.extend(
            {
                **marker,
                "date": str(date_value),
                "day": GRAPH_WEEKDAY_LABELS[index],
                "x": index,
                "y": avg_score,
            }
            for marker in day_markers
        )
        points.append({
            "date": str(date_value),
            "day": GRAPH_WEEKDAY_LABELS[index],
            "x": index,
            "pain_score": avg_score,
            "score_source": "pain_score_or_label" if scores else None,
            "log_count": len(scores),
            "markers": day_markers,
        })

    scored_points = [point for point in points if point["pain_score"] is not None]
    highest_point = max(scored_points, key=lambda point: point["pain_score"], default=None)

    return {
        "scale": {"min": 0, "max": 10, "label": "PAIN - 1-10 scale"},
        "range": {
            "type": "weekly",
            "start_date": str(graph_dates[0]),
            "end_date": str(graph_dates[-1]),
        },
        "points": points,
        "markers": all_markers,
        "highest_point": highest_point,
        "selected_marker": all_markers[0] if all_markers else None,
        "insight_cta": "See what may be influencing this",
    }




@router.get('/get-graph')
def get_graph(user: dict = Depends(get_current_user)):
    try:
        user_id=user['id']
        pain_logs=list(db["painlogs"].find({"user_id": user_id}, {"_id": 0}).sort("timestamp", -1).limit(30))
        intervention_logs=list(db["interventionlogs"].find({"user_id": user_id}, {"_id": 0,'intervention_type': 1,
        'intervention_name':1,'dose':1,'frequency':1,'date':1,'timestamp':1}).sort("timestamp", -1).limit(30))
        graph = _build_graph_payload(pain_logs, intervention_logs)

        return JSONResponse(content={"graph": graph, "pain_logs": pain_logs, "intervention_logs": intervention_logs}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


@router.get('/coach-history')
def get_coach_history(
    limit: int = 100,
    days: int = 14,
    type: str | None = None,
    user: dict = Depends(get_current_user)
):
    try:
        chat_mode = _normalize_chat_mode(type)
        print(f"[COACH_HISTORY] frontend_type={type!r} normalized_mode={chat_mode!r}")
        coach = Coach(user["id"], "history")
        return JSONResponse(
            content=coach.get_history_timeline(limit=limit, days=days, chat_mode=chat_mode),
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get('/get-messages')
def get_messages(session_id: str, limit: int = 100, user: dict = Depends(get_current_user)):
    try:
        user_id = user['id']
        messages = db['chathistory']

        cursor = messages.find(
            {
                "user_id": user_id,
                "session_id": session_id,
                "type": {"$type": "null"},
            },
            {
                "_id": 0,
                "timestamp": 1,
                "human_message": 1,
                "ai_message": 1,
                "coach_response": 1,
                "type": 1,
            }
        ).sort("timestamp", 1).limit(limit)

        conversation = []
        for doc in cursor:
            ai_message_raw = doc.get("ai_message")
            ai_response = doc.get("coach_response")

            if not ai_response:
                if isinstance(ai_message_raw, str):
                    try:
                        parsed_ai_message = json.loads(ai_message_raw)
                        ai_response = parsed_ai_message.get("response")
                    except (json.JSONDecodeError, TypeError):
                        ai_response = ai_message_raw
                elif isinstance(ai_message_raw, dict):
                    ai_response = ai_message_raw.get("response")

            conversation.append({
                "timestamp": doc.get("timestamp"),
                "human_message": doc.get("human_message"),
                "ai_response": ai_response,
            })

        return JSONResponse(
            content={
                "user_id": user_id,
                "session_id": session_id,
                "conversation": conversation,
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
