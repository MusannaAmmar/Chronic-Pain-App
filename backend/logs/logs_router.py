import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from utils import get_current_user
from dotenv import load_dotenv
from backend.logs.schema import PainLogs,DailyCheckIn,Access
from starlette import status
from bson import ObjectId
from bson import json_util


load_dotenv()


DB_URI = os.getenv("MONGODB_URI")
client = MongoClient(DB_URI)
db = client["chronic-pain-app"]
db_history = db["chathistory"]
db_painlogs = db["painlogs"]
daily_checkins=db["dailycheckins"]
db_pacing = db["pacinglogs"]


router = APIRouter(prefix="/logs", tags=["Logs"])


def _checkin_time_label(timestamp_value: datetime) -> str:
    hour = timestamp_value.hour
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 22:
        return "Evening"
    return "Night"


def serialize_log(log):
    if not log:
        return None
    log["_id"] = str(log["_id"])
    if isinstance(log.get("timestamp"), datetime):
        log["timestamp"] = log["timestamp"].isoformat()
    return log


def get_latest_pain_log(user_id):
    return db_painlogs.find_one(
        {"user_id": user_id},
        sort=[("timestamp", -1), ("_id", -1)],
    )


def get_ai_pain_logs(user_id):
    try:
        latest_doc = get_latest_pain_log(user_id)

        if not latest_doc:
            return {
                    "message": "You need to tell the coach first about your pain.",
                    "user_id_used": user_id
                }
        latest_doc = serialize_log(latest_doc)

        return {
                "pain_score": latest_doc.get("pain_score"),
                "body_area": latest_doc.get("body_area"),
                "pain_label": latest_doc.get("pain_label"),
                "activity": latest_doc.get("activity"),
                "date": latest_doc.get("date"),
                "timestamp": latest_doc.get("timestamp"),
            }
            
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    
@router.get("/ai-pain-logs")
def get_pain_logs(user: dict = Depends(get_current_user)):
    user_id = user['id']
    ai_logs = get_ai_pain_logs(user_id)
    try:
        return JSONResponse(content={
            "ai_logs": ai_logs,
            
        },status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    





@router.patch("/pain-logs")
def update_pain_log(
    pain_log: PainLogs,           # You can also use a partial model if needed
    user: dict = Depends(get_current_user)
):
    try:
        existing_log = get_latest_pain_log(user['id'])
        
        if not existing_log:
            return JSONResponse(
                content={"error": "You need to tell AI Coach first about your pain before updating it."},
                status_code=status.HTTP_404_NOT_FOUND
            )

        update_data = pain_log.dict(exclude_none=True)
        if not update_data:
            return JSONResponse(
                content={"error": "No pain log fields provided for update"},
                status_code=status.HTTP_400_BAD_REQUEST
            )
        update_data["date"] = datetime.now().strftime("%Y-%m-%d")
        update_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        update_data["updated_by_user"] = True
        update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        result = db_painlogs.update_one(
            {"_id": existing_log["_id"], "user_id": user['id']},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            return JSONResponse(
                content={"error": "No changes made"},
                status_code=400
            )

        return JSONResponse(
            content={
                "message": "Latest pain log updated successfully",
                "pain_log": serialize_log(db_painlogs.find_one({"_id": existing_log["_id"]}))
            },
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )
    



@router.post("/daily-checkin")
def daily_checkin(
    checkin: DailyCheckIn,
    user: dict = Depends(get_current_user)
):
    try:
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        existing_checkin = daily_checkins.find_one({
            "user_id": user['id'],
            "date": today
        })

        if existing_checkin:
            return JSONResponse(
                content={"error": "You can only check in once per day"},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        checkin_data = {
            "user_id": user['id'],
            "text": checkin.text,
            "timestamp": now,
            "date": today,
            "title": f"{_checkin_time_label(now)} check-in",
        }

        daily_checkins.insert_one(checkin_data)

        return JSONResponse(
            content={"message": "Daily check-in logged successfully"},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )




@router.patch("/access")
def update_access(
    access: Access,
    user: dict = Depends(get_current_user)
):
    try:
        # Update the user's access settings in the database
        result = db.users.update_one(
            {"_id": ObjectId(user['id'])},
            {"$set": {
            "microphone": access.microphone,
            "notifications": access.notifications
                
            }}
        )

        if result.modified_count == 0:
            return JSONResponse(
                content={"error": "No changes made to access settings"},
                status_code=400
            )

        return JSONResponse(
            content={"message": "Access settings updated successfully"},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )




@router.get("/latest-chat-history")
def get_latest_chat_history(session_id: str, user: dict = Depends(get_current_user)):
    try:
        latest_doc = db_history.find_one(
            {"user_id": user['id'], "session_id": session_id},
            sort=[("timestamp", -1)]
        )

        if not latest_doc:
            return JSONResponse(content={"message": "No chat history found for the user."}, status_code=404)

        # Convert ObjectId to string for JSON serialization
        latest_doc["_id"] = str(latest_doc["_id"])
        return JSONResponse(content={"latest_chat_history": latest_doc}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@router.get("/activity-logs")
def activity_logs(session_id: str, user: dict = Depends(get_current_user)):
    try:
        latest_log = db_pacing.find_one(
            {"user_id": user['id'], "session_id": session_id},
            sort=[("saved_at", -1)]
        )

        if not latest_log:
            return JSONResponse(
                content={"message": "No activity logs found for the user."},
                status_code=status.HTTP_404_NOT_FOUND
            )

        safe_log = json.loads(json_util.dumps(latest_log))
        return JSONResponse(
            content={"activity_logs": [safe_log]},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

