from typing import Optional

from pydantic import BaseModel


class PainLogs(BaseModel):
    pain_score: Optional[int] = None
    body_area: Optional[str] = None


class DailyCheckIn(BaseModel):
    text:str


class Access(BaseModel):
    microphone: bool
    notifications: bool
