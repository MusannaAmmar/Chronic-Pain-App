from pydantic import BaseModel
from typing import Optional

class Login(BaseModel):
    email: str
    # password: str


class SignUp(BaseModel):
    email: str
    is_active: bool = False
    role: str = "user"





