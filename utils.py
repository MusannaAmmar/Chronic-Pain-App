from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import HTTPException, Depends
from starlette import status
from pymongo import MongoClient
from dotenv import load_dotenv
import os

import jwt

load_dotenv()


DB_URI = os.getenv("MONGODB_URI")
client = MongoClient(DB_URI)
db=client['chronic-pain-app']
db_user=db["users"]
security = HTTPBearer()



def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, os.getenv('SECRET_KEY'), algorithms=['HS256'])
        email = payload.get('email')
        if email is None:
            raise credentials_exception
        user = db_user.find_one(
            {'email': email},
            {'password': 0, 'otp': 0, 'otp_created_at': 0}
        )
        if user is None:
            raise credentials_exception
        if '_id' in user:
            user['id'] = str(user.pop('_id'))
        return user
    except Exception:
        raise credentials_exception
