from backend.auth.schema import SignUp,Login
from fastapi import APIRouter,Depends,HTTPException,Request,File,UploadFile,Form
from pymongo import MongoClient
import pyotp
import jwt
import os
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from starlette import status
from datetime import datetime , timedelta
from fastapi.responses import JSONResponse
import base64
from utils import get_current_user
from bson import ObjectId
from backend.ai_agent.resend import send_otp_email

load_dotenv()



DB_URI = os.getenv("MONGODB_URI")
client = MongoClient(DB_URI)
db = client["chronic-pain-app"]

router = APIRouter(prefix="/auth", tags=["Auth"])

db_user=db["users"]
# OTP_TTL_SECONDS = 180

security = HTTPBearer()



def base64_encode_image(image_data: bytes) -> str:
    try:
        base64_bytes = base64.b64encode(image_data)
        return base64_bytes.decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while encoding image")


def generate_otp():
    otp_secret = pyotp.random_base32()
    otp = pyotp.TOTP(otp_secret, digits=4).now()
    return otp



@router.post("/signup")
def signup_user(request: SignUp):
    try:
       
        otp=generate_otp()
        # otp="1122"

        # Check if user already exists with this email
        existing_user = db_user.find_one({"email": request.email})

        if existing_user:
            # === EXISTING USER ===

            db_user.update_one(
                {"email": request.email},
                {
                    "$set": {
                        "role": request.role,
                        "otp": otp,
                        "otp_created_at": datetime.utcnow(),
                    }
                }
            )
            send_otp_email(request.email, otp)

            return JSONResponse(
                content={
                    "id": str(existing_user["_id"]),
                    "message": f"OTP has been sent to {request.email}.",
                    "status": "otp_sent",
                    "is_new_user": False,
                    "is_active": existing_user.get("is_active", False)
                },
                status_code=status.HTTP_200_OK
            )

        # === NEW USER ===
        result = db_user.insert_one({
            "email": request.email,
            "otp": otp,
            "is_active": False,
            "role": request.role,
            "otp_created_at": datetime.utcnow(),
        })
        send_otp_email(request.email, otp)

        return JSONResponse(
            content={
                "id": str(result.inserted_id),
                "message": f"Signed up successfully. OTP sent to your email {request.email}.",
                "status": "pending_verification",
                "is_new_user": True,
                "is_active": False
            },
            status_code=status.HTTP_201_CREATED
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )
        




@router.post('/resend-otp')
def resend_otp(email: str):
    try:
        user = db_user.find_one({'email': email})

        if not user:
            return JSONResponse(
                content={"message": "No user found with this email"},
                status_code=status.HTTP_404_NOT_FOUND
            )

        
        # new_otp="1122"
        new_otp=generate_otp()


        # Update user with new OTP and timestamp
        db_user.update_one(
            {'email': email},
            {
                '$set': {
                    'otp': new_otp,
                    'otp_created_at': datetime.utcnow()
                }
            }
        )
        send_otp_email(email, new_otp)

        return JSONResponse(
            content={
                "message": f"New OTP has been sent to {email}",
                "status": "otp_sent",
                "is_active": user.get('is_active', False),
            },
            status_code=status.HTTP_200_OK
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@router.post('/otp-verification')
def otp_verification(email: str, otp: str):
    try:
        user = db_user.find_one({'email': email})

        if not user:
            return JSONResponse(
                content={"message": "No user found with this email"},
                status_code=status.HTTP_404_NOT_FOUND
            )

        stored_otp = user.get('otp')
        otp_created_at = user.get('otp_created_at')

        if not stored_otp or not otp_created_at:
            return JSONResponse(
                content={"message": "Please request a new OTP"},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        # Check 180 seconds expiration
        if datetime.utcnow() > otp_created_at + timedelta(seconds=180):
            return JSONResponse(
                content={
                    "message": "OTP has expired. Please click on Resend OTP.",
                    "expired": True
                },
                status_code=status.HTTP_400_BAD_REQUEST
            )

        if stored_otp != otp:
            return JSONResponse(
                content={"message": "Invalid OTP"},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        # Success - Verify user
        db_user.update_one(
            {'email': email},
            {
                '$set': {
                    'is_active': True,
                    'otp_verified_at': datetime.utcnow()
                }
            }
        )

        payload={
            'email':email,
            'exp':datetime.utcnow() + timedelta(hours=24)
        }
        token=jwt.encode(payload, key=os.getenv("SECRET_KEY"), algorithm='HS256')

        return JSONResponse(
            content={
                "message": "OTP Verified Successfully",
                "is_active": True,
                "access_token": token
            },
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.patch('/update-profile')
def update_user_details(img_file: UploadFile | None = File(None),
    language: str | None = Form(None),
    country: str | None = Form(None),
    # microphone_access:bool | None = Form(None),
    # notifications_enabled:bool | None = Form(None),
    fullname: str | None = Form(None),
    user: dict = Depends(get_current_user),):
    try:
        update_data = {}
        if img_file:
            image_data = img_file.file.read()
            base64_image = base64_encode_image(image_data)
            update_data['image'] = base64_image
        if language:
            update_data['language'] = language
        if country:
            update_data['country'] = country
        if fullname:
            update_data['fullname'] = fullname
        if not update_data:
            return JSONResponse(content={
                'message': 'No data provided for update'
            }, status_code=status.HTTP_400_BAD_REQUEST)

        db_user.update_one(
            {'_id': ObjectId(user["id"])},
            {'$set': update_data}
        )

        return JSONResponse(content={
            'message': 'Profile updated successfully'
        }, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={
            'error': str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    



@router.get('/profile')
def get_user_profile(user: dict = Depends(get_current_user)):
    try:
        user_data = db_user.find_one({'_id': ObjectId(user["id"])}, {'otp': 0, 'otp_created_at': 0, 'otp_verified_at': 0})
        if not user_data:
            return JSONResponse(content={
                'message': 'User not found'
            }, status_code=status.HTTP_404_NOT_FOUND)

        user_data['_id'] = str(user_data['_id'])
        return JSONResponse(content={
            'user': user_data,
        }, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={
            'error': str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)



@router.delete('/delete-account')
def delete_account(user: dict = Depends(get_current_user)):
    try:
        user_id = ObjectId(user["id"])
        user_id_str = str(user_id)
        
        # Delete from users collection using _id
        db['users'].delete_one({'_id': user_id})
        
        # List of other collections to delete user data from
        collections_to_clean = [
            'chathistory',
            'dailycheckins',
            'insights',
            'interventionlogs',
            'pacinglogs',
            'painlogs',
            'partiallogs',
        ]
        
        # Delete user records from all other collections using user_id reference (stored as string)
        for collection_name in collections_to_clean:
            collection = db[collection_name]
            collection.delete_many({'user_id': user_id_str})
        
        return JSONResponse(content={
            'message': 'Account and all related data deleted successfully'
        }, status_code=status.HTTP_200_OK)
    except Exception as e:
        return JSONResponse(content={
            'error': str(e)
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)