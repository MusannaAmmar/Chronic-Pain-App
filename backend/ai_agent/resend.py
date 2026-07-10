import resend
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette import status

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")



def send_otp_email(email: str, otp: str):
    if not resend.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RESEND_API_KEY is not configured",
        )
    if "\\" in resend.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RESEND_API_KEY contains invalid escape characters",
        )

    try:
        return resend.Emails.send({
            "from": os.getenv("RESEND_FROM_EMAIL"),
            "to": [email],
            "subject": "Your Chronic Pain App verification code",
            "html": f"""
                <div style="font-family: Arial, sans-serif; line-height: 1.5;">
                    <h2>Your verification code</h2>
                    <p>Use this OTP to continue:</p>
                    <p style="font-size: 28px; font-weight: bold; letter-spacing: 6px;">{otp}</p>
                    <p>This code expires in 60 seconds.</p>
                </div>
            """,
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to send OTP email: {str(e)}",
        )
