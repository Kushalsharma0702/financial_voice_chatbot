# utils.py
import os
import random
import string
from datetime import datetime, timedelta
import logging
from twilio.rest import Client
from database import db_session, OTP # Import OTP model and db_session from database.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def generate_otp_code(length=6):
    """Generates a random N-digit OTP code."""
    return ''.join(random.choices(string.digits, k=length))

def send_sms_otp(to_number, account_sid, auth_token, messaging_service_sid):
    """Sends an OTP SMS via Twilio."""
    try:
        client = Client(account_sid, auth_token)
        otp_code = generate_otp_code()
        
        # Store OTP in database with expiry
        with db_session() as session: # Use the context manager
            expires_at = datetime.utcnow() + timedelta(minutes=5) # OTP valid for 5 minutes
            new_otp = OTP(phone_number=to_number, otp_code=otp_code, created_at=datetime.utcnow(), expires_at=expires_at)
            session.add(new_otp)
            # session.commit() is handled by db_session context manager
            logger.info(f"OTP {otp_code} stored for {to_number}, expires at {expires_at}")

        message = client.messages.create(
            messaging_service_sid=messaging_service_sid,
            to=to_number,
            body=f"Your OTP for financial bot is: {otp_code}. It is valid for 5 minutes. Do not share this with anyone."
        )
        logger.info(f"OTP sent to {to_number}: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Error sending OTP to {to_number}: {e}", exc_info=True)
        return False

def verify_otp_code(phone_number, user_entered_otp):
    """Verifies the user-entered OTP against the stored valid OTP."""
    with db_session() as session: # Use the context manager
        # Get the latest OTP for the phone number that has not expired
        latest_otp = session.query(OTP).filter(
            OTP.phone_number == phone_number,
            OTP.expires_at > datetime.utcnow()
        ).order_by(OTP.created_at.desc()).first()

        if latest_otp and latest_otp.otp_code == user_entered_otp:
            # Optionally, delete or invalidate the OTP after successful verification
            session.delete(latest_otp)
            # session.commit() is handled by db_session context manager
            logger.info(f"OTP {user_entered_otp} verified successfully for {phone_number}.")
            return True
        logger.warning(f"OTP verification failed for {phone_number}. Entered: {user_entered_otp}, Expected: {latest_otp.otp_code if latest_otp else 'None'} (or expired).")
        return False
#commit
def extract_digits_from_speech(speech_text):
    """
    Extracts contiguous digits from a speech transcription.
    Useful for account IDs and OTPs.
    """
    if not speech_text:
        return ""
    
    # Simple approach: find the first sequence of digits
    # For more robust parsing, consider LLM extraction or more complex regex
    digits = ''.join(filter(str.isdigit, speech_text))
    return digits

def hide_number(phone_number):
    """Hides parts of a phone number for privacy."""
    if not phone_number or len(phone_number) < 4:
        return phone_number
    return "X" * (len(phone_number) - 4) + phone_number[-4:]

