import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import cv2
import tempfile
from email.mime.image import MIMEImage

load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

def send_detection_email(label, confidence, image=None):

    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("Email credentials are not set.")
        return
    
    subject = f"Object Detected: {label}"
    body = f"The object '{label}' was detected with confidence {confidence:.2f}."

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(body, "plain"))

    temp_filename = None
    if image is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_filename = tmp.name
            cv2.imwrite(temp_filename, image)
        with open(temp_filename, "rb") as f:
            img_data = f.read()
            part = MIMEImage(img_data, name=f"detected_{label}.jpg")
            msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
