import os
from dataclasses import dataclass


@dataclass
class MonitorConfig:
    """
    Central configuration data class.
    All thresholds, paths, and credentials are managed here.
    In production, credentials should be set as environment variables.
    """

    # --- Detection Thresholds ---
    ear_threshold: float = 0.25         # Eye Aspect Ratio below this = eyes closed
    mar_threshold: float = 0.45         # Mouth Aspect Ratio above this = yawning
    
    # User Required: Trigger alarm only after X seconds of sustained state
    drowsy_time_secs: float = 3.0
    yawn_time_secs: float = 2.5
    distracted_time_secs: float = 5.0
    
    gaze_threshold_low: float = 0.42    # Iris ratio below this = looking left
    gaze_threshold_high: float = 0.58   # Iris ratio above this = looking right

    # --- Notification Credentials (Use environment variables in production) ---
    sender_email: str = os.getenv("SENDER_EMAIL", "mohansanjai1716@gmail.com")
    sender_password: str = os.getenv("SENDER_PASSWORD", "ynsnafpxrckfoybj")
    receiver_email: str = os.getenv("RECEIVER_EMAIL", "mohansanjai1716@gmail.com")
    receiver_sms: str = os.getenv("RECEIVER_SMS", "9514210203@vtext.com")
    email_rate_limit_secs: int = 60     # Minimum seconds between email/SMS alerts

    # --- File & Model Paths ---
    model_path: str = "D:/project/lung/Drowsiness_Detection/face_landmarker.task"
    log_csv: str = "alerts_log.csv"
    alerts_dir: str = "alerts"
