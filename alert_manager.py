import csv
import logging
import os
import queue
import smtplib
import time
import winsound
from datetime import datetime
from email.message import EmailMessage
from threading import Thread, Lock

import cv2
import numpy as np
import pyttsx3

from config import MonitorConfig

logger = logging.getLogger(__name__)


class VoiceSpeaker:
    """
    Dedicated background thread that owns the pyttsx3 engine.
    pyttsx3 must be used from the thread that created it,
    so we queue messages and process them on a single worker thread.
    """

    # Voice messages per alert type
    MESSAGES = {
        "DROWSINESS": [
            "Warning. Drowsiness detected. Please stay alert.",
            "Alert. You appear to be falling asleep. Pull over safely.",
            "Critical warning. Driver is drowsy. Stop the vehicle now.",
        ],
        "YAWNING": [
            "You have yawned. Consider taking a break.",
            "Fatigue detected. Please rest soon.",
        ],
        "DISTRACTED": [
            "Warning. Eyes off road. Please focus on driving.",
            "Distraction detected. Keep your eyes on the road.",
        ],
    }

    def __init__(self):
        self._queue = queue.Queue()
        self._counts = {k: 0 for k in self.MESSAGES}
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, alert_type: str):
        """Queue a spoken message for the given alert type."""
        self._queue.put(alert_type)

    def _run(self):
        """Worker — owns the TTS engine and blocks on each utterance."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)    # words per minute
            engine.setProperty("volume", 1.0)  # 0.0–1.0
            # Prefer a female voice if available
            voices = engine.getProperty("voices")
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break

            while True:
                alert_type = self._queue.get()      # blocks until a message arrives
                messages = self.MESSAGES.get(alert_type, [f"{alert_type} alert."])
                idx = self._counts.get(alert_type, 0)
                text = messages[min(idx, len(messages) - 1)]
                self._counts[alert_type] = idx + 1

                engine.say(text)
                engine.runAndWait()
                self._queue.task_done()
        except Exception as e:
            logger.error(f"VoiceSpeaker thread crashed: {e}")


class AlertManager:
    """
    Manages all system alerts and side-effects:
      - Escalating audio alarm (3-level siren)
      - Voice TTS spoken alerts (pyttsx3, rate-limited)
      - Screenshot capture & organized file storage
      - CSV incident logging
      - Async email & SMS notifications (rate-limited)
    """

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alarm_active = False
        self.last_email_ts = 0.0
        self.last_screenshot_ts = 0.0
        self.last_voice_ts = 0.0
        self._lock = Lock()
        self._speaker = VoiceSpeaker()
        self._initialize_storage()

    # ------------------------------------------------------------------ #
    #  Storage Initialization
    # ------------------------------------------------------------------ #

    def _initialize_storage(self):
        """Creates alert subdirectories and CSV header on first run."""
        for folder in ["DROWSINESS", "YAWNING", "DISTRACTED"]:
            os.makedirs(os.path.join(self.config.alerts_dir, folder), exist_ok=True)

        if not os.path.isfile(self.config.log_csv):
            try:
                with open(self.config.log_csv, mode='w', newline='') as f:
                    csv.writer(f).writerow(["Timestamp", "Alert_Type", "Image_File"])
                logger.info(f"Created alert log: {self.config.log_csv}")
            except Exception as e:
                logger.error(f"Failed to initialize CSV log: {e}")

    # ------------------------------------------------------------------ #
    #  Incident Logging
    # ------------------------------------------------------------------ #

    def log_incident(self, alert_type: str, frame: np.ndarray):
        """
        Saves a screenshot of the incident and appends a row to the CSV log.
        Returns the relative path to the image for web linking.
        """
        current_time = time.time()
        with self._lock:
            if current_time - self.last_screenshot_ts < 2.0:
                return None
            self.last_screenshot_ts = current_time

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{alert_type}_{timestamp_str}.jpg"
        rel_path = f"{alert_type}/{filename}"   # e.g. DROWSINESS/DROWSINESS_2026.jpg
        full_path = os.path.join(self.config.alerts_dir, alert_type, filename)
        
        # Burn Timestamp & Alert Type onto the image for permanent record
        frame_copy = frame.copy()
        ts_text = f"{time.ctime()} | ALERT: {alert_type}"
        cv2.putText(frame_copy, ts_text, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA) # Outline
        cv2.putText(frame_copy, ts_text, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # Text
        
        cv2.imwrite(full_path, frame_copy)

        try:
            with open(self.config.log_csv, mode='a', newline='') as f:
                csv.writer(f).writerow([time.ctime(), alert_type, rel_path])
        except Exception as e:
            logger.warning(f"Logging failed: {e}")
            
        return rel_path

    # ------------------------------------------------------------------ #
    #  Remote Notifications
    # ------------------------------------------------------------------ #

    def speak_alert(self, alert_type: str):
        """
        Dispatches a spoken TTS voice alert.
        Rate-limited independently from email — fires at most once every 8 seconds.
        """
        current_time = time.time()
        with self._lock:
            if current_time - self.last_voice_ts < 8.0:
                return
            self.last_voice_ts = current_time
        self._speaker.speak(alert_type)

    def dispatch_notification_async(self, alert_type: str):
        """
        Fires an email and SMS notification asynchronously.
        Rate-limited to once every `email_rate_limit_secs` seconds.
        """
        current_time = time.time()
        with self._lock:
            if current_time - self.last_email_ts < self.config.email_rate_limit_secs:
                return
            self.last_email_ts = current_time

        Thread(target=self._send_email_payload, args=(alert_type,), daemon=True).start()

    def _send_email_payload(self, alert_type: str):
        """Internal worker that sends email/SMS over SSL SMTP."""
        try:
            logger.info(f"Dispatching remote {alert_type} notification...")
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.config.sender_email, self.config.sender_password)

            for target in [self.config.receiver_email, self.config.receiver_sms]:
                if "@" in target and "1234567890" not in target:
                    msg = EmailMessage()
                    msg.set_content(
                        f"URGENT: Driver monitoring triggered a {alert_type} alert at {time.ctime()}!"
                    )
                    msg['Subject'] = f'Driver Alert: {alert_type} Detected'
                    msg['From'] = self.config.sender_email
                    msg['To'] = target
                    server.send_message(msg)

            server.quit()
            logger.info("Remote notification delivered successfully.")
        except Exception as e:
            logger.error(f"Notification dispatch failed: {e}")

    # ------------------------------------------------------------------ #
    #  Audio Alarm
    # ------------------------------------------------------------------ #

    def engage_audio_alarm(self):
        """Engages the escalating 3-level audio alarm in a background thread."""
        with self._lock:
            if self.alarm_active:
                return
            self.alarm_active = True
        Thread(target=self._sound_sequence, daemon=True).start()

    def disengage_audio_alarm(self):
        """Signals the alarm thread to stop and cleans up any looping sound."""
        with self._lock:
            self.alarm_active = False

    def _sound_sequence(self):
        """
        Three-level escalating alarm sequence:
          Level 1 (0-3s):  Low-pitched slow beeps
          Level 2 (3-6s):  High-pitched fast beeps
          Level 3 (6s+):   Continuous Windows critical system alarm
        """
        start_time = time.time()
        critical_loop_active = False

        while self.alarm_active:
            elapsed = time.time() - start_time

            if elapsed < 3.0:
                winsound.Beep(1000, 300)
                self._sleep_while_alarm(0.7)
            elif elapsed < 6.0:
                winsound.Beep(1500, 300)
                self._sleep_while_alarm(0.3)
            else:
                if not critical_loop_active:
                    winsound.PlaySound(
                        "SystemHand",
                        winsound.SND_ALIAS | winsound.SND_LOOP | winsound.SND_ASYNC
                    )
                    critical_loop_active = True
                self._sleep_while_alarm(0.5)

        if critical_loop_active:
            winsound.PlaySound(None, winsound.SND_PURGE)

    def _sleep_while_alarm(self, duration: float):
        """Interruptible sleep — stops immediately when alarm is disengaged."""
        start = time.time()
        while self.alarm_active and (time.time() - start) < duration:
            time.sleep(0.05)
