import csv
import logging
import os
import smtplib
import time
import winsound
from datetime import datetime
from email.message import EmailMessage
from threading import Thread, Lock

import cv2
import numpy as np

from config import MonitorConfig

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages all system alerts and side-effects:
      - Escalating audio alarm (3-level siren)
      - Screenshot capture & organized file storage
      - CSV incident logging
      - Async email & SMS notifications (rate-limited)
    """

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alarm_active = False
        self.last_email_ts = 0.0
        self.last_screenshot_ts = 0.0
        self._lock = Lock()
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
        Rate-limited to once per second to avoid disk flooding.
        """
        current_time = time.time()
        with self._lock:
            if current_time - self.last_screenshot_ts < 1.0:
                return
            self.last_screenshot_ts = current_time

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(
            self.config.alerts_dir, alert_type, f"{alert_type}_{timestamp_str}.jpg"
        )
        cv2.imwrite(img_path, frame)

        try:
            with open(self.config.log_csv, mode='a', newline='') as f:
                csv.writer(f).writerow([time.ctime(), alert_type, img_path])
        except PermissionError:
            logger.warning(f"Unable to write to {self.config.log_csv} — file may be open.")

    # ------------------------------------------------------------------ #
    #  Remote Notifications
    # ------------------------------------------------------------------ #

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
