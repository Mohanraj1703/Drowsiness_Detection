import cv2
import csv
import logging
import os
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from scipy.spatial import distance
from threading import Thread, Lock
from typing import Tuple, Optional
import imutils
import numpy as np
import winsound
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core as mp_core
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MonitorConfig:
    """Configuration data class containing system parameters and thresholds."""
    # Detection Thresholds
    ear_threshold: float = 0.25
    mar_threshold: float = 0.6
    drowsy_frame_check: int = 20
    yawn_frame_check: int = 10
    distracted_frame_check: int = 20
    gaze_threshold_low: float = 0.42
    gaze_threshold_high: float = 0.58
    
    # Notification & Accounts (Environment variables should be used in production)
    sender_email: str = os.getenv("SENDER_EMAIL", "mohansanjai1716@gmail.com")
    sender_password: str = os.getenv("SENDER_PASSWORD", "ynsnafpxrckfoybj")
    receiver_email: str = os.getenv("RECEIVER_EMAIL", "mohansanjai1716@gmail.com")
    receiver_sms: str = os.getenv("RECEIVER_SMS", "9514210203@vtext.com")
    email_rate_limit_secs: int = 60
    
    # Paths & Files
    model_path: str = "D:/project/lung/Drowsiness_Detection/face_landmarker.task"
    log_csv: str = "alerts_log.csv"
    alerts_dir: str = "alerts"


class FaceAnalyzer:
    """Stateless utility class for all facial metric calculations."""
    
    @staticmethod
    def calculate_ear(eye_landmarks: np.ndarray) -> float:
        """Calculates Eye Aspect Ratio (EAR)."""
        a = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        b = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        c = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (a + b) / (2.0 * c)

    @staticmethod
    def calculate_mar(mouth_landmarks: np.ndarray) -> float:
        """Calculates Mouth Aspect Ratio (MAR)."""
        a = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])
        b = distance.euclidean(mouth_landmarks[2], mouth_landmarks[3])
        return a / b if b != 0 else 0

    @staticmethod
    def calculate_head_pose(landmarks, w: int, h: int) -> Tuple[float, float]:
        """Calculates Head Pitch and Yaw using OpenCV solvePnP."""
        face_2d = np.array([
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)] 
            for idx in [1, 33, 263, 61, 291, 152]
        ], dtype=np.float64)
        
        # Generic 3D model points in World Coordinates
        face_3d = np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [-165.0, -170.0, -135.0],   # Left eye outer corner
            [165.0, -170.0, -135.0],    # Right eye outer corner
            [-150.0, 150.0, -125.0],    # Left mouth corner
            [150.0, 150.0, -125.0],     # Right mouth corner
            [0.0, 330.0, -65.0]         # Chin
        ], dtype=np.float64)
        
        focal_length = 1 * w
        camera_mat = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype=np.float64)
        dist_coef = np.zeros((4, 1), dtype=np.float64)
        
        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, camera_mat, dist_coef)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        return angles[0], angles[1]  # pitch, yaw

    @staticmethod
    def calculate_gaze_ratio(landmarks, w: int, h: int) -> float:
        """Calculates average gaze ratio based on iris position."""
        if len(landmarks) < 478:
            return 0.5
            
        def _iris_ratio(left_idx, right_idx, iris_idx):
            p_left = np.array([landmarks[left_idx].x * w, landmarks[left_idx].y * h])
            p_right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])
            p_iris = np.array([landmarks[iris_idx].x * w, landmarks[iris_idx].y * h])
            width = distance.euclidean(p_left, p_right)
            dist = distance.euclidean(p_left, p_iris)
            return dist / width if width > 0 else 0.5

        left_ratio = _iris_ratio(33, 133, 468)
        right_ratio = _iris_ratio(362, 263, 473)
        return (left_ratio + right_ratio) / 2.0


class AlertManager:
    """Handles audio, email notifications, and local file logging sequentially and asynchronously."""
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alarm_active = False
        self.last_email_ts = 0.0
        self.last_screenshot_ts = 0.0
        self.state_lock = Lock()
        
        self._initialize_storage()

    def _initialize_storage(self):
        """Prepares necessary directories and CSV file."""
        for folder in ["DROWSINESS", "YAWNING", "DISTRACTED"]:
            os.makedirs(os.path.join(self.config.alerts_dir, folder), exist_ok=True)

        if not os.path.isfile(self.config.log_csv):
            try:
                with open(self.config.log_csv, mode='w', newline='') as f:
                    csv.writer(f).writerow(["Timestamp", "Alert_Type", "Image_File"])
            except Exception as e:
                logger.error(f"Failed to initialize CSV log: {e}")

    def log_incident(self, alert_type: str, frame: np.ndarray):
        """Saves physical proof of the alert and updates the CSV registry."""
        current_time = time.time()
        with self.state_lock:
            if current_time - self.last_screenshot_ts < 1.0:
                return
            self.last_screenshot_ts = current_time

        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(self.config.alerts_dir, alert_type, f"{alert_type}_{timestamp_str}.jpg")
        cv2.imwrite(img_path, frame)

        try:
            with open(self.config.log_csv, mode='a', newline='') as f:
                csv.writer(f).writerow([time.ctime(), alert_type, img_path])
        except PermissionError:
            logger.warning(f"Unable to update {self.config.log_csv} (File may be in use).")

    def dispatch_notification_async(self, alert_type: str):
        """Asynchronously dispatches SMS/Email notifications adhering to rate limits."""
        current_time = time.time()
        with self.state_lock:
            if current_time - self.last_email_ts < self.config.email_rate_limit_secs:
                return
            self.last_email_ts = current_time

        def _send_payload():
            try:
                logger.info(f"Transmitting remote {alert_type} notification...")
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(self.config.sender_email, self.config.sender_password)
                
                targets = [self.config.receiver_email, self.config.receiver_sms]
                for target in targets:
                    if "@" in target and "1234567890" not in target:
                        msg = EmailMessage()
                        msg.set_content(f"URGENT: System triggered a {alert_type} alert at {time.ctime()}!")
                        msg['Subject'] = f'Driver Alert: {alert_type} Detected'
                        msg['From'] = self.config.sender_email
                        msg['To'] = target
                        server.send_message(msg)
                
                server.quit()
                logger.info("Remote payload delivered successfully.")
            except Exception as e:
                logger.error(f"Notification dispatch failed: {e}")

        Thread(target=_send_payload, daemon=True).start()

    def _sleep_while_alarm(self, duration: float):
        start = time.time()
        while self.alarm_active and (time.time() - start) < duration:
            time.sleep(0.05)

    def engage_audio_alarm(self):
        """Engages an escalating asynchronous audio warning."""
        with self.state_lock:
            if self.alarm_active:
                return
            self.alarm_active = True

        def _sound_sequence():
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
                        winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_LOOP | winsound.SND_ASYNC)
                        critical_loop_active = True
                    self._sleep_while_alarm(0.5)

            if critical_loop_active:
                winsound.PlaySound(None, winsound.SND_PURGE)

        Thread(target=_sound_sequence, daemon=True).start()

    def disengage_audio_alarm(self):
        """Disables the audio warning system safely."""
        with self.state_lock:
            self.alarm_active = False


class DriverSafetySystem:
    """Core controller tying MediaPipe inference, Analysis, and Alerts together."""
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alert_manager = AlertManager(config)
        self.face_mesh = self._init_mediapipe_model()
        
        # Action Context Identifiers
        self.metrics = {"drowsy_frames": 0, "yawn_frames": 0, "distract_frames": 0}
        self.alarm_cooldown = 0

    def _init_mediapipe_model(self) -> vision.FaceLandmarker:
        try:
            base_opts = mp_core.base_options.BaseOptions(model_asset_path=self.config.model_path)
            opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_running_mode.VisionTaskRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return vision.FaceLandmarker.create_from_options(opts)
        except Exception as e:
            logger.critical(f"MediaPipe initialization failure: {e}")
            raise

    def process_frame_state(self, frame: np.ndarray) -> np.ndarray:
        """Processes driver landmarks against thresholds and updates UI overlay."""
        frame = imutils.resize(frame, width=450)
        h, w, _ = frame.shape
        mp_frame = mp_image.Image(mp_image.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        result = self.face_mesh.detect_for_video(mp_frame, int(time.time() * 1000))
        if not result.face_landmarks:
            return frame

        landmarks = result.face_landmarks[0]
        
        # Extraction & Mathematics
        left_eye = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in [33, 160, 158, 133, 153, 144]])
        right_eye = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in [362, 385, 387, 263, 373, 380]])
        mouth = np.array([(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in [13, 14, 78, 308]])

        ear = (FaceAnalyzer.calculate_ear(left_eye) + FaceAnalyzer.calculate_ear(right_eye)) / 2.0
        mar = FaceAnalyzer.calculate_mar(mouth)
        pitch, yaw = FaceAnalyzer.calculate_head_pose(landmarks, w, h)
        gaze = FaceAnalyzer.calculate_gaze_ratio(landmarks, w, h)

        # Plot Overlays
        cv2.polylines(frame, [left_eye, right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)
        
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"P: {pitch:.1f} Y: {yaw:.1f} G: {gaze:.2f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        self._evaluate_thresholds(frame, ear, mar, pitch, yaw, gaze)
        return frame

    def _evaluate_thresholds(self, frame, ear, mar, pitch, yaw, gaze):
        """Cross-checks live frame metrics against limits to issue warnings."""
        flags = {"is_drowsy": False, "is_yawning": False, "is_distracted": False}
        
        # Processing Triggers
        if yaw < -12 or yaw > 12 or pitch < -20 or gaze < self.config.gaze_threshold_low or gaze > self.config.gaze_threshold_high:
            self.metrics["distract_frames"] += 1
            if self.metrics["distract_frames"] >= self.config.distracted_frame_check:
                flags["is_distracted"] = True
        else:
            self.metrics["distract_frames"] = 0

        if ear < self.config.ear_threshold:
            self.metrics["drowsy_frames"] += 1
            if self.metrics["drowsy_frames"] >= self.config.drowsy_frame_check:
                flags["is_drowsy"] = True
        else:
            self.metrics["drowsy_frames"] = 0

        if mar > self.config.mar_threshold:
            self.metrics["yawn_frames"] += 1
            if self.metrics["yawn_frames"] >= self.config.yawn_frame_check:
                flags["is_yawning"] = True
        else:
            self.metrics["yawn_frames"] = 0

        # Rendering & Execution
        if flags["is_drowsy"]: cv2.putText(frame, "*** DROWSY ALERT! ***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if flags["is_yawning"]: cv2.putText(frame, "*** YAWN ALERT! ***", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if flags["is_distracted"]: cv2.putText(frame, "*** DISTRACTED ALERT! ***", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if any(flags.values()):
            self.alarm_cooldown = 20
            reason = "DROWSINESS" if flags["is_drowsy"] else "YAWNING" if flags["is_yawning"] else "DISTRACTED"
            
            self.alert_manager.log_incident(reason, frame)
            self.alert_manager.dispatch_notification_async(reason)
            self.alert_manager.engage_audio_alarm()
        else:
            self.alarm_cooldown -= 1
            if self.alarm_cooldown <= 0:
                self.alert_manager.disengage_audio_alarm()

    def run_pipeline(self):
        """Main lifecycle block for driving the pipeline loops."""
        logger.info("Engaging Primary Camera Feed...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Camera interface unready/unavailable.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                output = self.process_frame_state(frame)
                
                cv2.imshow("System Interface", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Standard shutdown received via local UI interaction.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()


def main():
    config = MonitorConfig()
    try:
        system = DriverSafetySystem(config)
        system.run_pipeline()
    except Exception as e:
        logger.critical(f"Critical execution error: {e}")

if __name__ == "__main__":
    main()
