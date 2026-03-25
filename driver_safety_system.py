import logging
import time
from typing import Optional, TYPE_CHECKING

import cv2
import imutils
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import core as mp_core
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

from config import MonitorConfig
from face_analyzer import FaceAnalyzer
from alert_manager import AlertManager

if TYPE_CHECKING:
    from web_server import SharedState

logger = logging.getLogger(__name__)


class DriverSafetySystem:
    """
    Core controller that ties together:
      - MediaPipe face landmark inference
      - FaceAnalyzer metrics (EAR, MAR, Head Pose, Gaze)
      - AlertManager responses (alarm, logging, notifications)

    Runs a real-time pipeline loop reading frames from the webcam.
    """

    def __init__(self, config: MonitorConfig, shared_state: Optional["SharedState"] = None):
        self.config = config
        self.shared_state = shared_state
        self.alert_manager = AlertManager(config)
        self.face_mesh = self._init_mediapipe_model()
        # Hardware handle
        self.cap = None

        # Time-based sustained event detection (Start timestamps)
        self._drowsy_start_ts = 0.0
        self._yawn_start_ts = 0.0
        self._distracted_start_ts = 0.0
        self._active_alert = None
        self._alarm_cooldown = 0

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def _init_mediapipe_model(self) -> vision.FaceLandmarker:
        """Loads and returns the MediaPipe FaceLandmarker model."""
        try:
            base_opts = mp_core.base_options.BaseOptions(
                model_asset_path=self.config.model_path
            )
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
            logger.critical(f"MediaPipe model initialization failed: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Main Pipeline
    # ------------------------------------------------------------------ #

    def run_pipeline(self):
        """
        Runs the monitoring pipeline.
        If shared_state is active, it waits for the 'Start' signal from the web dashboard.
        Otherwise, it starts the camera immediately (standalone mode).
        """
        logger.info("Pipeline initialized. Standby...")

        while True:
            # If using web dashboard, wait until user clicks 'Start'
            if self.shared_state:
                # Reset metrics when stopped
                if not self.shared_state.system_running:
                    if self.cap: # Release camera if it was running and system stopped
                        self.cap.release()
                        self.cap = None
                        cv2.destroyAllWindows()
                        logger.info("Camera feed closed. Entering standby...")
                    time.sleep(0.5)
                    continue
            
            # Start camera if not already engaged
            if self.cap is None:
                if self.shared_state:
                    self.shared_state.update_status_frame("COMMUNICATING WITH HARDWARE...")
                
                logger.info("Engaging Primary Camera Feed...")
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Fast open on Windows
                
                # Settle sensors (skip first 10 frames of black/overexposed feed)
                if self.cap.isOpened():
                    for _ in range(10): 
                        self.cap.read()
                        time.sleep(0.01)
                    if self.shared_state:
                        self.shared_state.update_status_frame("CALIBRATING SENSORS...")
                else:
                    logger.error("Failed to connect to primary camera!")
                    if self.shared_state:
                        self.shared_state.update_status_frame("HARDWARE ERROR")
                    self.shared_state.system_running = False
                    continue # Go back to the start of the while True loop

            try:
                while True:
                    # check if stop signal received via web
                    if self.shared_state and not self.shared_state.system_running:
                        logger.info("Stop signal received. Disengaging camera.")
                        break

                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to read frame.")
                        break

                    output_frame = self._process_frame(frame)

                    # Handlers and controls
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        if self.shared_state: self.shared_state.system_running = False
                        break
                    
                    if self.shared_state is None:
                        cv2.imshow("Driver Safety System", output_frame)
                    
                # Outer loop standby
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                    cv2.destroyAllWindows()
                    logger.info("Camera feed closed. Entering standby...")
                
        # Only reached on global shutdown
        self.face_mesh.close()

    # ------------------------------------------------------------------ #
    #  Frame Processing
    # ------------------------------------------------------------------ #

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single camera frame and ensures feedback is pushed to UI.
        """
        frame = imutils.resize(frame, width=450)
        h, w, _ = frame.shape

        # CRITICAL FIX: Push raw frame to dashboard immediately so the feed isn't black
        if self.shared_state:
            self.shared_state.update_frame(frame)

        # Run MediaPipe inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
        result = self.face_mesh.detect_for_video(mp_frame, int(time.time() * 1000))

        if not result.face_landmarks:
            return frame

        landmarks = result.face_landmarks[0]

        # Extract landmark coordinates for each region
        left_eye = np.array(
            [(int(landmarks[p].x * w), int(landmarks[p].y * h))
             for p in [33, 160, 158, 133, 153, 144]], dtype=np.int32
        )
        right_eye = np.array(
            [(int(landmarks[p].x * w), int(landmarks[p].y * h))
             for p in [362, 385, 387, 263, 373, 380]], dtype=np.int32
        )
        mouth = np.array(
            [(int(landmarks[p].x * w), int(landmarks[p].y * h))
             for p in [13, 14, 78, 308]], dtype=np.int32
        )

        # Calculate all facial metrics
        ear = (FaceAnalyzer.calculate_ear(left_eye) + FaceAnalyzer.calculate_ear(right_eye)) / 2.0
        mar = FaceAnalyzer.calculate_mar(mouth)
        pitch, yaw = FaceAnalyzer.calculate_head_pose(landmarks, w, h)
        gaze = FaceAnalyzer.calculate_gaze_ratio(landmarks, w, h)

        # Draw facial region overlays
        cv2.polylines(frame, [left_eye, right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)

        # Draw telemetry HUD
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"P:{pitch:.1f} Y:{yaw:.1f} G:{gaze:.2f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # Evaluate all thresholds and trigger alerts
        self._evaluate_and_alert(frame, ear, mar, pitch, yaw, gaze)

        # Sync raw telemetry sensor values to shared state
        if self.shared_state:
            with self.shared_state.lock:
                self.shared_state.ear = ear
                self.shared_state.mar = mar
                self.shared_state.pitch = pitch
                self.shared_state.yaw = yaw
                self.shared_state.gaze = gaze

        return frame

    def _evaluate_and_alert(self, frame, ear, mar, pitch, yaw, gaze):
        """
        Threshold processing using precise time metrics (5-second requirement).
        """
        now = time.time()

        # --- Distraction Analysis ---
        dist_violated = (yaw < -15 or yaw > 15 or pitch < -25 
                         or gaze < self.config.gaze_threshold_low 
                         or gaze > self.config.gaze_threshold_high)
        is_distracted = False
        if dist_violated:
            if self._distracted_start_ts == 0: self._distracted_start_ts = now
            elif now - self._distracted_start_ts >= self.config.distracted_time_secs:
                is_distracted = True
        else:
            self._distracted_start_ts = 0

        # --- Drowsiness Analysis ---
        ear_violated = (ear < self.config.ear_threshold)
        is_drowsy = False
        if ear_violated:
            if self._drowsy_start_ts == 0: self._drowsy_start_ts = now
            elif now - self._drowsy_start_ts >= self.config.drowsy_time_secs:
                is_drowsy = True
        else:
            self._drowsy_start_ts = 0

        # --- Yawning Analysis ---
        yawn_violated = (mar > self.config.mar_threshold)
        is_yawning = False
        if yawn_violated:
            if self._yawn_start_ts == 0: self._yawn_start_ts = now
            elif now - self._yawn_start_ts >= self.config.yawn_time_secs:
                is_yawning = True
        else:
            self._yawn_start_ts = 0

        # --- Render Alert Banners ---
        if is_drowsy:
            cv2.putText(frame, "*** DROWSY ALERT! ***", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if is_yawning:
            cv2.putText(frame, "*** YAWN ALERT! ***", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if is_distracted:
            cv2.putText(frame, "*** DISTRACTED ALERT! ***", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # --- Trigger System Responses (Edge Triggered) ---
        if is_drowsy or is_yawning or is_distracted:
            self._alarm_cooldown = 20
            reason = "DROWSINESS" if is_drowsy else "YAWNING" if is_yawning else "DISTRACTED"

            # Only trigger once per event
            if self._active_alert != reason:
                self._active_alert = reason
                image_path = self.alert_manager.log_incident(reason, frame)
                self.alert_manager.dispatch_notification_async(reason)
                self.alert_manager.speak_alert(reason)
                self.alert_manager.engage_audio_alarm()

                if self.shared_state:
                    self.shared_state.record_alert(reason, image_path)
        else:
            self._active_alert = None
            self._alarm_cooldown -= 1
            if self._alarm_cooldown <= 0:
                self.alert_manager.disengage_audio_alarm()

        # Always sync alert flags to shared state
        if self.shared_state:
            self.shared_state.update_metrics(
                self.shared_state.ear, self.shared_state.mar,
                self.shared_state.pitch, self.shared_state.yaw,
                self.shared_state.gaze,
                is_drowsy, is_yawning, is_distracted
            )
