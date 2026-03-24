import logging
import time

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

logger = logging.getLogger(__name__)


class DriverSafetySystem:
    """
    Core controller that ties together:
      - MediaPipe face landmark inference
      - FaceAnalyzer metrics (EAR, MAR, Head Pose, Gaze)
      - AlertManager responses (alarm, logging, notifications)

    Runs a real-time pipeline loop reading frames from the webcam.
    """

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alert_manager = AlertManager(config)
        self.face_mesh = self._init_mediapipe_model()

        # Rolling frame counters for sustained-event detection
        self._drowsy_frames = 0
        self._yawn_frames = 0
        self._distract_frames = 0
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
        """Opens the webcam and runs the driver monitoring loop until 'q' is pressed."""
        logger.info("Engaging Primary Camera Feed...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Camera is unavailable. Check connection and try again.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame. Camera may have disconnected.")
                    break

                output_frame = self._process_frame(frame)
                cv2.imshow("Driver Safety System", output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Shutdown signal received. Closing system.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()

    # ------------------------------------------------------------------ #
    #  Frame Processing
    # ------------------------------------------------------------------ #

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single camera frame:
          1. Runs MediaPipe face landmark detection
          2. Extracts eye, mouth, head pose, and gaze metrics
          3. Draws HUD overlays on frame
          4. Evaluates alert conditions
        """
        frame = imutils.resize(frame, width=450)
        h, w, _ = frame.shape

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
        return frame

    def _evaluate_and_alert(self, frame, ear, mar, pitch, yaw, gaze):
        """
        Checks live metrics against thresholds using sustained-frame counters.
        Triggers alert overlays, audio alarm, logging, and notifications.
        """
        is_drowsy = False
        is_yawning = False
        is_distracted = False

        # --- Distraction: Head turned or eyes averted ---
        if (yaw < -12 or yaw > 12 or pitch < -20
                or gaze < self.config.gaze_threshold_low
                or gaze > self.config.gaze_threshold_high):
            self._distract_frames += 1
            if self._distract_frames >= self.config.distracted_frame_check:
                is_distracted = True
        else:
            self._distract_frames = 0

        # --- Drowsiness: Eyes closing ---
        if ear < self.config.ear_threshold:
            self._drowsy_frames += 1
            if self._drowsy_frames >= self.config.drowsy_frame_check:
                is_drowsy = True
        else:
            self._drowsy_frames = 0

        # --- Yawning: Mouth wide open ---
        if mar > self.config.mar_threshold:
            self._yawn_frames += 1
            if self._yawn_frames >= self.config.yawn_frame_check:
                is_yawning = True
        else:
            self._yawn_frames = 0

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

        # --- Trigger System Responses ---
        if is_drowsy or is_yawning or is_distracted:
            self._alarm_cooldown = 20
            reason = "DROWSINESS" if is_drowsy else "YAWNING" if is_yawning else "DISTRACTED"

            self.alert_manager.log_incident(reason, frame)
            self.alert_manager.dispatch_notification_async(reason)
            self.alert_manager.speak_alert(reason)
            self.alert_manager.engage_audio_alarm()
        else:
            self._alarm_cooldown -= 1
            if self._alarm_cooldown <= 0:
                self.alert_manager.disengage_audio_alarm()
