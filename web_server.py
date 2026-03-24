"""
Flask Web Server — Driver Safety System Dashboard
===================================================
Provides a mobile-friendly web dashboard accessible on the local network.

Endpoints:
    GET /              → Serves the HTML dashboard
    GET /video_feed    → MJPEG live stream of the annotated camera feed
    GET /api/metrics   → JSON: live EAR, MAR, pitch, yaw, gaze, status, session time
    GET /api/alerts    → JSON: alert history log and per-type counts
"""

import json
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template


# ------------------------------------------------------------------ #
#  Shared State (thread-safe bridge between pipeline and Flask)
# ------------------------------------------------------------------ #

@dataclass
class SharedState:
    """
    Thread-safe data container written by DriverSafetySystem
    and read by the Flask web server.
    """
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Latest annotated frame as JPEG bytes
    frame_bytes: Optional[bytes] = None

    # Latest metric values
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gaze: float = 0.5

    # Alert flags
    is_drowsy: bool = False
    is_yawning: bool = False
    is_distracted: bool = False

    # Session info
    session_start: float = field(default_factory=time.time)
    alert_counts: dict = field(default_factory=lambda: {
        "DROWSINESS": 0,
        "YAWNING": 0,
        "DISTRACTED": 0,
    })
    alert_history: List[dict] = field(default_factory=list)

    def update_frame(self, frame):
        """Encode and store the latest annotated frame as JPEG."""
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            with self.lock:
                self.frame_bytes = buf.tobytes()

    def update_metrics(self, ear, mar, pitch, yaw, gaze,
                       is_drowsy, is_yawning, is_distracted):
        """Atomically update all telemetry values."""
        with self.lock:
            self.ear = ear
            self.mar = mar
            self.pitch = pitch
            self.yaw = yaw
            self.gaze = gaze
            self.is_drowsy = is_drowsy
            self.is_yawning = is_yawning
            self.is_distracted = is_distracted

    def record_alert(self, alert_type: str):
        """Append an alert event to history and increment counter."""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "type": alert_type,
        }
        with self.lock:
            self.alert_counts[alert_type] = self.alert_counts.get(alert_type, 0) + 1
            self.alert_history.insert(0, entry)
            if len(self.alert_history) > 50:          # cap history at 50 entries
                self.alert_history.pop()

    def get_metrics_snapshot(self) -> dict:
        with self.lock:
            elapsed = int(time.time() - self.session_start)
            status = "SAFE"
            if self.is_drowsy:
                status = "DROWSY"
            elif self.is_yawning:
                status = "YAWNING"
            elif self.is_distracted:
                status = "DISTRACTED"
            return {
                "ear": round(self.ear, 3),
                "mar": round(self.mar, 3),
                "pitch": round(self.pitch, 1),
                "yaw": round(self.yaw, 1),
                "gaze": round(self.gaze, 3),
                "status": status,
                "session_seconds": elapsed,
                "is_drowsy": self.is_drowsy,
                "is_yawning": self.is_yawning,
                "is_distracted": self.is_distracted,
            }

    def get_alerts_snapshot(self) -> dict:
        with self.lock:
            return {
                "counts": dict(self.alert_counts),
                "history": list(self.alert_history),
            }


# ------------------------------------------------------------------ #
#  Flask Application Factory
# ------------------------------------------------------------------ #

def create_app(shared_state: SharedState) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    def generate_frames():
        """MJPEG generator — yields JPEG frames from shared state."""
        while True:
            with shared_state.lock:
                frame = shared_state.frame_bytes
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.04)   # ~25 fps cap

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/metrics")
    def api_metrics():
        return jsonify(shared_state.get_metrics_snapshot())

    @app.route("/api/alerts")
    def api_alerts():
        return jsonify(shared_state.get_alerts_snapshot())

    return app


def start_server(shared_state: SharedState, host: str = "0.0.0.0", port: int = 5000):
    """Starts the Flask development server in a daemon thread."""
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)           # suppress Flask request spam

    app = create_app(shared_state)
    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread
