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
from flask import Flask, Response, jsonify, render_template, redirect, url_for, request, flash, send_file, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import os

from database import db
from models import User


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

    # Status flags
    system_running: bool = False
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

    def update_status_frame(self, message: str):
        """Create and push a black background with status text for UI feedback."""
        import numpy as np
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(img, message, (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        self.update_frame(img)

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

    def record_alert(self, alert_type: str, image_file: Optional[str] = None):
        """Append an alert event to history with optional image link."""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "type": alert_type,
            "image": image_file
        }
        with self.lock:
            self.alert_counts[alert_type] = self.alert_counts.get(alert_type, 0) + 1
            self.alert_history.insert(0, entry)
            if len(self.alert_history) > 50:
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
                "system_running": self.system_running,
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
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'driver-safety-system-secret-12345')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        db.create_all()
        # Create a default user if none exists
        if not User.query.filter_by(email="admin@safety.com").first():
            admin = User(email="admin@safety.com")
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()

    def generate_frames():
        """MJPEG generator — yields JPEG frames. Yields standby image if stream empty."""
        import numpy as np
        standby_img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(standby_img, "SYSTEM STANDBY - READY", (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        _, standby_bytes = cv2.imencode(".jpg", standby_img)
        standby_frame = standby_bytes.tobytes()

        while True:
            try:
                with shared_state.lock:
                    frame = shared_state.frame_bytes
                
                # Use current frame or standby if none
                out = frame if frame else standby_frame
                
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + out + b"\r\n"
                )
                time.sleep(0.06)   # ~16 fps (lower CPU but smooth enough)
            except (GeneratorExit, ConnectionResetError):
                break

    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    @app.route("/login", methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False

            user = User.query.filter_by(email=email).first()
            if not user or not user.check_password(password):
                flash('Please check your login details and try again.')
                return redirect(url_for('login'))

            login_user(user, remember=remember)
            return redirect(url_for('dashboard'))

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('login'))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template("index.html")

    @app.route("/video_feed")
    @login_required
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/metrics")
    @login_required
    def api_metrics():
        return jsonify(shared_state.get_metrics_snapshot())

    @app.route("/api/system/toggle", methods=['POST'])
    @login_required
    def toggle_system():
        action = request.json.get('action')
        if action == 'start':
            shared_state.system_running = True
        elif action == 'stop':
            shared_state.system_running = False
        return jsonify({"status": "success", "system_running": shared_state.system_running})

    @app.route("/download/log")
    @login_required
    def download_log():
        """Allows direct download of the alerts CSV log."""
        try:
            return send_file(os.path.abspath("alerts_log.csv"), as_attachment=True)
        except Exception as e:
            return str(e), 404

    @app.route("/alerts/<path:filename>")
    @login_required
    def get_alert_image(filename):
        """Serves captured alert screenshots from the alerts directory."""
        return send_from_directory(os.path.abspath("alerts"), filename)

    @app.route("/api/alerts")
    @login_required
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
