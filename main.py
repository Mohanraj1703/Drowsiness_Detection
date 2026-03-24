"""
Driver Safety System - Entry Point
===================================
Run this file to start the driver monitoring pipeline.

The system launches two components in parallel:
  1. Flask web server (port 5000) — serves the live mobile dashboard
  2. Detection pipeline      — reads webcam and writes metrics to shared state

Usage:
    python main.py

Access the dashboard:
    PC:     http://localhost:5000
    Mobile: http://<your-local-ip>:5000   (same Wi-Fi)

Controls:
    Press Ctrl+C to stop the system.
"""
import logging
import socket

from config import MonitorConfig
from driver_safety_system import DriverSafetySystem
from web_server import SharedState, start_server

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Returns the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    logger.info("=== Driver Safety System Starting ===")
    config = MonitorConfig()
    shared_state = SharedState()

    # Start Flask web dashboard in a background thread
    start_server(shared_state, host="0.0.0.0", port=5000)
    local_ip = get_local_ip()
    print("\n" + "=" * 50)
    print("  📱 Dashboard ready!")
    print(f"  PC:     http://localhost:5000")
    print(f"  Mobile: http://{local_ip}:5000")
    print("=" * 50 + "\n")

    try:
        system = DriverSafetySystem(config, shared_state=shared_state)
        system.run_pipeline()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully.")
    except Exception as e:
        logger.critical(f"Unhandled system failure: {e}", exc_info=True)
    finally:
        logger.info("=== Driver Safety System Stopped ===")


if __name__ == "__main__":
    main()
