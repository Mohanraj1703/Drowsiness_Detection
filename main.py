"""
Driver Safety System - Entry Point
===================================
Run this file to start the driver monitoring pipeline.

Usage:
    python main.py

Controls:
    Press 'q' in the camera window to stop the system.
"""
import logging

from config import MonitorConfig
from driver_safety_system import DriverSafetySystem

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Driver Safety System Starting ===")
    config = MonitorConfig()

    try:
        system = DriverSafetySystem(config)
        system.run_pipeline()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully.")
    except Exception as e:
        logger.critical(f"Unhandled system failure: {e}", exc_info=True)
    finally:
        logger.info("=== Driver Safety System Stopped ===")


if __name__ == "__main__":
    main()
