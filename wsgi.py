import os
from web_server import create_app, SharedState
from config import MonitorConfig

# Initialize shared state and config for the web worker
shared_state = SharedState()
config = MonitorConfig()

app = create_app(shared_state, config)

if __name__ == "__main__":
    app.run()
