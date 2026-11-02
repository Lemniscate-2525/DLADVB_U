import subprocess

fastapi = subprocess.Popen(
    [
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ]
)

streamlit = subprocess.Popen(
    [
        "streamlit",
        "run",
        "app/streamlit_app.py"
    ]
)

fastapi.wait()

streamlit.wait()