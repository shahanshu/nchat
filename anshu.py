import subprocess
import webbrowser
import time
import os

# Start uvicorn server
uvicorn_process = subprocess.Popen(
    ["uvicorn", "main:app", "--reload"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait a bit for the server to start
time.sleep(10)

# Get absolute path of index.html
file_path = os.path.abspath("index.html")

# Open index.html in default browser
webbrowser.open(f"file://{file_path}")

# Keep script running so uvicorn doesn't stop
try:
    uvicorn_process.wait()
except KeyboardInterrupt:
    uvicorn_process.terminate()