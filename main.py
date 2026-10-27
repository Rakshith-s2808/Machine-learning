from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from collections import deque
import cv2
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

PARKING_BAYS = [
    [50, 100, 110, 60],
    [180, 100, 110, 60],
    [310, 100, 110, 60],
    [440, 100, 110, 60]
]

VOTE_WINDOW = 9
slot_votes = {i: deque(maxlen=VOTE_WINDOW) for i in range(len(PARKING_BAYS))}

live_data = {
    "total": len(PARKING_BAYS),
    "occupied": 0,
    "free": len(PARKING_BAYS),
    "rate": 0.0,
    "forecast_30min": 0.0,
    "alert": False
}

camera = cv2.VideoCapture(0)

def detect_and_generate_frames():
    global live_data
    while True:
        success, frame = camera.read()
        if not success:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Feed Unavailable", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        occupied_count = 0
        for i, (x, y, w, h) in enumerate(PARKING_BAYS):
            roi = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lap_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            raw_decision = 1 if lap_variance > 100 else 0
            slot_votes[i].append(raw_decision)
            is_occupied = 1 if sum(slot_votes[i]) > (len(slot_votes[i]) // 2) else 0

            if is_occupied:
                occupied_count += 1
                color = (0, 0, 255)
                label = f"Bay {i+1}: Busy"
            else:
                color = (0, 255, 0)
                label = f"Bay {i+1}: Free"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        total = len(PARKING_BAYS)
        free = total - occupied_count
        rate = occupied_count / total if total > 0 else 0.0
        pred_rate = min(1.0, max(0.0, (0.75 * rate) + (0.25 * 0.65)))

        live_data["total"] = total
        live_data["occupied"] = occupied_count
        live_data["free"] = free
        live_data["rate"] = round(rate * 100, 1)
        live_data["forecast_30min"] = round(pred_rate * 100, 1)
        live_data["alert"] = pred_rate >= 0.85

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        detect_and_generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/parking-status")
def get_status():
    return live_data
