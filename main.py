import os
import shutil
import time
import threading

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="OpenCV Video Streamer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Keep track of the currently uploaded video, whether detection is on,
# and a lock so only one capture reads the file at a time.
state = {
    "video_path": None,
    "detect": True,  # person detection on by default
}
lock = threading.Lock()

# Load the lightweight YOLOv8 nano model once at startup.
# COCO class 0 is "person" - we only ever keep that class.
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4
yolo_model = YOLO("yolov8n.pt")

# Streaming codec: WebP produces meaningfully smaller frames than JPEG at
# equivalent visual quality, so at the same bandwidth budget we can afford
# a higher quality setting (or higher resolution) than JPEG allowed.
STREAM_EXT = ".webp"
STREAM_CONTENT_TYPE = "image/webp"
STREAM_QUALITY = 80  # WebP quality (0-100); ~equivalent visual quality to JPEG 85 at a fraction of the size


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save the uploaded video to disk and set it as the active video."""
    allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{ext}'. Allowed: {allowed_ext}"},
        )

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Quick sanity check that OpenCV can actually open it
    cap = cv2.VideoCapture(dest_path)
    opened = cap.isOpened()
    cap.release()
    if not opened:
        os.remove(dest_path)
        return JSONResponse(
            status_code=400,
            content={"error": "OpenCV could not open this video file."},
        )

    with lock:
        state["video_path"] = dest_path

    return {"filename": file.filename, "status": "uploaded"}


def detect_persons(frame):
    """Run YOLO on a frame, keep only 'person' detections, and draw boxes."""
    results = yolo_model.predict(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    count = 0
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Persons: {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def generate_frames(video_path: str, detect: bool = True):
    """Generator that reads the video with OpenCV and yields WebP frames
    as a multipart/x-mixed-replace stream, paced to the source FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_delay = 1.0 / fps

    try:
        while True:
            start = time.time()
            success, frame = cap.read()
            if not success:
                break  # end of video

            if detect:
                frame = detect_persons(frame)

            ok, buffer = cv2.imencode(STREAM_EXT, frame, [int(cv2.IMWRITE_WEBP_QUALITY), STREAM_QUALITY])
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: " + STREAM_CONTENT_TYPE.encode() + b"\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                + frame_bytes + b"\r\n"
            )

            # pace playback to roughly match the source frame rate
            elapsed = time.time() - start
            remaining = frame_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    video_path = state["video_path"]
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "No video uploaded yet"})

    return StreamingResponse(
        generate_frames(video_path, detect=state["detect"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/toggle_detect")
def toggle_detect():
    with lock:
        state["detect"] = not state["detect"]
    return {"detect": state["detect"]}


@app.get("/status")
def status():
    return {"video_loaded": state["video_path"] is not None,
            "filename": os.path.basename(state["video_path"]) if state["video_path"] else None,
            "detect": state["detect"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# import os
# import shutil
# import time
# import threading

# import cv2
# import torch
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from ultralytics import YOLO

# app = FastAPI(title="OpenCV Video Streamer")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# STATIC_DIR = os.path.join(BASE_DIR, "static")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # Keep track of the currently uploaded video, whether detection is on,
# # and a lock so only one capture reads the file at a time.
# # Load the lightweight YOLOv8 nano model once at startup.
# # COCO class 0 is "person" - we only ever keep that class.
# PERSON_CLASS_ID = 0
# CONF_THRESHOLD = 0.4
# INFER_SIZE = 320   # inference resolution; 320 is a good speed/accuracy tradeoff on GPU too

# # Output stream settings - these control how much data has to travel over the
# # network per frame. If playback lags/slows down, the network can't keep up
# # with the current bitrate; lower STREAM_MAX_WIDTH and/or JPEG_QUALITY first.
# STREAM_MAX_WIDTH = 640   # frames wider than this get downscaled before sending
# JPEG_QUALITY = 65        # 0-100; lower = smaller frames = less bandwidth needed

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f"[startup] YOLO will run on: {DEVICE}"
#       + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.startswith("cuda") else " (no GPU detected)"))

# yolo_model = YOLO("yolov8n.pt")
# yolo_model.to(DEVICE)
# if DEVICE.startswith("cuda"):
#     # half-precision inference is faster on GPU with negligible accuracy loss for this model
#     yolo_model.model.half()

# state = {
#     "video_path": None,
#     "detect": True,     # person detection on by default
#     "fps": 25.0,
# }
# lock = threading.Lock()


# @app.get("/", response_class=HTMLResponse)
# def index():
#     index_path = os.path.join(STATIC_DIR, "index.html")
#     with open(index_path, "r", encoding="utf-8") as f:
#         return f.read()


# @app.post("/upload")
# async def upload_video(file: UploadFile = File(...)):
#     """Save the uploaded video to disk and set it as the active video."""
#     allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
#     ext = os.path.splitext(file.filename)[1].lower()
#     if ext not in allowed_ext:
#         return JSONResponse(
#             status_code=400,
#             content={"error": f"Unsupported file type '{ext}'. Allowed: {allowed_ext}"},
#         )

#     dest_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(dest_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Quick sanity check that OpenCV can actually open it, and grab its FPS
#     cap = cv2.VideoCapture(dest_path)
#     opened = cap.isOpened()
#     fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
#     cap.release()
#     if not opened:
#         os.remove(dest_path)
#         return JSONResponse(
#             status_code=400,
#             content={"error": "OpenCV could not open this video file."},
#         )

#     with lock:
#         state["video_path"] = dest_path
#         state["fps"] = fps if fps and fps > 0 else 25.0

#     return {"filename": file.filename, "status": "uploaded", "fps": state["fps"]}


# def detect_persons(frame):
#     """Run YOLO on a frame, keep only 'person' detections, and draw boxes."""
#     results = yolo_model.predict(
#         frame,
#         classes=[PERSON_CLASS_ID],
#         conf=CONF_THRESHOLD,
#         imgsz=INFER_SIZE,
#         device=DEVICE,
#         verbose=False,
#     )[0]

#     count = 0
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         count += 1
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"person {conf:.2f}"
#         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
#         cv2.putText(frame, label, (x1 + 2, y1 - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

#     cv2.putText(frame, f"Persons: {count}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
#     return frame


# def generate_frames(video_path: str, detect: bool = True):
#     """Generator that reads the video with OpenCV and yields JPEG frames
#     as a multipart/x-mixed-replace stream, paced to the source FPS."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if not fps or fps <= 0:
#         fps = 25.0
#     frame_delay = 1.0 / fps

#     try:
#         while True:
#             start = time.time()
#             success, frame = cap.read()
#             if not success:
#                 break  # end of video

#             if detect:
#                 frame = detect_persons(frame)

#             # downscale for network transport - detection already ran at
#             # full/INFER_SIZE resolution above, this only affects what gets sent
#             h, w = frame.shape[:2]
#             if w > STREAM_MAX_WIDTH:
#                 scale = STREAM_MAX_WIDTH / w
#                 frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)))

#             ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
#             if not ok:
#                 continue

#             frame_bytes = buffer.tobytes()
#             yield (
#                 b"--frame\r\n"
#                 b"Content-Type: image/jpeg\r\n"
#                 b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
#                 + frame_bytes + b"\r\n"
#             )

#             # pace playback to roughly match the source frame rate
#             elapsed = time.time() - start
#             remaining = frame_delay - elapsed
#             if remaining > 0:
#                 time.sleep(remaining)
#     finally:
#         cap.release()


# @app.get("/video_feed")
# def video_feed():
#     video_path = state["video_path"]
#     if not video_path or not os.path.exists(video_path):
#         return JSONResponse(status_code=404, content={"error": "No video uploaded yet"})

#     return StreamingResponse(
#         generate_frames(video_path, detect=state["detect"]),
#         media_type="multipart/x-mixed-replace; boundary=frame",
#         headers={
#             # discourage nginx-style reverse proxies (RunPod's HTTP proxy included)
#             # from buffering the stream before forwarding it to the browser
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#             "Pragma": "no-cache",
#             "X-Accel-Buffering": "no",
#             "Connection": "keep-alive",
#         },
#     )


# @app.post("/toggle_detect")
# def toggle_detect():
#     with lock:
#         state["detect"] = not state["detect"]
#     return {"detect": state["detect"]}


# @app.get("/status")
# def status():
#     return {"video_loaded": state["video_path"] is not None,
#             "filename": os.path.basename(state["video_path"]) if state["video_path"] else None,
#             "detect": state["detect"],
#             "fps": state["fps"]}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)