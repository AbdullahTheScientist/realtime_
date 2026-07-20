# import asyncio
# import os
# import shutil
# import time
# import threading
# import fractions

# import cv2
# import numpy as np
# import torch
# from av import VideoFrame
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from ultralytics import YOLO

# from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
# from aiortc import VideoStreamTrack

# # aiortc's H264 encoder defaults to a 3 Mbps ceiling, which can exceed a
# # constrained real-world link (confirmed ~2.7 Mbps here) once its congestion
# # control ramps up - and since RTP doesn't retransmit lost real-time video
# # packets, any resulting loss shows up directly as skipped/glitched frames.
# # Cap it to a safer ceiling with margin below the known link capacity.
# import aiortc.codecs.h264 as _h264_codec
# _h264_codec.DEFAULT_BITRATE = 700_000    # 700 kbps starting point
# _h264_codec.MIN_BITRATE = 300_000        # 300 kbps floor
# _h264_codec.MAX_BITRATE = 1_800_000      # 1.8 Mbps ceiling - safe margin under 2.7 Mbps

# STREAM_MAX_WIDTH = 640  # downscale before encoding; smaller frames = less work
#                          # for the encoder and less data needed for the same
#                          # perceived quality at a given bitrate ceiling

# app = FastAPI(title="OpenCV + YOLO WebRTC Streamer")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# STATIC_DIR = os.path.join(BASE_DIR, "static")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # ---------------------------------------------------------------------------
# # YOLO setup (same model/approach as the MJPEG version)
# # ---------------------------------------------------------------------------
# PERSON_CLASS_ID = 0
# CONF_THRESHOLD = 0.4
# INFER_SIZE = 320

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f"[startup] YOLO will run on: {DEVICE}"
#       + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.startswith("cuda") else " (no GPU detected)"))

# yolo_model = YOLO("yolov8n.pt")
# yolo_model.to(DEVICE)
# if DEVICE.startswith("cuda"):
#     yolo_model.model.half()


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


# # ---------------------------------------------------------------------------
# # App state
# # ---------------------------------------------------------------------------
# state = {
#     "video_path": None,
#     "detect": True,
#     "fps": 25.0,
# }
# lock = threading.Lock()

# # keep references to active peer connections so we can clean them up
# peer_connections: set[RTCPeerConnection] = set()


# @app.get("/", response_class=HTMLResponse)
# def index():
#     with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
#         return f.read()


# @app.post("/upload")
# async def upload_video(file: UploadFile = File(...)):
#     allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
#     ext = os.path.splitext(file.filename)[1].lower()
#     if ext not in allowed_ext:
#         return JSONResponse(status_code=400, content={"error": f"Unsupported file type '{ext}'"})

#     dest_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(dest_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     cap = cv2.VideoCapture(dest_path)
#     opened = cap.isOpened()
#     fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
#     cap.release()
#     if not opened:
#         os.remove(dest_path)
#         return JSONResponse(status_code=400, content={"error": "OpenCV could not open this video file."})

#     with lock:
#         state["video_path"] = dest_path
#         state["fps"] = fps if fps and fps > 0 else 25.0

#     return {"filename": file.filename, "status": "uploaded", "fps": state["fps"]}


# @app.post("/toggle_detect")
# def toggle_detect():
#     with lock:
#         state["detect"] = not state["detect"]
#     return {"detect": state["detect"]}


# @app.get("/status")
# def status():
#     return {
#         "video_loaded": state["video_path"] is not None,
#         "filename": os.path.basename(state["video_path"]) if state["video_path"] else None,
#         "detect": state["detect"],
#         "fps": state["fps"],
#     }


# # ---------------------------------------------------------------------------
# # WebRTC video track: reads the uploaded video, runs YOLO, emits real frames
# # ---------------------------------------------------------------------------
# class PersonDetectionTrack(VideoStreamTrack):
#     """A WebRTC video track that reads an uploaded video file with OpenCV,
#     optionally runs YOLO person detection on each frame, and yields the
#     result as a real encoded video stream (not per-frame images)."""

#     def __init__(self, video_path: str):
#         super().__init__()
#         self.cap = cv2.VideoCapture(video_path)
#         fps = self.cap.get(cv2.CAP_PROP_FPS)
#         self.fps = fps if fps and fps > 0 else 25.0
#         self.frame_interval = 1.0 / self.fps
#         self._start_time = None
#         self._frame_count = 0

#     async def recv(self):
#         try:
#             return await self._recv_impl()
#         except Exception as e:
#             import traceback
#             print("[track] ERROR in recv():", e)
#             traceback.print_exc()
#             raise

#     async def _recv_impl(self):
#         # pace ourselves to the source video's fps using an absolute clock,
#         # same drift-avoidance approach as the MJPEG version used client-side -
#         # here it matters server-side since we control frame timing directly
#         if self._start_time is None:
#             self._start_time = time.time()

#         target_time = self._start_time + self._frame_count * self.frame_interval
#         now = time.time()
#         if target_time > now:
#             await asyncio.sleep(target_time - now)

#         success, frame = await asyncio.get_event_loop().run_in_executor(None, self.cap.read)
#         if not success:
#             # loop the video so the demo keeps running instead of just stopping
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             success, frame = await asyncio.get_event_loop().run_in_executor(None, self.cap.read)
#             if not success:
#                 frame = np.zeros((360, 640, 3), dtype=np.uint8)

#         if state["detect"]:
#             frame = await asyncio.get_event_loop().run_in_executor(None, detect_persons, frame)

#         # downscale before handing to the encoder - smaller frames need less
#         # data for the same visual quality at a given bitrate ceiling, and
#         # cost the (CPU-bound) H264/VP8 encoder less work per frame
#         h, w = frame.shape[:2]
#         if w > STREAM_MAX_WIDTH:
#             scale = STREAM_MAX_WIDTH / w
#             frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)))

#         self._frame_count += 1

#         video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
#         pts = self._frame_count
#         video_frame.pts = pts
#         video_frame.time_base = fractions.Fraction(1, int(round(self.fps)))
#         return video_frame


# class OfferPayload(BaseModel):
#     sdp: str
#     type: str


# @app.post("/offer")
# async def offer(payload: OfferPayload):
#     if not state["video_path"]:
#         return JSONResponse(status_code=400, content={"error": "No video uploaded yet"})

#     offer_desc = RTCSessionDescription(sdp=payload.sdp, type=payload.type)

#     # STUN helps discover your own public address for ICE candidate gathering.
#     # If ICE fails to connect (common behind strict NATs / cloud firewalls),
#     # a TURN server is needed - see README notes.
#     ice_servers = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
#     turn_url = os.environ.get("TURN_URL")
#     if turn_url:
#         ice_servers.append(RTCIceServer(
#             urls=turn_url,
#             username=os.environ.get("TURN_USERNAME", ""),
#             credential=os.environ.get("TURN_CREDENTIAL", ""),
#         ))

#     pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
#     peer_connections.add(pc)

#     @pc.on("connectionstatechange")
#     async def on_connectionstatechange():
#         print(f"[webrtc] connection state: {pc.connectionState}")
#         if pc.connectionState in ("failed", "closed", "disconnected"):
#             await pc.close()
#             peer_connections.discard(pc)

#     track = PersonDetectionTrack(state["video_path"])
#     pc.addTrack(track)

#     await pc.setRemoteDescription(offer_desc)
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


# @app.on_event("shutdown")
# async def on_shutdown():
#     for pc in list(peer_connections):
#         await pc.close()
#     peer_connections.clear()


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)







import os
import shutil
import time
import threading

import cv2
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="OpenCV + YOLO MJPEG Streamer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static2")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static2")

# Transport tuning. The single-slot pipeline (below) already self-regulates
# frame RATE under congestion - a slow socket makes the consumer pull fewer
# frames and older ones are dropped, so no queue builds and latency stays flat.
# That means we don't need an adaptive quality ladder; a fixed, modest
# resolution + JPEG quality keeps per-frame size predictable, and JPEG encodes
# much faster and with far less frame-to-frame variance than WebP (the old
# variance was a real source of visible jitter).
STREAM_MAX_WIDTH = 560   # transport downscale; inference still runs full-res
JPEG_QUALITY = 65        # good quality/size tradeoff for a constrained link

# Keep track of the currently uploaded video, whether detection is on,
# and a lock so only one capture reads the file at a time.
# Load the lightweight YOLOv8 nano model once at startup.
# COCO class 0 is "person" - we only ever keep that class.
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4
INFER_SIZE = 320   # inference resolution; 320 is a good speed/accuracy tradeoff on GPU too

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[startup] YOLO will run on: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.startswith("cuda") else " (no GPU detected)"))

yolo_model = YOLO("yolov8n.pt")
yolo_model.to(DEVICE)
if DEVICE.startswith("cuda"):
    # half-precision inference is faster on GPU with negligible accuracy loss for this model
    yolo_model.model.half()

state = {
    "video_path": None,
    "detect": True,     # person detection on by default
    "fps": 25.0,
}
lock = threading.Lock()


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

    # Quick sanity check that OpenCV can actually open it, and grab its FPS
    cap = cv2.VideoCapture(dest_path)
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0
    cap.release()
    if not opened:
        os.remove(dest_path)
        return JSONResponse(
            status_code=400,
            content={"error": "OpenCV could not open this video file."},
        )

    with lock:
        state["video_path"] = dest_path
        state["fps"] = fps if fps and fps > 0 else 25.0

    return {"filename": file.filename, "status": "uploaded", "fps": state["fps"]}


def detect_persons(frame):
    """Run YOLO on a frame, keep only 'person' detections, and draw boxes."""
    results = yolo_model.predict(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=CONF_THRESHOLD,
        imgsz=INFER_SIZE,
        device=DEVICE,
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


# ---------------------------------------------------------------------------
# Frame pipeline: producer (decode + YOLO at source pace) -> single-slot
# buffer (newest frame wins, older frames dropped) -> per-client consumer
# (resize + JPEG encode + send)
# ---------------------------------------------------------------------------
class LatestFrameSlot:
    """Single-slot handoff between the producer and a stream consumer.

    put() overwrites whatever is in the slot, so a slow consumer never causes
    frames to queue up - it simply misses intermediate frames and always gets
    the newest one. This is what keeps latency flat and memory bounded (at
    most one frame is held here, regardless of how far the network falls
    behind)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def put(self, frame):
        with self._cond:
            self._frame = frame  # older undelivered frame is dropped here
            self._seq += 1
            self._cond.notify_all()

    def get_newer(self, last_seq, timeout=2.0):
        """Block until a frame newer than last_seq exists (or close/timeout).
        Returns (frame, seq); frame is None on timeout or close."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._closed or self._seq > last_seq, timeout=timeout
            )
            if self._seq > last_seq:
                return self._frame, self._seq
            return None, last_seq

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class FrameProducer(threading.Thread):
    """Reads the video with OpenCV at the source frame rate, runs YOLO on the
    full-resolution frame, and publishes results into a LatestFrameSlot.

    Runs on its own clock, completely decoupled from the network: if the
    consumer/network is slow that is invisible here, so playback position
    always tracks wall-clock time (real-time playback). If *processing* ever
    falls behind (YOLO hiccup, slow disk), it skips source frames to catch up
    rather than letting the whole stream slip."""

    def __init__(self, video_path: str, slot: LatestFrameSlot):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.slot = slot
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.slot.close()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        interval = 1.0 / fps

        start = time.perf_counter()
        frame_idx = 0
        # instrumentation: how well the producer holds source fps
        stat_t0 = start
        stat_frames = 0      # frames actually decoded + published this window
        stat_skipped = 0     # frames dropped (grab-only) to catch up this window
        stat_infer_s = 0.0   # cumulative YOLO time this window
        try:
            while not self._stop_event.is_set():
                # absolute-clock pacing (start + n*interval) instead of
                # per-frame sleep() so small timing errors don't accumulate
                # into playback drift
                target = start + frame_idx * interval
                now = time.perf_counter()
                if now < target:
                    time.sleep(target - now)
                elif now - target > interval:
                    # processing fell behind: drop (grab without decode) the
                    # frames we missed so playback stays synced to real time
                    behind = int((now - target) / interval)
                    for _ in range(behind):
                        if not cap.grab():
                            break
                        frame_idx += 1
                        stat_skipped += 1

                ok, frame = cap.read()
                if not ok:
                    # end of file: loop back to the start and re-anchor the
                    # pacing clock so playback stays smooth and real-time
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break  # genuinely unreadable, not just EOF
                    start = time.perf_counter()
                    frame_idx = 0
                frame_idx += 1

                # YOLO always sees the full-resolution decoded frame; network
                # adaptation happens later, on a copy in the consumer, and
                # never affects inference input
                if state["detect"]:
                    _inf0 = time.perf_counter()
                    frame = detect_persons(frame)
                    stat_infer_s += time.perf_counter() - _inf0

                self.slot.put(frame)
                stat_frames += 1

                elapsed = now - stat_t0
                if elapsed >= 2.0:
                    infer_ms = (stat_infer_s / stat_frames * 1000) if stat_frames else 0.0
                    print(f"[producer] {stat_frames / elapsed:.1f} fps produced, "
                          f"{stat_skipped} skipped, {infer_ms:.0f} ms/frame YOLO "
                          f"(source {fps:.0f} fps)")
                    stat_t0 = now
                    stat_frames = 0
                    stat_skipped = 0
                    stat_infer_s = 0.0
        finally:
            cap.release()
            self.slot.close()


def generate_frames(slot: LatestFrameSlot):
    """Per-client consumer: take the newest annotated frame, downscale for
    transport, JPEG-encode, and yield it as a multipart part.

    No adaptive ladder: the slot's newest-wins handoff means that when the
    TCP send blocks under congestion, we simply pull the next frame later and
    intervening frames were already dropped - so the stream self-throttles on
    frame rate with flat latency. All resizing here is transport-only;
    inference already ran at full resolution in the producer."""
    last_seq = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    # instrumentation: sent fps, encode cost, frame size, and socket-send stall
    stat_t0 = time.perf_counter()
    stat_sent = 0
    stat_enc_s = 0.0
    stat_send_s = 0.0
    stat_bytes = 0
    stat_send_max = 0.0
    _last_yield = stat_t0

    while True:
        frame, last_seq = slot.get_newer(last_seq)
        if frame is None:
            if slot.closed:
                break  # end of video / producer gone
            continue  # spurious timeout; keep waiting

        _enc0 = time.perf_counter()
        # the gap between finishing the previous yield and now is how long the
        # socket send blocked (backpressure) plus wait-for-frame; under
        # congestion this is where time goes
        send_block = _enc0 - _last_yield

        h, w = frame.shape[:2]
        if w > STREAM_MAX_WIDTH:
            scale = STREAM_MAX_WIDTH / w
            frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        enc_s = time.perf_counter() - _enc0

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
            + frame_bytes + b"\r\n"
        )
        _last_yield = time.perf_counter()

        stat_sent += 1
        stat_enc_s += enc_s
        stat_send_s += send_block
        stat_send_max = max(stat_send_max, send_block)
        stat_bytes += len(frame_bytes)
        window = _last_yield - stat_t0
        if window >= 2.0:
            mbps = stat_bytes * 8 / window / 1e6
            print(f"[consumer] {stat_sent / window:.1f} fps sent, "
                  f"enc {stat_enc_s / stat_sent * 1000:.0f} ms, "
                  f"send-stall avg {stat_send_s / stat_sent * 1000:.0f} ms "
                  f"max {stat_send_max * 1000:.0f} ms, "
                  f"{stat_bytes // stat_sent // 1024} KB/frame, {mbps:.1f} Mbps")
            stat_t0 = _last_yield
            stat_sent = 0
            stat_enc_s = 0.0
            stat_send_s = 0.0
            stat_send_max = 0.0
            stat_bytes = 0


@app.get("/video_feed")
def video_feed():
    video_path = state["video_path"]
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "No video uploaded yet"})

    slot = LatestFrameSlot()
    producer = FrameProducer(video_path, slot)
    producer.start()

    def stream():
        try:
            yield from generate_frames(slot)
        finally:
            # client disconnected or stream ended: tear the producer down so
            # no thread/capture outlives the request
            producer.stop()
            slot.close()

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            # discourage nginx-style reverse proxies from buffering the stream
            # before forwarding it to the browser
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
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
            "detect": state["detect"],
            "fps": state["fps"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)