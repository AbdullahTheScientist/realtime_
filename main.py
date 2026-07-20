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

app = FastAPI(title="OpenCV Video Streamer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static2")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static2")

# Keep track of the currently uploaded video, whether detection is on,
# and a lock so only one capture reads the file at a time.
# Load the lightweight YOLOv8 nano model once at startup.
# COCO class 0 is "person" - we only ever keep that class.
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4
INFER_SIZE = 320   # inference resolution; 320 is a good speed/accuracy tradeoff on GPU too

# ---------------------------------------------------------------------------
# Adaptive stream settings
# ---------------------------------------------------------------------------
# Each rung is (max_width, webp_quality, fps_divisor). The controller walks
# down this ladder when the network can't keep up and back up when there is
# sustained headroom. Resolution/quality degrade first; fps_divisor > 1
# (sending every Nth source frame) only appears in the last rungs, so frame
# rate is sacrificed strictly as a last resort.
STREAM_LADDER = [
    (640, 75, 1),
    (560, 68, 1),
    (480, 62, 1),
    (416, 55, 1),
    (360, 50, 1),
    (320, 45, 1),
    (288, 40, 1),
    (256, 35, 1),
    (256, 30, 2),   # last resort: half frame rate
    (224, 28, 3),   # emergency: third frame rate
]
START_LEVEL = 4  # start mid-ladder (360px, q50) and probe upward from there

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
# (adaptive resize + WebP encode + send)
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

                ok, frame = cap.read()
                if not ok:
                    break  # end of video
                frame_idx += 1

                # YOLO always sees the full-resolution decoded frame; network
                # adaptation happens later, on a copy in the consumer, and
                # never affects inference input
                if state["detect"]:
                    frame = detect_persons(frame)

                self.slot.put(frame)
        finally:
            cap.release()
            self.slot.close()


class AdaptiveStreamController:
    """Picks the STREAM_LADDER rung based on how the network is keeping up.

    Signal: per-frame utilization = (encode + send time) / frame budget.
    With a multipart stream over TCP, the send blocks once the socket buffer
    fills, so under congestion the send time directly reflects the link's
    real throughput - no separate probing needed.

    Hysteresis (prevents rapid oscillation):
      * downgrade fast - a few consecutive over-budget frames
      * upgrade slow  - sustained low utilization for UP_HOLD_SEC, and only
        if the measured bandwidth predicts the higher rung will fit with
        BW_SAFETY margin
      * cooldown after every switch so decisions settle before the next one
      * a rung that fails soon after we upgrade into it gets an exponentially
        increasing hold-off before we try it again
    """

    DOWN_UTIL = 0.92        # frame took >92% of its budget -> counts as overload
    DOWN_STREAK = 3         # this many consecutive overloaded frames -> downgrade
    UP_UTIL = 0.55          # frame under 55% of budget -> counts as headroom
    UP_HOLD_SEC = 4.0       # need this much continuous headroom to upgrade
    COOLDOWN_SEC = 2.0      # no switches at all right after a switch
    BW_SAFETY = 0.7         # predicted send time of higher rung must fit in 70% of budget
    FAIL_FAST_SEC = 10.0    # downgrade within this after an upgrade = "that rung failed"
    HOLDOFF_BASE_SEC = 8.0  # first hold-off for a failed rung (doubles, capped)
    HOLDOFF_MAX_SEC = 120.0
    EWMA_ALPHA = 0.25

    def __init__(self, ladder=STREAM_LADDER, start_level=START_LEVEL):
        self.ladder = ladder
        self.level = max(0, min(start_level, len(ladder) - 1))
        self._down_streak = 0
        self._calm_since = None
        self._last_switch = time.perf_counter()
        self._last_upgrade_level = None
        self._ewma_bps = None          # measured link throughput, bytes/sec
        self._ewma_frame_bytes = None  # typical encoded frame size at current rung
        self._holdoff = {}             # level -> (retry_after_ts, last_penalty_sec)

    def current(self):
        return self.ladder[self.level]

    def frame_budget(self, source_fps):
        return self.ladder[self.level][2] / source_fps

    def on_frame(self, nbytes, elapsed, budget):
        now = time.perf_counter()

        a = self.EWMA_ALPHA
        self._ewma_frame_bytes = nbytes if self._ewma_frame_bytes is None \
            else (1 - a) * self._ewma_frame_bytes + a * nbytes
        # only trust throughput samples where the socket actually pushed back;
        # sub-5ms "sends" just measured a memcpy into an empty socket buffer
        if elapsed > 0.005:
            bps = nbytes / elapsed
            self._ewma_bps = bps if self._ewma_bps is None \
                else (1 - a) * self._ewma_bps + a * bps

        util = elapsed / budget if budget > 0 else 0.0
        in_cooldown = (now - self._last_switch) < self.COOLDOWN_SEC

        if util > self.DOWN_UTIL:
            self._calm_since = None
            self._down_streak += 1
            if (self._down_streak >= self.DOWN_STREAK
                    and self.level < len(self.ladder) - 1
                    and not in_cooldown):
                self._register_failure_if_fresh_upgrade(now)
                self._switch(self.level + 1, now, "down")
        elif util < self.UP_UTIL:
            self._down_streak = 0
            if self._calm_since is None:
                self._calm_since = now
            target = self.level - 1
            if (target >= 0
                    and not in_cooldown
                    and (now - self._calm_since) >= self.UP_HOLD_SEC
                    and now >= self._holdoff.get(target, (0.0, 0.0))[0]
                    and self._bandwidth_allows(target, budget)):
                self._last_upgrade_level = target
                self._switch(target, now, "up")
        else:
            # mid-band: neither overloaded nor comfortable - reset both
            # counters so we stay put (this dead zone is the hysteresis core)
            self._down_streak = 0
            self._calm_since = None

    def _bandwidth_allows(self, target, current_budget):
        """Predict whether the higher rung's frames would still fit in their
        budget on the measured link, with safety margin."""
        if self._ewma_bps is None or self._ewma_frame_bytes is None:
            return True  # no congestion observed yet -> probe upward
        cur_w, cur_q, _ = self.ladder[self.level]
        tgt_w, tgt_q, tgt_div = self.ladder[target]
        # frame size scales ~quadratically with width and ~linearly with the
        # WebP quality setting in this range - a rough but serviceable model
        predicted = self._ewma_frame_bytes * (tgt_w / cur_w) ** 2 * (tgt_q / cur_q)
        target_budget = current_budget * tgt_div / self.ladder[self.level][2]
        return predicted / self._ewma_bps <= target_budget * self.BW_SAFETY

    def _register_failure_if_fresh_upgrade(self, now):
        """If we're downgrading shortly after an upgrade, the rung we tried
        clearly doesn't fit - back off exponentially before retrying it."""
        if (self._last_upgrade_level == self.level
                and (now - self._last_switch) < self.FAIL_FAST_SEC):
            _, last_pen = self._holdoff.get(self.level, (0.0, 0.0))
            pen = min(max(self.HOLDOFF_BASE_SEC, last_pen * 2), self.HOLDOFF_MAX_SEC)
            self._holdoff[self.level] = (now + pen, pen)

    def _switch(self, new_level, now, direction):
        w, q, div = self.ladder[new_level]
        print(f"[adaptive] {direction}: level {self.level} -> {new_level} "
              f"(width={w}, quality={q}, fps/{div})"
              + (f", est. bw {self._ewma_bps * 8 / 1e6:.2f} Mbps" if self._ewma_bps else ""))
        self.level = new_level
        self._last_switch = now
        self._down_streak = 0
        self._calm_since = None


def generate_frames(slot: LatestFrameSlot, source_fps: float):
    """Per-client consumer: takes the newest annotated frame, downscales and
    WebP-encodes it per the adaptive controller, and yields it as a
    multipart part. All resizing here is transport-only - inference already
    ran at full resolution in the producer."""
    controller = AdaptiveStreamController()
    last_seq = 0
    next_send = time.perf_counter()

    while True:
        frame, last_seq = slot.get_newer(last_seq)
        if frame is None:
            if slot.closed:
                break  # end of video / producer gone
            continue  # spurious timeout; keep waiting

        width, quality, divisor = controller.current()
        budget = controller.frame_budget(source_fps)

        # fps divisor (last-resort rungs only): skip frames until the next
        # send window; because the slot always hands us the newest frame,
        # skipping never adds latency
        now = time.perf_counter()
        if divisor > 1 and now < next_send:
            continue

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        if w > width:
            scale = width / w
            frame = cv2.resize(frame, (width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".webp", frame,
                                  [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)])
        if not ok:
            continue
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/webp\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
            + frame_bytes + b"\r\n"
        )

        # time from encode start until the chunk was actually accepted by the
        # transport; under congestion the socket blocks, so this measures the
        # real network drain rate - the controller's whole input signal
        elapsed = time.perf_counter() - t0
        controller.on_frame(len(frame_bytes), elapsed, budget)
        next_send = max(next_send + budget, t0)


@app.get("/video_feed")
def video_feed():
    video_path = state["video_path"]
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "No video uploaded yet"})

    source_fps = state["fps"]
    slot = LatestFrameSlot()
    producer = FrameProducer(video_path, slot)
    producer.start()

    def stream():
        try:
            yield from generate_frames(slot, source_fps)
        finally:
            # client disconnected or stream ended: tear the producer down so
            # no thread/capture outlives the request
            producer.stop()
            slot.close()

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            # discourage nginx-style reverse proxies (RunPod's HTTP proxy included)
            # from buffering the stream before forwarding it to the browser
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)