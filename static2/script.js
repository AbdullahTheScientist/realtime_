const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusMsg = document.getElementById("statusMsg");
const streamBox = document.getElementById("streamBox");
const streamTitle = document.getElementById("streamTitle");
const canvas = document.getElementById("videoStream");
const ctx = canvas.getContext("2d");
const detectToggle = document.getElementById("detectToggle");

// ---------------------------------------------------------------------------
// Jitter buffer: the server sends a steady 15 fps, but TCP delivers those
// bytes in bursts. Painting each frame the instant it arrives (what a plain
// <img> does) turns that burstiness into visible stutter. Instead we decode
// frames into a small queue and paint them at a fixed cadence, holding a tiny
// cushion so a late-arriving frame doesn't starve the display. This trades a
// small, bounded latency (~a few frames) for smooth playback.
// ---------------------------------------------------------------------------
const DISPLAY_FPS = 15;
const DISPLAY_INTERVAL = 1000 / DISPLAY_FPS;
const TARGET_BUFFER = 3;   // frames to accumulate before (re)starting playback
const MAX_BUFFER = 8;      // hard cap; drop oldest beyond this to bound latency

let frameQueue = [];
let needCushion = true;    // re-arm the startup cushion after any dry-out
let lastPaint = 0;
let rafId = null;

let abortCtrl = null;
let recvBuf = new Uint8Array(0);
const textDecoder = new TextDecoder("latin1");

function concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function indexOf(buf, needle, from) {
  outer: for (let i = from; i <= buf.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (buf[i + j] !== needle[j]) continue outer;
    }
    return i;
  }
  return -1;
}

const HEADER_SEP = new Uint8Array([13, 10, 13, 10]); // \r\n\r\n

// Pull complete JPEG parts out of recvBuf. Each part is:
//   --frame\r\n Content-Type:...\r\n Content-Length: N\r\n\r\n <N bytes> \r\n
function drainParts() {
  while (true) {
    const sep = indexOf(recvBuf, HEADER_SEP, 0);
    if (sep === -1) return; // headers incomplete
    const header = textDecoder.decode(recvBuf.subarray(0, sep));
    const m = header.match(/Content-Length:\s*(\d+)/i);
    if (!m) {
      // malformed header block; drop up to the separator and resync
      recvBuf = recvBuf.subarray(sep + HEADER_SEP.length);
      continue;
    }
    const len = parseInt(m[1], 10);
    const start = sep + HEADER_SEP.length;
    if (recvBuf.length < start + len) return; // body not fully arrived yet
    const jpeg = recvBuf.subarray(start, start + len);
    enqueueFrame(jpeg.slice()); // copy out before we advance the buffer
    recvBuf = recvBuf.subarray(start + len);
  }
}

function enqueueFrame(bytes) {
  const blob = new Blob([bytes], { type: "image/jpeg" });
  // createImageBitmap decodes off the main thread; ignore late decodes after stop
  createImageBitmap(blob).then((bmp) => {
    if (abortCtrl === null) { bmp.close(); return; }
    frameQueue.push(bmp);
    while (frameQueue.length > MAX_BUFFER) {
      frameQueue.shift().close(); // drop oldest to bound latency
    }
  }).catch(() => {}); // skip undecodable frame
}

function renderLoop(ts) {
  rafId = requestAnimationFrame(renderLoop);

  // wait until a startup cushion has accumulated, so brief delivery gaps
  // don't immediately starve the display
  if (needCushion) {
    if (frameQueue.length < TARGET_BUFFER) return;
    needCushion = false;
    lastPaint = ts;
  }

  if (ts - lastPaint < DISPLAY_INTERVAL) return; // hold the fixed cadence

  if (frameQueue.length === 0) {
    needCushion = true; // buffer dried out; re-arm the cushion
    return;
  }

  const bmp = frameQueue.shift();
  if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
    canvas.width = bmp.width;
    canvas.height = bmp.height;
  }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  // advance by a fixed step (not ts) so cadence stays even under rAF jitter
  lastPaint += DISPLAY_INTERVAL;
  if (ts - lastPaint > DISPLAY_INTERVAL * 2) lastPaint = ts; // resync if we fell far behind
}

function stopStream() {
  if (abortCtrl) {
    abortCtrl.abort();
    abortCtrl = null;
  }
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  frameQueue.forEach((b) => b.close());
  frameQueue = [];
  recvBuf = new Uint8Array(0);
  needCushion = true;
}

async function startStream() {
  stopStream();
  abortCtrl = new AbortController();
  rafId = requestAnimationFrame(renderLoop);

  let res;
  try {
    res = await fetch("/video_feed?t=" + Date.now(), { signal: abortCtrl.signal });
  } catch (err) {
    if (err.name !== "AbortError") statusMsg.textContent = "Stream error: " + err.message;
    return;
  }
  if (!res.ok || !res.body) {
    statusMsg.textContent = "Stream failed to start (status " + res.status + ")";
    return;
  }

  const reader = res.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      recvBuf = concat(recvBuf, value);
      drainParts();
    }
  } catch (err) {
    if (err.name !== "AbortError") console.error("Stream read error:", err);
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = videoInput.files[0];
  if (!file) {
    statusMsg.textContent = "Please choose a video file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  statusMsg.textContent = "Uploading...";
  streamBox.style.display = "none";
  stopStream();

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      statusMsg.textContent = "Error: " + (data.error || "upload failed");
      return;
    }
    statusMsg.textContent = `Uploaded "${data.filename}". Starting stream...`;
    streamTitle.textContent = `Live Stream: ${data.filename}`;
    streamBox.style.display = "block";
    startStream();
  } catch (err) {
    statusMsg.textContent = "Upload failed: " + err.message;
  } finally {
    uploadBtn.disabled = false;
  }
});

detectToggle.addEventListener("change", async () => {
  try {
    await fetch("/toggle_detect", { method: "POST" });
  } catch (err) {
    console.error("Failed to toggle detection:", err);
  }
  // detection is applied live per-frame server-side; no stream restart needed
});
