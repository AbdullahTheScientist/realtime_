const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusMsg = document.getElementById("statusMsg");
const streamBox = document.getElementById("streamBox");
const streamTitle = document.getElementById("streamTitle");
const detectToggle = document.getElementById("detectToggle");
const canvas = document.getElementById("videoCanvas");
const bufferMsg = document.getElementById("bufferMsg");
const progressWrap = document.getElementById("progressWrap");
const progressBar = document.getElementById("progressBar");
const progressLabel = document.getElementById("progressLabel");
const ctx = canvas.getContext("2d");

// --- Jitter-buffered MJPEG player -----------------------------------------
// A plain <img src="/video_feed"> paints frames the instant they arrive,
// so any network jitter (common on real internet links) shows up directly
// as stutter/skipped frames. Instead, we read the raw multipart stream
// ourselves, queue a handful of decoded frames first, then play them out
// on a steady timer. This trades a small fixed delay (a few hundred ms)
// for much smoother playback.

const TARGET_BUFFER_FRAMES = 6; // frames to queue before playback starts
let frameQueue = [];
let playing = false;
let playTimer = null;
let currentAbortController = null;

function stopStream() {
  playing = false;
  if (playTimer) {
    clearTimeout(playTimer);
    playTimer = null;
  }
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
  frameQueue.forEach((bmp) => bmp.close && bmp.close());
  frameQueue = [];
}

function findSubarray(buf, pattern, from) {
  outer: for (let i = from; i <= buf.length - pattern.length; i++) {
    for (let j = 0; j < pattern.length; j++) {
      if (buf[i + j] !== pattern[j]) continue outer;
    }
    return i;
  }
  return -1;
}

const CRLFCRLF = new Uint8Array([13, 10, 13, 10]);

async function readMjpegStream(url, onFrame) {
  currentAbortController = new AbortController();
  const res = await fetch(url, { signal: currentAbortController.signal });
  if (!res.ok || !res.body) throw new Error("Stream request failed: " + res.status);

  const reader = res.body.getReader();
  let buf = new Uint8Array(0);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // append newly-received bytes onto our working buffer
    const merged = new Uint8Array(buf.length + value.length);
    merged.set(buf, 0);
    merged.set(value, buf.length);
    buf = merged;

    // try to extract as many complete frames as are currently available
    while (true) {
      const headerEnd = findSubarray(buf, CRLFCRLF, 0);
      if (headerEnd === -1) break; // headers not fully received yet

      const headerText = new TextDecoder().decode(buf.subarray(0, headerEnd));
      const lengthMatch = headerText.match(/Content-Length:\s*(\d+)/i);
      const typeMatch = headerText.match(/Content-Type:\s*([\w/]+)/i);
      if (!lengthMatch) {
        // malformed/unexpected chunk; drop up to end of headers and retry
        buf = buf.subarray(headerEnd + 4);
        continue;
      }
      const contentLength = parseInt(lengthMatch[1], 10);
      const mimeType = typeMatch ? typeMatch[1] : "image/jpeg";
      const frameStart = headerEnd + 4;
      const frameEnd = frameStart + contentLength;

      if (buf.length < frameEnd + 2) break; // full frame not received yet

      const imgBytes = buf.slice(frameStart, frameEnd);
      onFrame(imgBytes, mimeType);

      // advance past this frame + trailing CRLF, then past next boundary line
      let next = frameEnd + 2;
      const nextHeaderStart = findSubarray(buf, CRLFCRLF, next);
      buf = nextHeaderStart === -1 ? buf.subarray(next) : buf.subarray(next);
    }
  }
}

async function startStream(fps) {
  stopStream();
  playing = true;
  const interval = 1000 / (fps && fps > 0 ? fps : 25);
  bufferMsg.textContent = "Buffering...";

  const onFrame = async (imgBytes, mimeType) => {
    try {
      const blob = new Blob([imgBytes], { type: mimeType });
      const bitmap = await createImageBitmap(blob);
      frameQueue.push(bitmap);
    } catch (e) {
      // corrupt/partial frame, skip it
    }
  };

  readMjpegStream("/video_feed?t=" + Date.now(), onFrame).catch((err) => {
    if (err.name !== "AbortError") console.error("Stream error:", err);
  });

  const playLoop = () => {
    if (!playing) return;

    if (frameQueue.length > 0) {
      const bmp = frameQueue.shift();
      if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas.width = bmp.width;
        canvas.height = bmp.height;
      }
      ctx.drawImage(bmp, 0, 0);
      bmp.close && bmp.close();
      bufferMsg.textContent = `Buffer: ${frameQueue.length} frame(s)`;
    } else {
      bufferMsg.textContent = "Waiting for frames...";
    }

    playTimer = setTimeout(playLoop, interval);
  };

  // wait until we have a small buffer built up before starting playback,
  // so brief network hiccups don't immediately cause a stall
  const waitForBuffer = () => {
    if (!playing) return;
    if (frameQueue.length >= TARGET_BUFFER_FRAMES) {
      bufferMsg.textContent = "";
      playLoop();
    } else {
      setTimeout(waitForBuffer, 50);
    }
  };
  waitForBuffer();
}

function uploadWithProgress(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      let data;
      try {
        data = JSON.parse(xhr.responseText);
      } catch (e) {
        reject(new Error("Server returned an invalid response"));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data);
      } else {
        reject(new Error(data.error || `Upload failed (status ${xhr.status})`));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort", () => reject(new Error("Upload aborted")));

    xhr.open("POST", "/upload");
    xhr.send(formData);
  });
}

uploadBtn.addEventListener("click", async () => {
  const file = videoInput.files[0];
  if (!file) {
    statusMsg.textContent = "Please choose a video file first.";
    return;
  }

  uploadBtn.disabled = true;
  streamBox.style.display = "none";
  stopStream();

  progressWrap.style.display = "block";
  progressBar.style.width = "0%";
  progressLabel.textContent = "0%";
  statusMsg.textContent = "Uploading...";

  try {
    const data = await uploadWithProgress(file, (pct) => {
      progressBar.style.width = pct + "%";
      progressLabel.textContent = pct + "%";
      statusMsg.textContent = pct < 100 ? "Uploading..." : "Processing on server...";
    });

    statusMsg.textContent = `Uploaded "${data.filename}". Starting stream...`;
    streamTitle.textContent = `Live Stream: ${data.filename}`;
    streamBox.style.display = "block";
    progressWrap.style.display = "none";

    startStream(data.fps);
  } catch (err) {
    statusMsg.textContent = "Upload failed: " + err.message;
    progressWrap.style.display = "none";
    console.error("Upload error:", err);
  } finally {
    uploadBtn.disabled = false;
  }
});

detectToggle.addEventListener("change", async () => {
  try {
    const res = await fetch("/toggle_detect", { method: "POST" });
    const data = await res.json();
    console.log("detect:", data.detect);
  } catch (err) {
    console.error("Failed to toggle detection:", err);
  } finally {
    // restart the stream connection so the new mode takes effect immediately
    fetch("/status")
      .then((r) => r.json())
      .then((s) => startStream(s.fps));
  }
}); 