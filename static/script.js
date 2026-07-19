const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusMsg = document.getElementById("statusMsg");
const streamBox = document.getElementById("streamBox");
const streamTitle = document.getElementById("streamTitle");
const videoCanvas = document.getElementById("videoCanvas");
const streamStatus = document.getElementById("streamStatus");
const detectToggle = document.getElementById("detectToggle");
const ctx = videoCanvas.getContext("2d");

let socket = null;

function closeSocket() {
  if (socket) {
    socket.onclose = null; // avoid firing the "disconnected" message on a deliberate close
    socket.close();
    socket = null;
  }
}

function startStream() {
  closeSocket();

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${proto}//${window.location.host}/ws/video_feed`);
  socket.binaryType = "blob";

  socket.onopen = () => {
    streamStatus.textContent = "";
  };

  socket.onmessage = async (event) => {
    // Text messages are error payloads (e.g. "no video uploaded"); binary
    // messages are WebP frames to draw straight onto the canvas.
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      streamStatus.textContent = msg.error || "";
      return;
    }

    try {
      const bitmap = await createImageBitmap(event.data);
      if (videoCanvas.width !== bitmap.width || videoCanvas.height !== bitmap.height) {
        videoCanvas.width = bitmap.width;
        videoCanvas.height = bitmap.height;
      }
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
    } catch (err) {
      console.error("Failed to decode frame:", err);
    }
  };

  socket.onerror = () => {
    streamStatus.textContent = "Stream connection error.";
  };

  socket.onclose = () => {
    streamStatus.textContent = "Stream ended.";
  };
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
  closeSocket();

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
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
    // The WebSocket loop reads the shared "detect" flag on every frame, so
    // toggling it takes effect on the next frame with no reconnect needed.
    await fetch("/toggle_detect", { method: "POST" });
  } catch (err) {
    console.error("Failed to toggle detection:", err);
  }
});