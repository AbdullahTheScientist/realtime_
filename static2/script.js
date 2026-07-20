const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusMsg = document.getElementById("statusMsg");
const streamBox = document.getElementById("streamBox");
const streamTitle = document.getElementById("streamTitle");
const videoStream = document.getElementById("videoStream");
const detectToggle = document.getElementById("detectToggle");

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
  videoStream.src = ""; // stop any previous stream

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

    // cache-bust so the browser opens a fresh MJPEG connection
    videoStream.src = "/video_feed?t=" + Date.now();
    streamBox.style.display = "block";
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
  } finally {
    // restart the stream connection so the new mode takes effect immediately
    videoStream.src = "/video_feed?t=" + Date.now();
  }
});