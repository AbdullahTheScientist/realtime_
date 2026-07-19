const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusMsg = document.getElementById("statusMsg");
const streamBox = document.getElementById("streamBox");
const streamTitle = document.getElementById("streamTitle");
const videoStream = document.getElementById("videoStream");
const streamStatus = document.getElementById("streamStatus");
const detectToggle = document.getElementById("detectToggle");

let pc = null;

function closeConnection() {
  if (pc) {
    pc.close();
    pc = null;
  }
  videoStream.srcObject = null;
}

// aiortc's /offer endpoint answers with a complete (non-trickle) SDP, so we
// wait for ICE gathering to finish before sending the offer - this is the
// standard pattern for a single-shot offer/answer exchange.
function waitForIceGatheringComplete(peerConnection) {
  if (peerConnection.iceGatheringState === "complete") {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    function check() {
      if (peerConnection.iceGatheringState === "complete") {
        peerConnection.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    }
    peerConnection.addEventListener("icegatheringstatechange", check);
  });
}

async function startStream() {
  closeConnection();
  streamStatus.textContent = "Connecting...";

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.ontrack = (event) => {
    videoStream.srcObject = event.streams[0];
    streamStatus.textContent = "";
  };

  pc.onconnectionstatechange = () => {
    if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
      streamStatus.textContent = "Stream connection lost.";
    } else if (pc.connectionState === "closed") {
      streamStatus.textContent = "Stream ended.";
    }
  };

  pc.addTransceiver("video", { direction: "recvonly" });

  const offerDesc = await pc.createOffer();
  await pc.setLocalDescription(offerDesc);
  await waitForIceGatheringComplete(pc);

  try {
    const res = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });
    const answer = await res.json();

    if (!res.ok) {
      streamStatus.textContent = answer.error || "Failed to start stream.";
      return;
    }

    await pc.setRemoteDescription(answer);
  } catch (err) {
    streamStatus.textContent = "Connection failed: " + err.message;
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
  closeConnection();

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
    await startStream();
  } catch (err) {
    statusMsg.textContent = "Upload failed: " + err.message;
  } finally {
    uploadBtn.disabled = false;
  }
});

detectToggle.addEventListener("change", async () => {
  try {
    // The video track reads the shared "detect" flag on every frame, so
    // toggling it takes effect on the next frame - no reconnect needed.
    await fetch("/toggle_detect", { method: "POST" });
  } catch (err) {
    console.error("Failed to toggle detection:", err);
  }
});