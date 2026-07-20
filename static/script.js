const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusMsg = document.getElementById("statusMsg");
const streamBox = document.getElementById("streamBox");
const streamTitle = document.getElementById("streamTitle");
const detectToggle = document.getElementById("detectToggle");
const videoEl = document.getElementById("videoEl");
const connMsg = document.getElementById("connMsg");
const progressWrap = document.getElementById("progressWrap");
const progressBar = document.getElementById("progressBar");
const progressLabel = document.getElementById("progressLabel");

let pc = null;

function stopStream() {
  if (pc) {
    pc.close();
    pc = null;
  }
  videoEl.srcObject = null;
}

async function startWebRTC() {
  stopStream();
  connMsg.textContent = "Connecting...";

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = (event) => {
    videoEl.srcObject = event.streams[0];
    connMsg.textContent = "";
  };

  pc.oniceconnectionstatechange = () => {
    console.log("ICE state:", pc.iceConnectionState);
    if (pc.iceConnectionState === "checking") {
      connMsg.textContent = "Negotiating connection...";
    } else if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "disconnected") {
      connMsg.textContent = "Connection failed - see console/README for NAT/TURN notes.";
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // wait for ICE gathering to complete so the SDP we send includes candidates
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
    } else {
      const check = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", check);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", check);
    }
  });

  const res = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    connMsg.textContent = "Failed to start stream: " + (err.error || res.status);
    return;
  }

  const answer = await res.json();
  await pc.setRemoteDescription(answer);
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

    await startWebRTC();
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
  }
  // no restart needed - detection is checked live per-frame server-side
});