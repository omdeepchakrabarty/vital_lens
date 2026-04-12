const preview = document.getElementById("preview");
const recordBtn = document.getElementById("recordBtn");
const statusEl = document.getElementById("status");

const metricEls = {
  bpm: document.getElementById("bpm"),
  hrv: document.getElementById("hrv"),
  sbp: document.getElementById("sbp"),
  dbp: document.getElementById("dbp"),
};

let stream;
let recorder;
let chunks = [];
let isRecording = false;

async function initCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: 640, height: 480, frameRate: 30 },
  });
  preview.srcObject = stream;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function uploadRecording(blob) {
  const formData = new FormData();
  formData.append("file", blob, "capture.webm");

  const response = await fetch("/api/process", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Upload failed");
  }

  return response.json();
}

function renderMetrics(metrics) {
  ["bpm", "hrv", "sbp", "dbp"].forEach((k) => {
    metricEls[k].textContent = Number(metrics[k]).toFixed(1);
  });
}

async function startRecording() {
  chunks = [];
  recorder = new MediaRecorder(stream, { mimeType: "video/webm" });

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  recorder.onstop = async () => {
    try {
      setStatus("Uploading...");
      const blob = new Blob(chunks, { type: "video/webm" });
      const result = await uploadRecording(blob);
      renderMetrics(result.metrics);
      setStatus("Completed");
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    } finally {
      recordBtn.disabled = false;
      recordBtn.textContent = "Start Recording";
      isRecording = false;
    }
  };

  recorder.start();
  setStatus("Recording (6s)...");
  recordBtn.textContent = "Recording...";
  isRecording = true;

  setTimeout(() => {
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
  }, 6000);
}

recordBtn.addEventListener("click", async () => {
  if (isRecording) return;
  recordBtn.disabled = true;
  await startRecording();
});

initCamera().catch((err) => {
  console.error(err);
  setStatus(`Camera error: ${err.message}`);
  recordBtn.disabled = true;
});
