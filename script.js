import { loadSession, predict, draw, PALETTE } from "./lib/onnxYolo.js";

// ปรับชื่ออาหารและชื่อโมเดลได้ตรงนี้
const TITLE = "Food Snap";
const DEFAULT_NAMES = "Pizza, Sandwich, Sushi";
const EMOJI = ["🍕", "🥪", "🍣"];
const MODEL_URL = "./models/seg.onnx";

document.title = `${TITLE} | YOLO Segmentation`;

const $ = (id) => document.getElementById(id);
const imageInput = $("image-input");
const modelInput = $("model-input");
const dropzone = $("dropzone");
const resultArea = $("result-area");
const canvas = $("result-canvas");
const ctx = canvas.getContext("2d");
const statusEl = $("model-status");
const busyOverlay = $("busy-overlay");
const errorEl = $("error-message");
const predictionResults = $("prediction-results");
const detectionList = $("detection-list");
const confidenceInput = $("confidence");
const confidenceValue = $("confidence-value");
const namesInput = $("class-names");

let session = null;
let modelState = "loading";
let selectedFile = null;
let result = null;
let confidence = 0.25;
let classNames = DEFAULT_NAMES;
let activeJob = 0;

function setModelState(state) {
  modelState = state;
  statusEl.className = `status-pill ${state}`;
  statusEl.innerHTML = `<span class="status-dot"></span>${
    state === "ready" ? "โมเดลพร้อมแล้ว" :
    state === "loading" ? "กำลังโหลดโมเดล…" : "ไม่พบโมเดล"
  }`;
  $("missing-model").classList.toggle("hidden", state !== "missing");
}

async function initializeModel() {
  try {
    session = await loadSession(MODEL_URL);
    setModelState("ready");
    if (selectedFile) runPrediction();
  } catch (error) {
    console.error("Model loading failed:", error);
    setModelState("missing");
  }
}

function chooseFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  result = null;
  dropzone.classList.add("hidden");
  resultArea.classList.remove("hidden");
  predictionResults.classList.add("hidden");
  $("save-png").classList.add("hidden");
  $("save-json").classList.add("hidden");
  errorEl.classList.add("hidden");
  if (session) runPrediction();
  else if (modelState === "missing") {
    errorEl.textContent = "ยังไม่มีโมเดล กรุณาเลือกไฟล์ .onnx ในตั้งค่าเพิ่มเติม";
    errorEl.classList.remove("hidden");
  } else {
    busyOverlay.classList.remove("hidden");
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}

async function runPrediction() {
  if (!selectedFile || !session) return;
  const job = ++activeJob;
  busyOverlay.classList.remove("hidden");
  errorEl.classList.add("hidden");
  try {
    const img = await loadImage(selectedFile);
    const r = await predict(session, img, confidence);
    if (job !== activeJob) return;
    const list = classNames.split(",").map(s => s.trim());
    draw(canvas, img, r.dets, list, "polygon");
    result = { ...r, list, w: img.naturalWidth, h: img.naturalHeight };
    renderResults();
  } catch (error) {
    if (job !== activeJob) return;
    errorEl.textContent = "ทายไม่สำเร็จ: " + (error?.message || error);
    errorEl.classList.remove("hidden");
    predictionResults.classList.add("hidden");
  } finally {
    if (job === activeJob) busyOverlay.classList.add("hidden");
  }
}

function renderResults() {
  const dets = result ? [...result.dets].sort((a,b) => b.conf - a.conf) : [];
  predictionResults.classList.remove("hidden");
  $("result-heading").textContent = dets.length ? `เจอ ${dets.length} อย่างในรูป!` : "🤔 ไม่เจออาหารเลย";
  $("empty-hint").classList.toggle("hidden", dets.length !== 0);
  detectionList.replaceChildren();

  dets.forEach((d) => {
    const li = document.createElement("li");
    li.className = "detection-item";
    const top = document.createElement("div");
    top.className = "detection-top";
    const name = document.createElement("span");
    name.textContent = `${EMOJI[d.cls] ?? "🍽️"} ${result.list[d.cls] ?? d.cls}`;
    const percent = document.createElement("span");
    percent.textContent = `${(d.conf * 100).toFixed(0)}%`;
    percent.style.color = PALETTE[d.cls % PALETTE.length];
    top.append(name, percent);
    const track = document.createElement("div");
    track.className = "conf-track";
    const fill = document.createElement("div");
    fill.className = "conf-fill";
    fill.style.width = `${Math.max(0, Math.min(100, d.conf * 100))}%`;
    track.append(fill);
    li.append(top, track);
    detectionList.append(li);
  });

  $("timing").textContent = `ใช้เวลาทาย ${result.ms} ms · ประมวลผลในเครื่องคุณ ไม่ส่งรูปออกไปไหน`;
  $("save-png").classList.remove("hidden");
  $("save-json").classList.remove("hidden");
}

function downloadBlob(blob, filename) {
  if (!blob) return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function savePng() {
  canvas.toBlob(blob => downloadBlob(blob, "food-snap-result.png"), "image/png");
}

function saveJson() {
  if (!result) return;
  const data = {
    image: { width: result.w, height: result.h },
    detections: result.dets.map(d => ({
      label: result.list[d.cls] ?? d.cls,
      confidence: +d.conf.toFixed(4),
      box: d.box,
      polygon: d.polygon
    }))
  };
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], {type:"application/json"}), "food-snap-result.json");
}

dropzone.addEventListener("click", () => imageInput.click());
dropzone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); imageInput.click(); }
});
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  chooseFile(e.dataTransfer.files[0]);
});
imageInput.addEventListener("change", e => {
  chooseFile(e.target.files[0]);
  e.target.value = "";
});
$("new-image").addEventListener("click", () => imageInput.click());
$("save-png").addEventListener("click", savePng);
$("save-json").addEventListener("click", saveJson);

window.addEventListener("paste", e => {
  const file = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith("image/"));
  if (file) chooseFile(file);
});

confidenceInput.addEventListener("input", () => {
  confidence = Number(confidenceInput.value);
  confidenceValue.textContent = confidence.toFixed(2);
  if (selectedFile && session) runPrediction();
});
namesInput.addEventListener("change", () => {
  classNames = namesInput.value;
  if (selectedFile && session) runPrediction();
});
modelInput.addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    setModelState("loading");
    session = await loadSession(new Uint8Array(await file.arrayBuffer()));
    setModelState("ready");
    if (selectedFile) runPrediction();
  } catch (error) {
    console.error(error);
    setModelState("missing");
    errorEl.textContent = "โหลดโมเดลไม่ได้: " + (error?.message || error);
    errorEl.classList.remove("hidden");
  } finally {
    e.target.value = "";
  }
});

initializeModel();
