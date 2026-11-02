// YOLO Detection / Segmentation ในเบราว์เซอร์ด้วย ONNX Runtime Web
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

export const SIZE = 640;
export const PALETTE = ["#FF6B3D", "#2FB36D", "#3D8BFF", "#E55CC0", "#9B6BFF", "#00B8D4"];
export const loadSession = (src) => ort.InferenceSession.create(src, { executionProviders: ["wasm"] });

function letterbox(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const r = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const dx = Math.floor((SIZE - nw) / 2), dy = Math.floor((SIZE - nh) / 2);
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(img, 0, 0, w, h, dx, dy, nw, nh);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = px[i * 4] / 255;
    f[n + i] = px[i * 4 + 1] / 255;
    f[2 * n + i] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]), r, dx, dy };
}

const iouOf = (a, b) => {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u ? inter / u : 0;
};
function nms(cands, thr) {
  cands.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const d of cands) if (keep.every((k) => k.cls !== d.cls || iouOf(k.box, d.box) < thr)) keep.push(d);
  return keep;
}
export function trace(g, w, h) {
  const at = (x, y) => x >= 0 && y >= 0 && x < w && y < h && g[y * w + x] === 1;
  let sx = -1, sy = -1;
  for (let y = 0; y < h && sx < 0; y++) for (let x = 0; x < w; x++) if (at(x, y)) { sx = x; sy = y; break; }
  if (sx < 0) return [];
  const DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, 1, 1, 1, 0, -1, -1, -1];
  const pts = [[sx, sy]];
  let cx = sx, cy = sy, back = 4;
  for (let step = 0; step < w * h; step++) {
    let moved = false;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8, nx = cx + DX[d], ny = cy + DY[d];
      if (at(nx, ny)) {
        const pd = (back + k - 1) % 8, px = cx + DX[pd], py = cy + DY[pd];
        cx = nx; cy = ny;
        for (let j = 0; j < 8; j++) if (cx + DX[j] === px && cy + DY[j] === py) { back = j; break; }
        moved = true; break;
      }
    }
    if (!moved || (cx === sx && cy === sy)) break;
    pts.push([cx, cy]);
  }
  return pts;
}
function rdp(pts, eps) {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0], [bx, by] = pts[pts.length - 1];
  const L = Math.hypot(bx - ax, by - ay) || 1;
  let md = 0, mi = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs((by - ay) * pts[i][0] - (bx - ax) * pts[i][1] + bx * ay - by * ax) / L;
    if (d > md) { md = d; mi = i; }
  }
  if (md > eps) return rdp(pts.slice(0, mi + 1), eps).slice(0, -1).concat(rdp(pts.slice(mi), eps));
  return [pts[0], pts[pts.length - 1]];
}
export function simplifyLoop(pts, eps) {
  if (pts.length < 4) return pts;
  let k = 0, md = 0;
  pts.forEach((p, i) => { const d = Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]); if (d > md) { md = d; k = i; } });
  const a = rdp(pts.slice(0, k + 1), eps);
  const b = rdp(pts.slice(k).concat([pts[0]]), eps);
  return a.slice(0, -1).concat(b.slice(0, -1));
}
export function maskToPolygon(d, P, NM, MW, MH) {
  const s = MW / SIZE;
  const x1 = Math.max(0, Math.floor(d.box[0] * s)), y1 = Math.max(0, Math.floor(d.box[1] * s));
  const x2 = Math.min(MW - 1, Math.ceil(d.box[2] * s)), y2 = Math.min(MH - 1, Math.ceil(d.box[3] * s));
  const g = new Uint8Array(MW * MH), plane = MW * MH;
  for (let y = y1; y <= y2; y++) for (let x = x1; x <= x2; x++) {
    const idx = y * MW + x;
    let v = 0;
    for (let m = 0; m < NM; m++) v += d.coef[m] * P[m * plane + idx];
    if (v > 0) g[idx] = 1;
  }
  return simplifyLoop(trace(g, MW, MH), 1.0);
}
export function decode(out, keys, lb, conf, iouThr) {
  const protoKey = keys.find((k) => out[k].dims.length === 4);
  const predKey = keys.find((k) => out[k].dims.length === 3);
  if (!predKey) throw new Error("ไม่พบ output tensor สำหรับผลตรวจจับ");
  const D = out[predKey].data, [, A, B] = out[predKey].dims;
  const P = protoKey ? out[protoKey].data : null;
  const [, NM, MH, MW] = protoKey ? out[protoKey].dims : [0, 0, 0, 0];
  let cands = [];
  if (A > B) {
    for (let i = 0; i < A; i++) {
      const o = i * B, c = D[o + 4];
      if (c < conf) continue;
      cands.push({ box: [D[o], D[o + 1], D[o + 2], D[o + 3]], conf: c, cls: Math.round(D[o + 5]), coef: D.slice(o + 6, o + 6 + NM) });
    }
  } else {
    const nc = A - 4 - NM, N = B;
    for (let i = 0; i < N; i++) {
      let best = 0, cls = 0;
      for (let c = 0; c < nc; c++) { const s = D[(4 + c) * N + i]; if (s > best) { best = s; cls = c; } }
      if (best < conf) continue;
      const cx = D[i], cy = D[N + i], w = D[2 * N + i], h = D[3 * N + i];
      const coef = new Float32Array(NM);
      for (let m = 0; m < NM; m++) coef[m] = D[(4 + nc + m) * N + i];
      cands.push({ box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], conf: best, cls, coef });
    }
    cands = nms(cands, iouThr);
  }
  return cands.slice(0, 100).map((d) => {
    const polygon = P ? maskToPolygon(d, P, NM, MW, MH).map(([x, y]) => [((x * SIZE) / MW - lb.dx) / lb.r, ((y * SIZE) / MH - lb.dy) / lb.r]) : [];
    const [a, b, c, e] = d.box;
    return { cls: d.cls, conf: d.conf, polygon, box: [(a - lb.dx) / lb.r, (b - lb.dy) / lb.r, (c - lb.dx) / lb.r, (e - lb.dy) / lb.r] };
  });
}
export async function predict(session, img, conf = 0.25, iouThr = 0.45) {
  const t0 = performance.now();
  const lb = letterbox(img);
  const out = await session.run({ [session.inputNames[0]]: lb.tensor });
  const keys = session.outputNames;
  console.log("ONNX outputs:", keys.map((k) => `${k}: [${out[k].dims}]`).join(" | "));
  const dets = decode(out, keys, lb, conf, iouThr);
  return { dets, ms: Math.round(performance.now() - t0) };
}
export function draw(canvas, img, dets, names, mode) {
  const w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(w, h) / 400));
  const fs = Math.max(14, Math.round(Math.max(w, h) / 45));
  dets.forEach((d) => {
    const col = PALETTE[d.cls % PALETTE.length];
    if (mode === "polygon" && d.polygon.length > 2) {
      ctx.beginPath();
      d.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = col + "66"; ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke();
    } else {
      ctx.strokeStyle = col; ctx.lineWidth = lw;
      ctx.strokeRect(d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]);
    }
    const label = `${names[d.cls] ?? d.cls} ${(d.conf * 100).toFixed(0)}%`;
    ctx.font = `600 ${fs}px sans-serif`;
    const tw = ctx.measureText(label).width + 8, ty = Math.max(0, d.box[1] - fs - 6);
    ctx.fillStyle = col; ctx.fillRect(d.box[0], ty, tw, fs + 6);
    ctx.fillStyle = "#fff"; ctx.fillText(label, d.box[0] + 4, ty + fs);
  });
}
