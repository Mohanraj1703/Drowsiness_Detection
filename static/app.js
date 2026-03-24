/**
 * Driver Safety System — Dashboard JavaScript
 * Polls /api/metrics every 500ms and /api/alerts every 2s
 * Updates all UI elements in real-time.
 */

const METRICS_INTERVAL = 500;   // ms
const ALERTS_INTERVAL  = 2000;  // ms

// ── DOM refs ──────────────────────────────────────────────
const $statusBanner  = document.getElementById('status-banner');
const $statusText    = document.getElementById('status-text');
const $statusIcon    = document.getElementById('status-icon');
const $statusDot     = document.getElementById('status-dot');
const $sessionTimer  = document.getElementById('session-timer');

const $valEar   = document.getElementById('val-ear');
const $valMar   = document.getElementById('val-mar');
const $valPitch = document.getElementById('val-pitch');
const $valYaw   = document.getElementById('val-yaw');
const $valGaze  = document.getElementById('val-gaze');

const $barEar   = document.getElementById('bar-ear');
const $barMar   = document.getElementById('bar-mar');
const $barPitch = document.getElementById('bar-pitch');
const $barYaw   = document.getElementById('bar-yaw');

const $gazeIndicator = document.getElementById('gaze-indicator');

const $cardEar   = document.getElementById('card-ear');
const $cardMar   = document.getElementById('card-mar');
const $cardPitch = document.getElementById('card-pitch');
const $cardYaw   = document.getElementById('card-yaw');
const $cardGaze  = document.getElementById('card-gaze');

const $countDrowsy   = document.getElementById('count-drowsy');
const $countYawn     = document.getElementById('count-yawn');
const $countDistract = document.getElementById('count-distract');
const $alertHistory  = document.getElementById('alert-history');

// ── Helpers ───────────────────────────────────────────────

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function formatSeconds(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':');
}

function setAlert(el, active) {
  if (active) el.classList.add('alert');
  else        el.classList.remove('alert');
}

// ── Metrics Polling ───────────────────────────────────────

async function fetchMetrics() {
  try {
    const res  = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    updateMetrics(data);
  } catch (_) { /* server restarting — ignore */ }
}

function updateMetrics(d) {
  // Session timer
  $sessionTimer.textContent = formatSeconds(d.session_seconds);

  // EAR — danger if below 0.25, normalise 0→0.5 as 0→100%
  const earPct  = clamp(d.ear / 0.5 * 100, 0, 100);
  const earAlert = d.is_drowsy;
  $valEar.textContent = d.ear.toFixed(3);
  $barEar.style.width = earPct + '%';
  setAlert($valEar, earAlert);
  setAlert($barEar, earAlert);
  setAlert($cardEar, earAlert);

  // MAR — danger if above 0.6, normalise 0→1 as 0→100%
  const marPct   = clamp(d.mar * 100, 0, 100);
  const marAlert = d.is_yawning;
  $valMar.textContent = d.mar.toFixed(3);
  $barMar.style.width = marPct + '%';
  setAlert($valMar, marAlert);
  setAlert($barMar, marAlert);
  setAlert($cardMar, marAlert);

  // Pitch — range -90..90, centre at 50%
  const pitchPct   = clamp((d.pitch + 90) / 180 * 100, 0, 100);
  const pitchAlert = d.is_distracted && (d.pitch < -20);
  $valPitch.textContent = d.pitch.toFixed(1) + '°';
  $barPitch.style.width = Math.abs(d.pitch) / 90 * 50 + '%';
  $barPitch.style.marginLeft = d.pitch < 0 ? (50 - Math.abs(d.pitch) / 90 * 50) + '%' : '50%';
  setAlert($valPitch, pitchAlert);
  setAlert($cardPitch, pitchAlert);

  // Yaw — range -90..90
  const yawAlert = d.is_distracted && (Math.abs(d.yaw) > 12);
  $valYaw.textContent = d.yaw.toFixed(1) + '°';
  $barYaw.style.width = Math.abs(d.yaw) / 90 * 50 + '%';
  $barYaw.style.marginLeft = d.yaw < 0 ? (50 - Math.abs(d.yaw) / 90 * 50) + '%' : '50%';
  setAlert($valYaw, yawAlert);
  setAlert($cardYaw, yawAlert);

  // Gaze — 0..1, indicator moves L→R across the track
  const gazeAlert = d.is_distracted && (d.gaze < 0.42 || d.gaze > 0.58);
  $valGaze.textContent = d.gaze.toFixed(3);
  $gazeIndicator.style.left = clamp(d.gaze * 100, 2, 98) + '%';
  setAlert($gazeIndicator, gazeAlert);
  setAlert($cardGaze, gazeAlert);

  // Status banner + dot
  updateStatus(d.status, d.is_drowsy, d.is_yawning, d.is_distracted);
}

const STATUS_MAP = {
  SAFE:       { cls: 'safe',       icon: '✅', text: 'Driver Alert & Safe' },
  DROWSY:     { cls: 'drowsy',     icon: '😴', text: 'DROWSINESS DETECTED — WAKE UP!' },
  YAWNING:    { cls: 'yawning',    icon: '🥱', text: 'Yawning Detected — Take a Break' },
  DISTRACTED: { cls: 'distracted', icon: '👀', text: 'Distraction Detected — Eyes on Road!' },
};

function updateStatus(status) {
  const config = STATUS_MAP[status] || STATUS_MAP['SAFE'];

  // Banner
  $statusBanner.className = 'status-banner ' + config.cls;
  $statusIcon.textContent = config.icon;
  $statusText.textContent = config.text;

  // Nav dot
  $statusDot.className = 'status-dot ' + (
    status === 'SAFE' ? 'safe' :
    status === 'DROWSY' ? 'danger' : 'warning'
  );
}

// ── Alerts Polling ────────────────────────────────────────

let _lastHistoryLength = 0;

async function fetchAlerts() {
  try {
    const res  = await fetch('/api/alerts');
    if (!res.ok) return;
    const data = await res.json();
    updateAlerts(data);
  } catch (_) {}
}

function updateAlerts(data) {
  // Counts
  $countDrowsy.textContent   = data.counts.DROWSINESS || 0;
  $countYawn.textContent     = data.counts.YAWNING    || 0;
  $countDistract.textContent = data.counts.DISTRACTED || 0;

  // History list
  const history = data.history || [];
  if (history.length === _lastHistoryLength) return;   // nothing new
  _lastHistoryLength = history.length;

  if (history.length === 0) {
    $alertHistory.innerHTML = '<div class="history-empty">No alerts yet — drive safe! 🚗</div>';
    return;
  }

  $alertHistory.innerHTML = history.map(entry => `
    <div class="history-item">
      <span class="history-badge badge-${entry.type}">${entry.type}</span>
      <span class="history-time">${entry.time}</span>
    </div>
  `).join('');
}

// ── Init ──────────────────────────────────────────────────

fetchMetrics();
fetchAlerts();
setInterval(fetchMetrics, METRICS_INTERVAL);
setInterval(fetchAlerts,  ALERTS_INTERVAL);
