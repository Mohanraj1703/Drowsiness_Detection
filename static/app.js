/**
 * Driver Safety System — Dashboard JavaScript
 * Polls /api/metrics every 500ms and /api/alerts every 2s
 * Updates all UI elements in real-time.
 */

const METRICS_INTERVAL = 500;   // ms
const ALERTS_INTERVAL  = 2000;  // ms

// ── DOM refs ──────────────────────────────────────────────
const $statusPulse   = document.getElementById('status-pulse');
const $statusLabel   = document.getElementById('status-label');
const $statusDot     = document.getElementById('status-dot');
const $sessionTimer  = document.getElementById('session-timer');

const $btnStart      = document.getElementById('btn-start');
const $btnStop       = document.getElementById('btn-stop');
const $videoStream   = document.getElementById('video-feed');

const $valEar   = document.getElementById('val-ear');
const $valMar   = document.getElementById('val-mar');
const $valGaze  = document.getElementById('val-gaze');

const $barEar   = document.getElementById('bar-ear');
const $barMar   = document.getElementById('bar-mar');
const $gazeIndicator = document.getElementById('gaze-indicator');

const $cardEar   = document.getElementById('card-ear');
const $cardMar   = document.getElementById('card-mar');
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

// ── System Controls ───────────────────────────────────────

async function toggleSystem(action) {
  try {
    const res = await fetch('/api/system/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
    const data = await res.json();
    if (data.status === 'success') {
      updateSystemUI(data.system_running);
      
      // Force stream refresh if starting
      if (action === 'start') {
        $statusLabel.textContent = 'Hardware Initializing...';
        $statusLabel.className = 'text-xs font-bold uppercase tracking-widest text-amber-500 animate-pulse';
        $videoStream.src = '/video_feed?t=' + Date.now();
      } else {
        $videoStream.src = ''; 
      }
    }
  } catch (err) {
    console.error("Failed to toggle system", err);
  }
}

function updateSystemUI(isRunning) {
  if (isRunning) {
    $btnStart.classList.add('hidden');
    $btnStop.classList.remove('hidden');
    $statusLabel.textContent = 'System Active • Monitoring Live';
    $statusLabel.className = 'text-xs font-bold uppercase tracking-widest text-emerald-400';
    $statusPulse.className = 'w-3 h-3 rounded-full bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.8)] animate-pulse';
  } else {
    $btnStart.classList.remove('hidden');
    $btnStop.classList.add('hidden');
    $statusLabel.textContent = 'System Ready • Standby';
    $statusLabel.className = 'text-xs font-bold uppercase tracking-widest text-slate-500';
    $statusPulse.className = 'w-3 h-3 rounded-full bg-slate-700 shadow-none';
  }
}

$btnStart.onclick = () => toggleSystem('start');
$btnStop.onclick  = () => toggleSystem('stop');

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
  updateSystemUI(d.system_running);
  $sessionTimer.textContent = formatSeconds(d.session_seconds);

  if (!d.system_running) {
    [$valEar, $valMar, $valGaze].forEach(el => el.textContent = '—');
    [$barEar, $barMar].forEach(el => el.style.width = '0%');
    return;
  }

  // EAR
  const earPct = clamp(d.ear / 0.5 * 100, 0, 100);
  $valEar.textContent = d.ear.toFixed(3);
  $barEar.style.width = earPct + '%';
  $cardEar.classList.toggle('border-rose-500/50', d.is_drowsy);
  $cardEar.classList.toggle('bg-rose-500/10', d.is_drowsy);

  // MAR
  const marPct = clamp(d.mar * 100, 0, 100);
  $valMar.textContent = d.mar.toFixed(3);
  $barMar.style.width = marPct + '%';
  $cardMar.classList.toggle('border-amber-500/50', d.is_yawning);

  // Gaze
  $valGaze.textContent = (d.gaze * 100).toFixed(0) + '%';
  $gazeIndicator.style.left = clamp(d.gaze * 100, 5, 95) + '%';
  $cardGaze.classList.toggle('border-indigo-500/50', d.is_distracted);

  // Nav dot
  updateNavDot(d.status);
}

function updateNavDot(status) {
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
  $countDrowsy.textContent   = data.counts.DROWSINESS || 0;
  $countYawn.textContent     = data.counts.YAWNING    || 0;
  $countDistract.textContent = data.counts.DISTRACTED || 0;

  const history = data.history || [];
  if (history.length === _lastHistoryLength) return;
  _lastHistoryLength = history.length;

  if (history.length === 0) {
    $alertHistory.innerHTML = `
      <div class="history-empty text-center py-10">
        <div class="text-2xl mb-2 opacity-20">🛡️</div>
        <p class="text-[10px] font-bold text-slate-600 uppercase tracking-widest">No Alerts Detected</p>
      </div>`;
    return;
  }

  $alertHistory.innerHTML = history.map(entry => `
    <div class="flex flex-col gap-2 p-3 bg-white/5 rounded-xl border border-white/5 group hover:border-white/10 transition-all">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="w-1.5 h-1.5 rounded-full ${entry.type === 'DROWSINESS' ? 'bg-rose-500' : 'bg-amber-500'}"></span>
          <span class="text-[11px] font-bold uppercase tracking-wider text-white">${entry.type}</span>
        </div>
        <span class="text-[10px] font-medium text-slate-500 tabular-nums">${entry.time}</span>
      </div>
      ${entry.image ? `
        <a href="/alerts/${entry.image}" target="_blank" class="block mt-2 overflow-hidden rounded-xl border border-white/10 group-hover:border-indigo-500/30 transition-all">
          <img src="/alerts/${entry.image}" loading="lazy" class="w-full h-16 object-cover opacity-80 hover:opacity-100 hover:scale-110 transition-all duration-500" alt="Incident Capture">
        </a>
      ` : ''}
    </div>
  `).join('');
}

// ── Init ──────────────────────────────────────────────────

fetchMetrics();
fetchAlerts();
setInterval(fetchMetrics, METRICS_INTERVAL);
setInterval(fetchAlerts,  ALERTS_INTERVAL);
