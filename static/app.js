/**
 * Driver Safety Dashboard - Web Client
 * =====================================
 * Client-side MediaPipe Face Landmarker detection.
 * Calculates EAR/MAR in browser and POSTs to server.
 */

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

(async () => {
    // ── DOM Elements ──
    const $btnStart      = document.getElementById('btn-start');
    const $btnStop       = document.getElementById('btn-stop');
    const $videoFeed     = document.getElementById('webcam');
    const $canvas        = document.getElementById('output-canvas');
    const $placeholder   = document.getElementById('feed-placeholder');
    const $statusLabel   = document.getElementById('status-label');
    const $statusDot     = document.getElementById('status-pulse');

    $btnStart.disabled = true;
    $btnStart.textContent = "LOADING AI MODULE...";
    $btnStart.style.opacity = "0.5";
    $btnStart.style.cursor = "not-allowed";

    // ── MediaPipe Core ──
    let faceLandmarker;
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
        
        $btnStart.disabled = false;
        $btnStart.textContent = "ENGAGE MONITOR";
        $btnStart.style.opacity = "1";
        $btnStart.style.cursor = "pointer";
    } catch (e) {
        console.error("Failed to load FaceLandmarker:", e);
        $btnStart.textContent = "LOAD FAILED. REFRESH.";
        return;
    }
    
    const $valEar = document.getElementById('val-ear');
    const $valMar = document.getElementById('val-mar');
    const $barEar = document.getElementById('bar-ear');
    const $barMar = document.getElementById('bar-mar');
    const $cardEar = document.getElementById('card-ear');
    const $cardMar = document.getElementById('card-mar');
    
    const $sessionTimer  = document.getElementById('session-timer');
    const $cntDrowsy     = document.getElementById('count-drowsy');
    const $cntYawn       = document.getElementById('count-yawn');
    const $cntDistract   = document.getElementById('count-distract');
    const $alertHistory  = document.getElementById('alert-history');

    const canvasCtx = $canvas.getContext('2d');
    let lastVideoTime = -1;

    // ── Geometry ──
    const dist = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    const calculateEar = (pts, ids) => {
        const p = ids.map(i => pts[i]);
        return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]));
    };
    const calculateMar = (pts, ids) => dist(pts[ids[0]], pts[ids[1]]) / dist(pts[ids[2]], pts[ids[3]]);

    // ── Detection Loop ──
    async function predict() {
        if (!$videoFeed.currentTime || $videoFeed.currentTime === lastVideoTime) {
            window.requestAnimationFrame(predict);
            return;
        }
        lastVideoTime = $videoFeed.currentTime;

        const results = faceLandmarker.detectForVideo($videoFeed, Date.now());
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const pts = results.faceLandmarks[0];
            const ear = calculateEar(pts, [33, 160, 158, 133, 153, 144]);
            const mar = calculateMar(pts, [13, 14, 78, 308]);

            // Instant UI Update
            $valEar.textContent = ear.toFixed(3);
            $valMar.textContent = mar.toFixed(3);
            $barEar.style.width = Math.min(ear * 200, 100) + '%';
            $barMar.style.width = Math.min(mar * 150, 100) + '%';
            
            $cardEar.classList.toggle('bg-rose-500/10', ear < 0.22);
            $cardMar.classList.toggle('bg-amber-500/10', mar > 0.5);

            // POST Telemetry to Server
            if (Math.floor(Date.now() / 500) % 2 === 0) {
                syncMetrics(ear, mar);
            }
        }
        window.requestAnimationFrame(predict);
    }

    async function syncMetrics(ear, mar) {
        try {
            const res = await fetch('/api/metrics/report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ear, mar })
            });
            const data = await res.json();
            
            // Sync Session Info
            if (data.session) {
                $sessionTimer.textContent = data.session.timer;
                $cntDrowsy.textContent = data.session.counts.DROWSINESS || 0;
                $cntYawn.textContent = data.session.counts.YAWNING || 0;
                $cntDistract.textContent = data.session.counts.DISTRACTED || 0;
            }

            // Sync Nav Status
            if (data.status !== 'SAFE') {
                $statusLabel.textContent = `ALERT: ${data.status}`;
                $statusDot.className = 'status-dot danger';
            } else {
                $statusLabel.textContent = 'MONITORING LIVE';
                $statusDot.className = 'status-dot safe';
            }
        } catch (e) {}
    }

    // ── Controls ──
    $btnStart.onclick = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        $videoFeed.srcObject = stream;
        $videoFeed.onloadeddata = () => {
            $videoFeed.classList.remove('opacity-0');
            $placeholder.classList.add('hidden');
            $btnStart.classList.add('hidden');
            $btnStop.classList.remove('hidden');
            predict();
        };
        fetch('/api/system/toggle', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action:'start'})});
    };

    $btnStop.onclick = () => {
        if ($videoFeed.srcObject) $videoFeed.srcObject.getTracks().forEach(t => t.stop());
        $videoFeed.classList.add('opacity-0');
        $placeholder.classList.remove('hidden');
        $btnStart.classList.remove('hidden');
        $btnStop.classList.add('hidden');
        fetch('/api/system/toggle', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action:'stop'})});
    };

    // ── Background Polling for Alerts History ──
    setInterval(async () => {
        const res = await fetch('/api/alerts');
        const data = await res.json();
        if (data.history) {
            $alertHistory.innerHTML = data.history.map(e => `
                <div class="flex flex-col gap-2 p-3 bg-white/5 rounded-xl border border-white/5">
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] font-black uppercase text-white">${e.type}</span>
                        <span class="text-[10px] text-slate-500">${e.time}</span>
                    </div>
                </div>
            `).join('');
        }
    }, 3000);

})();
