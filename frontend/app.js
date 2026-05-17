// =============================================================================
// AVM Smart Track - Multi-Camera Tracking System  (v16)
// Manages physical cameras with independent frame processing, cross-camera
// tracking using cosine similarity on 512D ArcFace embeddings, floor traffic
// analysis, and a verify-before-assign identity pipeline.
// =============================================================================

// Minimum frames a face must be seen before it receives a confirmed ID
const CONFIRM_FRAME_THRESHOLD = 10;

const API_BASE = 'http://localhost:8000/api/v1';
const ENDPOINTS = {
    detect: `${API_BASE}/detection/detect-base64`,
    features: `${API_BASE}/recognition/extract-features`,
    search: `${API_BASE}/recognition/search`,
    health: `${API_BASE}/health/health`,
    auth: `${API_BASE}/auth`,
    tracking: `${API_BASE}/tracking`,
    users: `${API_BASE}/users`,
    alerts: `${API_BASE}/alerts`,
};

const COLORS = {
    WHITE: '#F1F5F9',
    GREEN: '#10B981',
    RED: '#EF4444',
    YELLOW: '#F59E0B',
    BLUE: '#3B82F6',
    CYAN: '#06B6D4',
    PURPLE: '#8B5CF6',
};

// Camera floor assignments (camera index -> floor number)
const CAMERA_FLOORS = {
    0: { floor: 1, name: 'Kamera 1', location: 'Giriş / Kat 1' },
    1: { floor: 2, name: 'Kamera 2', location: 'Kat 2' },
    2: { floor: 3, name: 'Kamera 3', location: 'Kat 3' },
};

// =============================================================================
// AUTH
// =============================================================================

const auth = {
    token: localStorage.getItem('authToken') || null,
    username: localStorage.getItem('authUsername') || null,
    role: localStorage.getItem('authRole') || null,
};

function authHeaders() {
    if (!auth.token) return { 'Content-Type': 'application/json' };
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth.token}`,
    };
}

function isLoggedIn() { return !!auth.token; }

async function doLogin() {
    const username = document.getElementById('loginUser').value.trim();
    const password = document.getElementById('loginPass').value;
    if (!username || !password) { showStatus('Kullanıcı adı ve şifre girin'); return; }

    try {
        const res = await fetch(`${ENDPOINTS.auth}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (!res.ok) { showStatus(`Giriş hatası: ${data.detail || 'Geçersiz bilgiler'}`); return; }

        auth.token = data.access_token;
        auth.username = username;
        localStorage.setItem('authToken', auth.token);
        localStorage.setItem('authUsername', auth.username);

        const me = await fetch(`${ENDPOINTS.auth}/me`, { headers: { 'Authorization': `Bearer ${auth.token}` } });
        if (me.ok) {
            const meData = await me.json();
            auth.role = meData.role;
            localStorage.setItem('authRole', auth.role);
        }

        updateAuthUI();
        showStatus(`Giriş başarılı: ${username} (${auth.role})`);
        registerAllCamerasOnBackend();
    } catch (e) {
        showStatus('Sunucu bağlantı hatası');
    }
}

function doLogout() {
    if (auth.token) {
        fetch(`${ENDPOINTS.auth}/logout`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${auth.token}` },
        }).catch(() => { });
    }
    auth.token = null;
    auth.username = null;
    auth.role = null;
    localStorage.removeItem('authToken');
    localStorage.removeItem('authUsername');
    localStorage.removeItem('authRole');
    updateAuthUI();
    showStatus('Çıkış yapıldı');
}

function updateAuthUI() {
    const section = document.getElementById('authSection');
    const info = document.getElementById('authInfo');
    if (isLoggedIn()) {
        section.style.display = 'none';
        info.style.display = 'flex';
        document.getElementById('authUsername').textContent = auth.username;
        document.getElementById('authRole').textContent = auth.role || '?';
    } else {
        section.style.display = 'flex';
        info.style.display = 'none';
    }
}

function handleAuthError(response) {
    if (response.status === 401) {
        showStatus('Oturum süresi doldu, tekrar giriş yapın');
        doLogout();
        return true;
    }
    if (response.status === 403) {
        showStatus('Yetkiniz yok!');
        return true;
    }
    return false;
}

// =============================================================================
// CAMERA MANAGER - Manages individual camera instances
// =============================================================================

class CameraInstance {
    constructor(index) {
        this.index = index;
        this.config = CAMERA_FLOORS[index];
        this.cameraId = `cam-floor-${this.config.floor}`;

        // DOM elements
        this.video = document.getElementById(`video${index}`);
        this.canvas = document.getElementById(`canvas${index}`);
        this.ctx = this.canvas.getContext('2d');
        this.noFeed = document.getElementById(`noFeed${index}`);
        this.card = document.getElementById(`cameraCard${index}`);

        // State
        this.stream = null;
        this.isRunning = false;
        this.isProcessing = false;
        this.currentFrameData = { faces: [] };
        this.selectedFaceIndex = -1;
        this.deviceId = null;

        // Stats
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 0;

        // Canvas click handler
        this.canvas.addEventListener('click', (e) => this._onCanvasClick(e));
    }

    async start(deviceId) {
        // Try multiple STRICT constraint strategies.
        // We MUST use 'exact' for deviceId, otherwise Chrome quietly falls back to the laptop camera.
        const strategies = [];

        if (deviceId) {
            // Sadece Cihaz ID'si ile hiçbir çözünürlük dayatmadan donanımın kendi varsayılanını bulmasını sağlıyoruz
            strategies.push({
                video: { deviceId: { exact: deviceId } }
            });
        } else {
            // No deviceId - use default camera
            strategies.push({ video: true });
        }

        let lastError = null;

        for (const constraints of strategies) {
            try {
                console.log(`Camera ${this.index}: trying constraints`, JSON.stringify(constraints));
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);

                // Verify we got the right camera
                const trackSettings = this.stream.getVideoTracks()[0]?.getSettings();
                const actualDeviceId = trackSettings?.deviceId;
                console.log(`Camera ${this.index}: opened device=${actualDeviceId?.substring(0, 20)}...`);

                this.video.srcObject = this.stream;
                this.deviceId = actualDeviceId || deviceId;

                // Explicitly call play to prevent browser throttling
                this.video.play().catch(e => console.warn(`Camera ${this.index} play warning:`, e));

                // Wait for video to be ready (at least 50x50 size and HAVE_ENOUGH_DATA)
                await new Promise((resolve, reject) => {
                    let attempts = 0;
                    const check = setInterval(() => {
                        attempts++;
                        if (this.video.readyState === this.video.HAVE_ENOUGH_DATA && this.video.videoWidth > 0) {
                            clearInterval(check);
                            resolve();
                        }
                        if (attempts > 60) { // 6 seconds timeout
                            clearInterval(check);
                            reject(new Error("Video stream timeout - siyah ekran sorunu. Kamera donanımı kilitlendi."));
                        }
                    }, 100);
                });

                this.canvas.width = this.video.videoWidth || 640;
                this.canvas.height = this.video.videoHeight || 480;
                this.isRunning = true;
                this.noFeed.style.display = 'none';
                this.card.classList.add('active');

                // Show actual device label in UI
                const trackLabel = this.stream.getVideoTracks()[0]?.label || '';
                if (trackLabel) {
                    document.getElementById(`camName${this.index}`).textContent = trackLabel.substring(0, 30);
                }

                // Update UI
                const statusChip = document.getElementById(`camStatus${this.index}`);
                statusChip.textContent = 'Aktif';
                statusChip.className = 'camera-status-chip on';

                this._startProcessingLoop();
                console.log(`Camera ${this.index}: started successfully (${trackLabel})`);
                return true;

            } catch (error) {
                lastError = error;
                console.warn(`Camera ${this.index}: strategy failed:`, error.message);
                // Clean up failed attempt
                if (this.stream) {
                    this.stream.getTracks().forEach(t => t.stop());
                    this.stream = null;
                }
            }
        }

        console.error(`Camera ${this.index}: all strategies failed`, lastError);
        showStatus(`Kamera ${this.index + 1} başlatılamadı: ${lastError?.message || 'Bilinmeyen hata'}`);

        // Update UI to show error
        const statusChip = document.getElementById(`camStatus${this.index}`);
        statusChip.textContent = 'Hata';
        statusChip.className = 'camera-status-chip off';

        return false;
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
        }
        this.stream = null;
        this.isRunning = false;
        this.video.srcObject = null;
        this.noFeed.style.display = 'flex';
        this.card.classList.remove('active');
        this.currentFrameData = { faces: [] };

        const statusChip = document.getElementById(`camStatus${this.index}`);
        statusChip.textContent = 'Kapalı';
        statusChip.className = 'camera-status-chip off';

        document.getElementById(`camActive${this.index}`).textContent = '0';
        document.getElementById(`camFps${this.index}`).textContent = '0';
        document.getElementById(`camProcess${this.index}`).textContent = '0ms';
    }

    _startProcessingLoop() {
        let localFrameCount = 0;
        let lastFpsTime = performance.now();

        const loop = () => {
            if (!this.isRunning) return;

            localFrameCount++;
            const now = performance.now();

            if (now - lastFpsTime >= 1000) {
                document.getElementById(`camFps${this.index}`).textContent = localFrameCount;
                this.fps = localFrameCount;
                localFrameCount = 0;
                lastFpsTime = now;
            }

            try {
                if (this.video && this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
                    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

                    // Draw face boxes
                    if (this.currentFrameData && this.currentFrameData.faces) {
                        this.currentFrameData.faces.forEach((face, idx) => {
                            this._drawFaceBox(face, idx);
                        });
                    }
                }
            } catch (e) { /* draw error, skip */ }

            // Process every 10th frame
            if (localFrameCount % 10 === 0 && !this.isProcessing) {
                this._processFrame().catch(() => { });
            }

            requestAnimationFrame(loop);
        };

        requestAnimationFrame(loop);
    }

    _drawFaceBox(face, index) {
        const ctx = this.ctx;
        const isPending = face.trackStatus === 'PENDING';
        const isConfirmed = face.trackStatus === 'CONFIRMED';

        // --- Border colour logic ---
        if (index === this.selectedFaceIndex) {
            ctx.strokeStyle = COLORS.YELLOW;
            ctx.lineWidth = 3;
        } else if (isConfirmed && face.matched) {
            ctx.strokeStyle = COLORS.GREEN;
            ctx.lineWidth = 2;
        } else if (isPending) {
            ctx.strokeStyle = COLORS.PURPLE;
            ctx.lineWidth = 2;
            // Dashed border to signal "still verifying"
            ctx.setLineDash([6, 4]);
        } else {
            ctx.strokeStyle = COLORS.WHITE;
            ctx.lineWidth = 2;
        }

        ctx.strokeRect(face.x, face.y, face.w, face.h);
        ctx.setLineDash([]); // Reset dash

        // --- Label ---
        let label;
        if (isPending) {
            const fc = face.frameCount || 0;
            label = `Doğrulanıyor... (${fc}/${CONFIRM_FRAME_THRESHOLD})`;
        } else if (isConfirmed && face.matched) {
            label = `✓ ${face.name}`;
        } else {
            label = 'Takip Ediliyor...';
        }

        const fontSize = 13;
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const padding = 4;

        if (isPending) {
            ctx.fillStyle = 'rgba(139, 92, 246, 0.85)'; // Purple
        } else if (isConfirmed && face.matched) {
            ctx.fillStyle = 'rgba(16, 185, 129, 0.85)'; // Green
        } else {
            ctx.fillStyle = 'rgba(59, 130, 246, 0.85)'; // Blue
        }
        ctx.fillRect(face.x, face.y - fontSize - 8, textWidth + padding * 2, fontSize + padding + 2);

        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, face.x + padding, face.y - 4);

        // Hash ID below box (only for CONFIRMED)
        if (face.hashId && isConfirmed) {
            ctx.font = `500 10px Inter, sans-serif`;
            ctx.fillStyle = 'rgba(6, 182, 212, 0.9)';
            ctx.fillText(face.hashId, face.x, face.y + face.h + 12);
        }
    }

    async _processFrame() {
        if (this.isProcessing) return;
        this.isProcessing = true;
        const startTime = performance.now();

        try {
            // 1. Detect faces
            const detectionResult = await this._detectFaces();
            if (!detectionResult || detectionResult.faces.length === 0) {
                document.getElementById(`camActive${this.index}`).textContent = '0';
                this.currentFrameData = { faces: [] };
                this.isProcessing = false;
                return;
            }

            const faces = [];
            const trackingDetections = [];

            for (const face of detectionResult.faces) {
                // 2. Extract features
                const featureResult = await this._extractFeatures(face);
                if (!featureResult) continue;

                const faceData = {
                    ...face,
                    embedding: featureResult.embedding,
                    model: featureResult.model_used,
                    matched: false, // Will be updated from tracker response
                    name: 'Takip Ediliyor...',
                    distance: 0,
                    hashId: null,
                };

                faces.push(faceData);

                // Prepare tracking detection
                trackingDetections.push({
                    x: face.x,
                    y: face.y,
                    width: face.w,
                    height: face.h,
                    embedding: featureResult.embedding,
                    name: faceData.matched ? faceData.name : null,
                    confidence: faceData.matched ? (1 - faceData.distance) : 0.0,
                });
            }

            // 4. Send to tracking API
            if (trackingDetections.length > 0) {
                const trackResult = await this._updateTracking(trackingDetections);

                // Map hash IDs + status back to face data
                if (trackResult && trackResult.active_tracks) {
                    trackResult.active_tracks.forEach((track, i) => {
                        if (i < faces.length) {
                            faces[i].trackStatus = track.status || 'PENDING';
                            faces[i].frameCount = track.frame_count || 0;

                            if (track.status === 'CONFIRMED') {
                                faces[i].hashId = track.hash_id;
                                faces[i].matched = true;
                                faces[i].name = track.hash_id;
                            } else {
                                // PENDING - don't show an ID yet
                                faces[i].hashId = null;
                                faces[i].matched = false;
                                faces[i].name = null;
                            }
                        }
                    });
                }
            }

            this.currentFrameData = { faces };
            document.getElementById(`camActive${this.index}`).textContent = faces.length;

            // Auto-select single face
            if (faces.length === 1) {
                this.selectedFaceIndex = 0;
            }

        } catch (error) {
            console.error(`Camera ${this.index} processing error:`, error);
        }

        const elapsed = performance.now() - startTime;
        document.getElementById(`camProcess${this.index}`).textContent = Math.round(elapsed) + 'ms';
        this.isProcessing = false;
    }

    async _detectFaces() {
        try {
            const result = getCanvasAsBase64(this.canvas, 0.7);

            const response = await fetch(ENDPOINTS.detect, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: result.data, min_confidence: 0.5 }),
            });

            const data = await response.json();

            if (data.success && data.face_locations && data.face_locations.length > 0) {
                return {
                    faces: data.face_locations.map(f => ({
                        x: Math.round(f.x / result.scale),
                        y: Math.round(f.y / result.scale),
                        w: Math.round(f.width / result.scale),
                        h: Math.round(f.height / result.scale),
                    })),
                };
            }
            return { faces: [] };
        } catch (error) {
            console.error(`Camera ${this.index} detection error:`, error);
            return null;
        }
    }

    async _extractFeatures(face) {
        try {
            const padding = 30;
            const x1 = Math.max(0, face.x - padding);
            const y1 = Math.max(0, face.y - padding);
            const x2 = Math.min(this.canvas.width, face.x + face.w + padding);
            const y2 = Math.min(this.canvas.height, face.y + face.h + padding);

            const roiCanvas = document.createElement('canvas');
            roiCanvas.width = x2 - x1;
            roiCanvas.height = y2 - y1;

            if (roiCanvas.width < 50 || roiCanvas.height < 50) return null;

            const roiCtx = roiCanvas.getContext('2d');
            roiCtx.drawImage(this.canvas, x1, y1, roiCanvas.width, roiCanvas.height,
                0, 0, roiCanvas.width, roiCanvas.height);

            const imageData = roiCanvas.toDataURL('image/png').split(',')[1];

            const response = await fetch(ENDPOINTS.features, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ face_roi: imageData, model: 'arcface' }),
            });

            const data = await response.json();
            if (data.success) {
                return { embedding: data.embedding, model_used: data.model_used };
            }
            return null;
        } catch (error) {
            console.error(`Camera ${this.index} feature extraction error:`, error);
            return null;
        }
    }

    async _updateTracking(detections) {
        try {
            const response = await fetch(`${ENDPOINTS.tracking}/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camera_id: this.cameraId,
                    detections: detections,
                }),
            });

            const data = await response.json();
            if (data.success) {
                return data;
            }
        } catch (error) {
            console.error(`Camera ${this.index} tracking update error:`, error);
        }
        return null;
    }

    _onCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        if (this.currentFrameData && this.currentFrameData.faces) {
            for (let i = 0; i < this.currentFrameData.faces.length; i++) {
                const face = this.currentFrameData.faces[i];
                if (x >= face.x && x <= face.x + face.w &&
                    y >= face.y && y <= face.y + face.h) {
                    this.selectedFaceIndex = i;
                    // Set as active camera for registration
                    activeCameraIndex = this.index;
                    showStatus(`Kamera ${this.index + 1} - Yüz ${i + 1} seçildi`);
                    break;
                }
            }
        }
    }

    getSelectedFace() {
        if (this.selectedFaceIndex >= 0 && this.currentFrameData && this.currentFrameData.faces) {
            return this.currentFrameData.faces[this.selectedFaceIndex];
        }
        return null;
    }
}

// =============================================================================
// GLOBAL STATE
// =============================================================================

const cameras = [
    new CameraInstance(0),
    new CameraInstance(1),
    new CameraInstance(2),
];

let activeCameraIndex = 0;
let availableDevices = [];
let registeredPeople = {};
let floorReportInterval = null;
let trackRefreshInterval = null;
let alertRefreshInterval = null;

const appState = {
    threshold: 1.0,
    topK: 3,
};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    updateAuthUI();
    await checkApiHealth();
    await loadRegisteredPeople();
    await discoverCameras();

    // Enter key for modal
    document.getElementById('personName').addEventListener('keydown', (e) => {
        if (e.code === 'Enter') {
            e.preventDefault();
            confirmRegister();
        }
    });

    // Space key for registration
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            registerPerson();
        }
    });

    // Start periodic updates
    trackRefreshInterval = setInterval(updateActiveTracksUI, 3000);
    setInterval(updateFloorBarsFromTracking, 2000);

    // Alerts
    await loadWantedPersons();
    await loadActiveAlerts();
    await loadSecurityAlertLog();
    alertRefreshInterval = setInterval(() => {
        loadActiveAlerts();
        loadSecurityAlertLog();
    }, 2000);
});

// =============================================================================
// CAMERA DISCOVERY & MANAGEMENT
// =============================================================================

// Virtual / non-physical camera keywords to skip
const VIRTUAL_CAMERA_KEYWORDS = ['obs', 'virtual', 'snap camera', 'xsplit', 'manycam', 'droidcam'];

function isVirtualCamera(label) {
    const lower = (label || '').toLowerCase();
    return VIRTUAL_CAMERA_KEYWORDS.some(kw => lower.includes(kw));
}

async function discoverCameras() {
    try {
        // First try without permission to see what devices are visible
        let devices = await navigator.mediaDevices.enumerateDevices();
        let videoDevices = devices.filter(d => d.kind === 'videoinput');

        // If labels are empty, we need permission first. 
        // We MUST NOT do the "open stream and stop" hack here, because it locks the laptop camera on Windows.
        if (videoDevices.length > 0 && !videoDevices[0].label) {
            console.log('Camera labels not available. Prompting user naturally.');
            showStatus("Lütfen tarayıcınızın adres çubuğundaki kilit ikonundan kamera izni verip sayfayı yenileyin!");
            // Sadece izin isteyip akışı durduruyoruz, kilitlenmemesi için.
            try {
                const reqStream = await navigator.mediaDevices.getUserMedia({ video: true });
                reqStream.getTracks().forEach(t => t.stop());
                // Yeniden taramak yerine sayfayı yenilemesini isteyeceğiz ki OS cihazı garantili bıraksın.
                setTimeout(() => { window.location.reload(); }, 2000);
                return;
            } catch (e) {
                showStatus("Kamera izni verilmedi!");
                return;
            }
        }

        // Filter out virtual cameras (OBS, Snap Camera, etc.)
        const physicalDevices = videoDevices.filter(d => !isVirtualCamera(d.label));
        const virtualSkipped = videoDevices.length - physicalDevices.length;
        if (virtualSkipped > 0) {
            console.log(`Skipped ${virtualSkipped} virtual camera(s)`);
        }

        videoDevices = physicalDevices;
        availableDevices = videoDevices;

        // Custom Mapping for specific hardware:
        // Kat 1 (Slot 0): Laptop Camera (Integrated)
        // Kat 2 (Slot 1): Brio 300
        // Kat 3 (Slot 2): A4Tech / other USB
        let mappedDevices = [null, null, null];
        let unassigned = [];

        videoDevices.forEach(d => {
            const label = (d.label || '').toLowerCase();
            if (label.includes('brio')) {
                mappedDevices[1] = d;
            } else if (label.includes('a4') || label.includes('tech') || label.includes('1080p') || label.includes('usb')) {
                if (!mappedDevices[2]) mappedDevices[2] = d;
                else unassigned.push(d);
            } else if (label.includes('integrated') || label.includes('webcam') || label.includes('pc camera') || label.includes('laptop')) {
                if (!mappedDevices[0]) mappedDevices[0] = d;
                else unassigned.push(d);
            } else {
                unassigned.push(d);
            }
        });

        // Fill remaining empty slots with any unassigned cameras
        for (let i = 0; i < 3; i++) {
            if (!mappedDevices[i] && unassigned.length > 0) {
                mappedDevices[i] = unassigned.shift();
            }
        }

        // Apply mapped devices back to availableDevices
        availableDevices = mappedDevices.filter(d => d !== null);

        console.log('=== CAMERA MAPPING (Physical Only) ===');
        mappedDevices.forEach((d, i) => {
            if (d) {
                console.log(`  Slot ${i} (Kat ${i + 1}): ${d.label || '(no label)'} | ID: ${d.deviceId.substring(0, 15)}...`);
                document.getElementById(`camName${i}`).textContent = (d.label || `Kamera ${i + 1}`).substring(0, 30);
            } else {
                console.log(`  Slot ${i} (Kat ${i + 1}): BOŞ - Cihaz bulunamadı`);
            }
        });

        // Show camera discovery panel
        renderCameraAssignmentPanel();

        showStatus(`${availableDevices.length} fiziksel kamera bulundu`);
    } catch (e) {
        console.error('Camera discovery error:', e);
        showStatus('Kamera keşfi hatası - tarayıcı ayarlarından izin verin');
    }
}

function renderCameraAssignmentPanel() {
    const panel = document.getElementById('cameraAssignmentPanel');
    if (!panel) return;

    if (availableDevices.length === 0) {
        panel.innerHTML = '<div style="color:var(--accent-red);font-size:12px;">⚠️ Hiç kamera bulunamadı</div>';
        return;
    }

    let html = '<div style="font-size:12px; color:var(--text-muted); margin-bottom:8px;">Bulunan kameralar:</div>';
    availableDevices.forEach((d, i) => {
        const label = d.label || `Kamera (ID: ${d.deviceId.substring(0, 12)}...)`;
        const isAssigned = i < 3;
        html += `<div style="padding:6px 10px; margin-bottom:4px; background:rgba(59,130,246,0.08); border:1px solid var(--border-color); border-radius:6px; font-size:12px; display:flex; justify-content:space-between; align-items:center;">`;
        html += `<span style="color:${isAssigned ? 'var(--accent-green)' : 'var(--text-muted)'}">📷 ${label.substring(0, 35)}</span>`;
        html += `<span style="font-size:11px; color:var(--text-muted);">${isAssigned ? 'Kat ' + CAMERA_FLOORS[i].floor : 'Atanmamış'}</span>`;
        html += `</div>`;
    });

    if (availableDevices.length > 3) {
        html += `<div style="font-size:11px; color:var(--accent-orange); margin-top:6px;">⚠️ ${availableDevices.length - 3} kamera fazla (max 3)</div>`;
    }

    panel.innerHTML = html;
}

async function startAllCameras() {
    showStatus('Önceki bağlantılar temizleniyor...');
    stopAllCameras(); // Clear any existing handles first

    // Let hardware and OS fully release every camera handle
    await new Promise(r => setTimeout(r, 1500));

    showStatus('Kameralar başlatılıyor...');
    document.getElementById('startAllBtn').disabled = true;

    // NOTE: We do NOT call discoverCameras() here again.
    // It was already called on page load (DOMContentLoaded).
    // Calling it again opens a temp stream that locks the device on Windows.
    // If user presses "Kameraları Tara" manually, availableDevices is already fresh.

    if (availableDevices.length === 0) {
        // First-time: discover once
        await discoverCameras();
        // Give the OS time to release the permission-probe stream
        await new Promise(r => setTimeout(r, 2000));
    }

    if (availableDevices.length === 0) {
        showStatus('Hiç fiziksel kamera bulunamadı! USB bağlantılarını kontrol edin.');
        document.getElementById('startAllBtn').disabled = false;
        return;
    }

    const cameraCount = Math.min(3, availableDevices.length);
    console.log(`Starting ${cameraCount} physical camera(s)...`);

    // Start cameras SEQUENTIALLY.
    // Order: Slot 0 (Integrated), then Slot 1 (Brio), then Slot 2
    let startedCount = 0;
    const openedDeviceIds = new Set();
    const startOrder = [0, 1, 2]; // İlk dahiliyi başlat, sonra hariciye geç
    const MAX_RETRIES = 2;

    for (const i of startOrder) {
        const device = availableDevices[i];
        if (!device) continue; // No camera mapped to this slot

        const deviceId = device.deviceId;

        if (openedDeviceIds.has(deviceId)) {
            console.warn(`Camera ${i}: device already opened, skipping`);
            continue;
        }

        showStatus(`Kat ${i + 1} başlatılıyor: ${device.label || 'Bilinmeyen'}...`);

        let success = false;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            success = await cameras[i].start(deviceId);
            if (success) break;
            console.warn(`Camera ${i}: attempt ${attempt}/${MAX_RETRIES} failed, retrying after delay...`);
            await new Promise(r => setTimeout(r, 2000));
        }

        if (success) {
            startedCount++;
            openedDeviceIds.add(cameras[i].deviceId || deviceId);
            console.log(`Camera ${i}: STARTED OK`);
        } else {
            console.warn(`Camera ${i}: FAILED to start after ${MAX_RETRIES} attempts`);
        }

        // 3 s delay between cameras – crucial for Windows USB/UVC drivers to settle
        await new Promise(r => setTimeout(r, 3000));
    }

    // Last resort: if nothing started, try default camera on slot 0
    if (startedCount === 0) {
        showStatus('Varsayılan kamera ile deneniyor...');
        const success = await cameras[0].start(null);
        if (success) startedCount++;
    }

    if (startedCount > 0) {
        document.getElementById('stopAllBtn').disabled = false;
        document.getElementById('captureBtn').disabled = false;
        showStatus(`${startedCount}/${cameraCount} kamera başlatıldı`);

        // Register cameras on backend
        await registerAllCamerasOnBackend();
    } else {
        document.getElementById('startAllBtn').disabled = false;
        showStatus('Hiçbir kamera başlatılamadı! Tarayıcı ayarlarını kontrol edin.');
    }
}

function stopAllCameras() {
    cameras.forEach(cam => cam.stop());
    document.getElementById('startAllBtn').disabled = false;
    document.getElementById('stopAllBtn').disabled = true;
    document.getElementById('captureBtn').disabled = true;
    showStatus('Tüm kameralar durduruldu');
}

async function registerAllCamerasOnBackend() {
    if (!isLoggedIn()) return;

    for (let i = 0; i < 3; i++) {
        const conf = CAMERA_FLOORS[i];
        try {
            await fetch(`${ENDPOINTS.tracking}/cameras/register`, {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({
                    camera_id: `cam-floor-${conf.floor}`,
                    location: conf.location,
                    floor: conf.floor,
                }),
            });
        } catch (e) { /* Camera may already be registered, that's fine */ }
    }
}

// =============================================================================
// FACE PROCESSING UTILITIES
// =============================================================================

function getCanvasAsBase64(canvas, quality) {
    const maxW = 480;
    if (canvas.width > maxW) {
        const scale = maxW / canvas.width;
        const tmp = document.createElement('canvas');
        tmp.width = maxW;
        tmp.height = Math.round(canvas.height * scale);
        tmp.getContext('2d').drawImage(canvas, 0, 0, tmp.width, tmp.height);
        return { data: tmp.toDataURL('image/jpeg', quality || 0.7).split(',')[1], scale };
    }
    return { data: canvas.toDataURL('image/jpeg', quality || 0.7).split(',')[1], scale: 1 };
}

async function searchFace(embedding) {
    try {
        const response = await fetch(ENDPOINTS.search, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                embedding: embedding,
                top_k: appState.topK,
                threshold: appState.threshold,
            }),
        });

        const data = await response.json();
        if (data.success) {
            return { matches: data.matches };
        }
    } catch (error) {
        console.error('Search error:', error);
    }
    return null;
}

// =============================================================================
// REGISTRATION
// =============================================================================

function registerPerson() {
    if (!isLoggedIn()) {
        showStatus('Kişi kaydetmek için önce giriş yapın');
        return;
    }

    // Find first camera with a face
    let selectedCam = cameras[activeCameraIndex];
    let face = selectedCam.getSelectedFace();

    if (!face) {
        // Try all cameras
        for (let i = 0; i < cameras.length; i++) {
            const cam = cameras[i];
            if (cam.currentFrameData && cam.currentFrameData.faces && cam.currentFrameData.faces.length > 0) {
                cam.selectedFaceIndex = 0;
                activeCameraIndex = i;
                face = cam.currentFrameData.faces[0];
                break;
            }
        }
    }

    if (!face || !face.embedding) {
        showStatus('Kaydetmek için önce yüzünüzü kameraya gösterin');
        return;
    }

    const modal = document.getElementById('registerModal');
    modal.classList.add('active');
    const input = document.getElementById('personName');
    input.value = '';
    setTimeout(() => input.focus(), 100);
}

async function confirmRegister() {
    const personName = document.getElementById('personName').value.trim();
    if (!personName) {
        showStatus('Lütfen bir isim girin');
        return;
    }

    const cam = cameras[activeCameraIndex];
    const face = cam.getSelectedFace();
    if (!face || !face.embedding) {
        showStatus('Yüz verisi bulunamadı');
        document.getElementById('registerModal').classList.remove('active');
        return;
    }

    const loading = document.getElementById('loading');
    loading.classList.add('active');

    try {
        const response = await fetch(`${API_BASE}/users/register-face`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({
                name: personName,
                embedding: face.embedding,
            }),
        });

        if (handleAuthError(response)) {
            loading.classList.remove('active');
            document.getElementById('registerModal').classList.remove('active');
            return;
        }

        const result = await response.json();

        if (response.ok && result.success) {
            if (!registeredPeople[personName]) {
                registeredPeople[personName] = { faceCount: 0 };
            }
            registeredPeople[personName].faceCount++;
            localStorage.setItem('registeredPeople', JSON.stringify(registeredPeople));
            updateRegisteredList();
            showStatus(`${personName} başarıyla kaydedildi!`);
        } else {
            showStatus(`Kayıt hatası: ${result.detail || 'Bilinmeyen hata'}`);
        }
    } catch (error) {
        console.error('Registration error:', error);
        showStatus('Sunucu bağlantı hatası');
    }

    loading.classList.remove('active');
    document.getElementById('registerModal').classList.remove('active');
    cam.selectedFaceIndex = -1;
}

function cancelRegister() {
    document.getElementById('registerModal').classList.remove('active');
}

// =============================================================================
// REGISTERED PEOPLE
// =============================================================================

function updateRegisteredList() {
    const list = document.getElementById('registeredList');
    const people = Object.entries(registeredPeople);

    document.getElementById('totalPeople').textContent = people.length;

    if (people.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">👤</div>
                <div>Henüz kimse kayıtlı değil</div>
            </div>`;
        return;
    }

    list.innerHTML = people.map(([name, data]) => `
        <div class="person-item">
            <div>
                <div class="person-name">${name}</div>
                <div class="person-meta">Yüz: ${data.faceCount || 0}</div>
            </div>
            <button class="person-delete-btn" onclick="deletePerson('${name}')">Sil</button>
        </div>
    `).join('');
}

async function deletePerson(name) {
    if (!isLoggedIn()) { showStatus('Silmek için giriş yapın'); return; }
    if (!confirm(`"${name}" silinsin mi?`)) return;

    try {
        await fetch(`${API_BASE}/users/name/${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: authHeaders(),
        });
    } catch (e) { /* */ }

    delete registeredPeople[name];
    localStorage.setItem('registeredPeople', JSON.stringify(registeredPeople));
    updateRegisteredList();
    showStatus(`${name} silindi`);
}

async function loadRegisteredPeople() {
    const stored = localStorage.getItem('registeredPeople');
    if (stored) {
        registeredPeople = JSON.parse(stored);
        updateRegisteredList();
    }
}

function resetDatabase() {
    clearAllData();
}

async function clearAllData() {
    if (!isLoggedIn()) { showStatus('Veritabanı temizlemek için giriş yapın'); return; }
    if (!confirm('Tüm kayıtlı veriler silinecek. Emin misiniz?')) return;

    try {
        const response = await fetch(`${API_BASE}/users/clear-all`, {
            method: 'DELETE',
            headers: authHeaders(),
        });

        if (handleAuthError(response)) return;

        if (response.ok) {
            const data = await response.json();
            registeredPeople = {};
            localStorage.removeItem('registeredPeople');
            updateRegisteredList();
            showStatus(`Veritabanı temizlendi (${data.deleted_count || 0} yüz silindi)`);
        }
    } catch (err) {
        showStatus('Sunucu bağlantı hatası');
    }
}

// =============================================================================
// FLOOR TRAFFIC BARS (LIVE)
// =============================================================================

function updateFloorBarsFromTracking() {
    // Confirmed persons (hash_id known): deduplicate across cameras.
    // Last camera in iteration order wins — so if cam-floor-1 AND cam-floor-2
    // both see the same hash_id simultaneously, the person is counted on floor 2
    // (they have already moved there). Unconfirmed (PENDING) faces have no ID
    // yet, so they are counted per-camera without deduplication.
    const confirmedFloor = new Map(); // hashId -> floor
    const pendingCounts = { 1: 0, 2: 0, 3: 0 };

    cameras.forEach((cam, i) => {
        const floor = CAMERA_FLOORS[i].floor;
        if (!cam.currentFrameData || !cam.currentFrameData.faces) return;
        cam.currentFrameData.faces.forEach(face => {
            if (face.hashId) {
                confirmedFloor.set(face.hashId, floor); // overwrite = latest cam wins
            } else {
                pendingCounts[floor]++;
            }
        });
    });

    const floorCounts = { 1: 0, 2: 0, 3: 0 };
    confirmedFloor.forEach(floor => { floorCounts[floor]++; });
    for (let f = 1; f <= 3; f++) floorCounts[f] += pendingCounts[f];

    const maxCount = Math.max(1, ...Object.values(floorCounts));

    for (let f = 1; f <= 3; f++) {
        const count = floorCounts[f];
        document.getElementById(`floor${f}Count`).textContent = count;
        document.getElementById(`floor${f}Bar`).style.width = `${(count / maxCount) * 100}%`;
    }
}

// =============================================================================
// ACTIVE TRACKS UI
// =============================================================================

async function updateActiveTracksUI() {
    if (!isLoggedIn()) return;

    try {
        const response = await fetch(`${ENDPOINTS.tracking}/tracks`, {
            headers: authHeaders(),
        });

        if (handleAuthError(response)) return;
        if (!response.ok) return;

        const data = await response.json();

        if (!data.success) return;

        const list = document.getElementById('activeTracksList');
        document.getElementById('totalActiveTracks').textContent = data.total;

        if (data.tracks.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🎯</div>
                    <div>Aktif takip yok</div>
                </div>`;
            return;
        }

        list.innerHTML = data.tracks.map(track => {
            const floorClass = `f${track.floor || 1}`;
            return `
                <div class="track-item">
                    <div class="track-header">
                        <span class="track-id">${track.hash_id || track.track_id}</span>
                        <span class="floor-chip ${floorClass}">Kat ${track.floor || '?'}</span>
                    </div>
                    <div class="track-header" style="margin-bottom:0;">
                        <span class="track-name">${track.name || 'Bilinmeyen'}</span>
                        <span class="person-duration" style="color:var(--accent-orange);font-size:11px;">${track.duration_seconds}s</span>
                    </div>
                    <div class="track-meta">
                        <span>📷 ${track.camera_id}</span>
                        <span>🔄 ${track.camera_history ? track.camera_history.length : 0} geçiş</span>
                    </div>
                </div>`;
        }).join('');

    } catch (error) {
        console.error('Track refresh error:', error);
    }
}

// =============================================================================
// FLOOR TRAFFIC REPORT
// =============================================================================

async function loadFloorReport() {
    if (!isLoggedIn()) {
        showStatus('Rapor için giriş yapın');
        return;
    }

    try {
        const response = await fetch(`${ENDPOINTS.tracking}/floor-report`, {
            headers: authHeaders(),
        });

        if (handleAuthError(response)) return;

        const data = await response.json();

        if (!data.success) {
            showStatus('Rapor oluşturulamadı');
            return;
        }

        renderReport(data);
        showStatus('Rapor güncellendi');

    } catch (error) {
        console.error('Report error:', error);
        showStatus('Rapor yüklenirken hata oluştu');
    }
}

function renderReport(data) {
    const container = document.getElementById('reportContent');
    const summary = data.summary || {};
    const floors = data.floor_details || {};
    const persons = data.person_movements || [];

    let html = '';

    // Summary cards
    html += `
        <div class="report-summary">
            <div class="summary-card">
                <div class="value">${summary.total_unique_visitors || 0}</div>
                <div class="label">Toplam Ziyaretçi</div>
            </div>
            <div class="summary-card">
                <div class="value">${summary.busiest_floor || '-'}</div>
                <div class="label">En Yoğun Kat</div>
            </div>
            <div class="summary-card">
                <div class="value">${summary.total_floors_monitored || 0}</div>
                <div class="label">İzlenen Kat</div>
            </div>
        </div>`;

    // Floor details
    html += `<div class="report-section">
        <div class="report-section-title">Kat Detayları</div>`;

    const floorKeys = Object.keys(floors).sort();
    if (floorKeys.length === 0) {
        html += '<div class="empty-state" style="padding:10px;"><div>Henüz kat verisi yok</div></div>';
    } else {
        floorKeys.forEach(key => {
            const f = floors[key];
            html += `
                <div class="movement-item">
                    <div class="person-header">
                        <span class="person-id">Kat ${f.floor}</span>
                        <span class="person-duration">${f.unique_visitors} ziyaretçi</span>
                    </div>
                    <div class="track-meta">
                        <span>Toplam: ${f.total_visits} ziyaret</span>
                        <span>Ort. Süre: ${f.average_duration_minutes}dk</span>
                    </div>
                </div>`;
        });
    }
    html += '</div>';

    // Person movements
    html += `<div class="report-section">
        <div class="report-section-title">Kişi Hareketleri</div>`;

    if (persons.length === 0) {
        html += '<div class="empty-state" style="padding:10px;"><div>Henüz hareket verisi yok</div></div>';
    } else {
        persons.forEach(p => {
            const displayName = p.name || p.hash_id;
            const durations = p.floor_durations || {};

            html += `
                <div class="movement-item">
                    <div class="person-header">
                        <span class="person-id">${displayName}</span>
                        <span class="person-duration">${p.total_duration_minutes || 0}dk</span>
                    </div>
                    <div class="floor-path">`;

            (p.floors_visited || []).forEach((floor, idx) => {
                if (idx > 0) html += '<span class="floor-arrow">→</span>';

                const isMostVisited = (p.most_visited_floor == floor);
                const highlightStyle = isMostVisited ? 'border: 2px solid var(--accent-orange); font-weight: bold;' : '';

                html += `<span class="floor-chip f${floor}" style="${highlightStyle}">Kat ${floor}</span>`;
            });

            html += '</div>';

            if (p.most_visited_floor) {
                html += `<div style="font-size: 11px; margin-top: 5px; color: var(--accent-orange);">📍 En çok Kat ${p.most_visited_floor}'de bulundu</div>`;
            }

            // Per-floor durations
            const durEntries = Object.entries(durations);
            if (durEntries.length > 0) {
                html += '<div class="track-meta" style="margin-top:6px;">';
                durEntries.forEach(([floor, dur]) => {
                    html += `<span>Kat ${floor}: ${dur}s</span>`;
                });
                html += '</div>';
            }

            html += '</div>';
        });
    }
    html += '</div>';

    container.innerHTML = html;
}

// =============================================================================
// STATUS & HEALTH
// =============================================================================

async function checkApiHealth() {
    try {
        const response = await fetch(ENDPOINTS.health);
        const data = await response.json();

        const badge = document.getElementById('statusBadge');
        const text = document.getElementById('statusText');

        if (data.status === 'healthy') {
            badge.classList.remove('disconnected');
            text.textContent = 'API Bağlı';
        } else {
            badge.classList.add('disconnected');
            text.textContent = 'API Sorunlu';
        }
    } catch (error) {
        const badge = document.getElementById('statusBadge');
        const text = document.getElementById('statusText');
        badge.classList.add('disconnected');
        text.textContent = 'API Bağlanamıyor';
    }
}

let statusTimeout = null;

function showStatus(message) {
    const el = document.getElementById('statusMessage');
    el.textContent = message;
    el.classList.add('visible');
    console.log('[STATUS]', message);

    if (statusTimeout) clearTimeout(statusTimeout);
    statusTimeout = setTimeout(() => {
        el.classList.remove('visible');
    }, 4000);
}

// Legacy function name support
function updateStatus(message) { showStatus(message); }

// =============================================================================
// ALERT SYSTEM & WANTED PERSONS
// =============================================================================

// Global variable to hold the dropped/selected file
let _wantedFile = null;

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function parseApiDate(value, epochValue) {
    if (typeof epochValue === 'number') {
        return new Date(epochValue * 1000);
    }
    if (typeof value === 'number') {
        return new Date(value > 1000000000000 ? value : value * 1000);
    }
    if (typeof value === 'string' && value.trim()) {
        const parsed = new Date(value);
        if (!Number.isNaN(parsed.getTime())) return parsed;
    }
    return null;
}

function formatApiDateTime(value, epochValue) {
    const date = parseApiDate(value, epochValue);
    if (!date) return '-';
    return date.toLocaleString('tr-TR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function showAddWantedModal() {
    if (!isLoggedIn()) {
        showStatus('Bu işlem için giriş yapmalısınız');
        return;
    }
    _wantedFile = null;
    document.getElementById('wantedModal').classList.add('active');
    document.getElementById('wantedName').value = '';
    document.getElementById('wantedDesc').value = '';
    document.getElementById('wantedPhoto').value = '';
    // Reset drop zone
    document.getElementById('wantedDropLabel').style.display = 'block';
    document.getElementById('wantedDropPreview').style.display = 'none';
    document.getElementById('wantedDropZone').style.borderColor = 'var(--border)';

    _initDropZone();
}

function cancelAddWanted() {
    _wantedFile = null;
    document.getElementById('wantedModal').classList.remove('active');
}

let _dropZoneInitialized = false;
function _initDropZone() {
    if (_dropZoneInitialized) return;
    _dropZoneInitialized = true;

    const dropZone = document.getElementById('wantedDropZone');
    const fileInput = document.getElementById('wantedPhoto');

    // Click to open file picker (via setTimeout to avoid thread blocking)
    dropZone.addEventListener('click', () => {
        setTimeout(() => fileInput.click(), 0);
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            _handleWantedFile(e.target.files[0]);
        }
    });

    // Drag events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = 'var(--danger)';
        dropZone.style.background = 'oklch(18% .006 222)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            _handleWantedFile(files[0]);
        }
    });
}

function _handleWantedFile(file) {
    // Validate it's an image
    if (!file.type.startsWith('image/')) {
        alert('Lütfen bir görsel dosyası seçin (JPG, PNG)');
        return;
    }
    _wantedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('wantedPreviewImg').src = e.target.result;
        document.getElementById('wantedFileName').textContent = file.name;
        document.getElementById('wantedDropLabel').style.display = 'none';
        document.getElementById('wantedDropPreview').style.display = 'block';
        document.getElementById('wantedDropZone').style.borderColor = 'var(--accent)';
    };
    reader.readAsDataURL(file);
}

async function confirmAddWanted() {
    const nameInput = document.getElementById('wantedName').value.trim();
    const descInput = document.getElementById('wantedDesc').value.trim();

    if (!nameInput || !_wantedFile) {
        alert('İsim ve fotoğraf zorunludur!');
        return;
    }

    document.getElementById('loading').classList.add('active');

    try {
        const formData = new FormData();
        formData.append('file', _wantedFile);

        const url = new URL(`${ENDPOINTS.alerts}/wanted`);
        url.searchParams.append('name', nameInput);
        if (descInput) url.searchParams.append('description', descInput);

        const response = await fetch(url.toString(), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${auth.token}`,
            },
            body: formData,
        });

        const data = await response.json();

        if (handleAuthError(response)) return;

        if (data.success) {
            showStatus('Aranan kişi başarıyla eklendi');
            cancelAddWanted();
            await loadWantedPersons();
        } else {
            alert('Hata: ' + (data.detail || data.message || 'Bilinmeyen hata'));
        }
    } catch (e) {
        console.error('Add wanted person error:', e);
        showStatus('İşlem başarısız');
    } finally {
        document.getElementById('loading').classList.remove('active');
    }
}

async function loadWantedPersons() {
    if (!isLoggedIn()) return;

    try {
        const res = await fetch(`${ENDPOINTS.alerts}/wanted`, { headers: authHeaders() });
        const data = await res.json();
        if (handleAuthError(res)) return;

        const container = document.getElementById('wantedList');
        if (!data.success || !data.wanted_persons || data.wanted_persons.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🛡️</div>
                    <div>Aranan kişi yok</div>
                </div>`;
            return;
        }

        let html = '';
        data.wanted_persons.forEach(p => {
            const addedAt = formatApiDateTime(p.added_at);
            html += `
                <div class="person-item">
                    <div>
                        <div class="person-name" style="color: var(--accent-red);">${escapeHtml(p.name)}</div>
                        <div class="person-meta">${p.description || 'Açıklama yok'}</div>
                        <div class="person-meta">Kayit: ${escapeHtml(addedAt)}</div>
                    </div>
                    <button class="person-delete-btn" onclick="deleteWantedPerson('${escapeHtml(p.wanted_id)}')">Sil</button>
                </div>
            `;
        });
        container.innerHTML = html;

    } catch (e) {
        console.error('Load wanted persons error:', e);
    }
}

async function deleteWantedPerson(id) {
    if (!confirm('Bu kaydı silmek istediğinize emin misiniz?')) return;

    try {
        const res = await fetch(`${ENDPOINTS.alerts}/wanted/${id}`, {
            method: 'DELETE',
            headers: authHeaders(),
        });
        const data = await res.json();
        if (handleAuthError(res)) return;

        if (data.success) {
            showStatus('Aranan kişi silindi');
            await loadWantedPersons();
        }
    } catch (e) {
        showStatus('Silme işlemi başarısız');
    }
}

async function loadActiveAlerts() {
    if (!isLoggedIn()) return;

    try {
        const res = await fetch(`${ENDPOINTS.alerts}/active`, { headers: authHeaders() });
        const data = await res.json();

        if (handleAuthError(res)) {
            clearInterval(alertRefreshInterval);
            return;
        }

        const container = document.getElementById('activeAlertsList');
        const activeCount = data.total || 0;
        document.getElementById('totalAlerts').textContent = activeCount;
        const badge = document.getElementById('alertTabBadge');
        if (badge) {
            badge.textContent = activeCount;
            badge.classList.toggle('visible', activeCount > 0);
        }

        if (!data.success || !data.alerts || data.alerts.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">✅</div>
                    <div>Aktif alarm yok</div>
                </div>`;
            return;
        }

        let html = '';
        data.alerts.forEach(a => {
            const sim = (a.similarity_score * 100).toFixed(1);
            const timeStr = formatApiDateTime(a.timestamp, a.timestamp_epoch);

            html += `
                <div class="track-item" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.05);">
                    <div class="track-header">
                        <span class="track-id" style="color: var(--accent-red);">🚨 ALARM</span>
                        <span class="track-name" style="color: white; font-weight: 600;">${escapeHtml(a.wanted_name)}</span>
                    </div>
                    <div class="track-meta" style="margin-top:4px;">
                        <span style="color: var(--accent-orange);">Sim: %${sim}</span>
                        <span>Kam: ${escapeHtml(a.camera_id)}</span>
                        <span>Kat: ${escapeHtml(a.floor)}</span>
                    </div>
                    <div class="track-meta" style="margin-top:4px; display:flex; justify-content:space-between; align-items:center;">
                        <span>Zaman: ${escapeHtml(timeStr)}</span>
                        <div>
                            <button class="btn btn-sm btn-secondary" onclick="resolveAlert('${escapeHtml(a.alert_id)}')" style="font-size:10px; padding:2px 6px;">Kapat</button>
                        </div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;

    } catch (e) {
        console.error('Load active alerts error:', e);
    }
}

async function loadSecurityAlertLog() {
    if (!isLoggedIn()) return;

    try {
        const res = await fetch(`${ENDPOINTS.alerts}/history?limit=50`, { headers: authHeaders() });
        const data = await res.json();
        if (handleAuthError(res)) return;

        const container = document.getElementById('securityAlertLog');
        const countEl = document.getElementById('secAlertCount');
        if (!container) return;

        const alerts = data.alerts || [];
        if (countEl) countEl.textContent = alerts.length;

        if (!data.success || alerts.length === 0) {
            container.innerHTML = `<div class="empty">Alarm gecmisi yok</div>`;
            return;
        }

        container.innerHTML = alerts.map(a => {
            const sim = ((a.similarity_score || 0) * 100).toFixed(1);
            const timeStr = formatApiDateTime(a.timestamp, a.timestamp_epoch);
            const description = a.wanted_description || 'Aciklama yok';
            return `
                <div class="alert-item">
                    <div class="track-header">
                        <span class="track-id" style="color: var(--danger);">${escapeHtml(a.alert_level || 'ALARM')}</span>
                        <span class="track-name" style="color: var(--text);">${escapeHtml(a.wanted_name || 'Bilinmeyen')}</span>
                    </div>
                    <div class="track-meta" style="margin-top:4px;">
                        <span>${escapeHtml(timeStr)}</span>
                        <span>${escapeHtml(a.camera_id || '-')}</span>
                        <span>Kat ${escapeHtml(a.floor ?? '-')}</span>
                        <span>%${escapeHtml(sim)}</span>
                    </div>
                    <div class="track-meta" style="margin-top:4px;">
                        <span>${escapeHtml(description)}</span>
                        <span>${escapeHtml(a.status || '')}</span>
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.error('Load security alert log error:', e);
    }
}

async function resolveAlert(id) {
    try {
        const res = await fetch(`${ENDPOINTS.alerts}/resolve/${id}`, {
            method: 'POST',
            headers: authHeaders(),
        });
        if (handleAuthError(res)) return;

        const data = await res.json();
        if (data.success) {
            showStatus('Alarm kapatıldı');
            await loadActiveAlerts();
            await loadSecurityAlertLog();
        }
    } catch (e) {
        showStatus('Alarm kapatılamadı');
    }
}
