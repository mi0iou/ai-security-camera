#!/usr/bin/env python3
"""
AI Security Camera Dashboard
Modern UI with live video feed from shared frame buffer
Reads frames published by main.py - no direct camera access
"""

from flask import Flask, render_template_string, jsonify, Response
from database_manager import DatabaseManager
from frame_buffer import frame_buffer  # Shared buffer with main.py
from collections import Counter
import traceback
import time

app = Flask(__name__)
db = DatabaseManager('database/security.db')


def generate_mjpeg():
    """Generator for MJPEG stream from shared frame buffer"""
    while True:
        frame = frame_buffer.get_frame()
        
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)  # ~20 FPS max for browser
        else:
            # No frame available yet - wait and retry
            time.sleep(0.2)


@app.route('/video_feed')
def video_feed():
    """Video streaming route - reads from shared buffer"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stream_status')
def stream_status():
    """Check if frames are being received"""
    age = frame_buffer.get_frame_age()
    stats = frame_buffer.get_stats()
    return jsonify({
        'active': age < 5.0,
        'frame_age': age,
        'stats': stats
    })


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset all counters"""
    try:
        db.reset_statistics()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """API endpoint for database stats (plates, alerts)"""
    try:
        stats = db.get_statistics(24)
        return jsonify({
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live_counts')
def api_live_counts():
    """API endpoint for live detection counts from current frame"""
    meta = frame_buffer.get_meta()
    if meta:
        return jsonify({
            'live_counts': meta.get('live_counts', {}),
            'total': meta.get('detection_count', 0)
        })
    return jsonify({'live_counts': {}, 'total': 0})


@app.route('/api/events')
def api_events():
    """API endpoint for recent events"""
    try:
        events = db.get_recent_events(hours=24, limit=50)
        return jsonify({
            'events': events
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def dashboard():
    try:
        stats = db.get_statistics(24)
        events = db.get_recent_events(hours=24, limit=50)
        plates = db.get_all_known_plates()
        
        html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Security Command Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-card-hover: #22222e;
            --accent-cyan: #00f0ff;
            --accent-green: #00ff88;
            --accent-red: #ff3366;
            --accent-orange: #ff9500;
            --accent-purple: #9966ff;
            --text-primary: #ffffff;
            --text-secondary: #8888aa;
            --text-muted: #555566;
            --border-color: #2a2a3a;
            --glow-cyan: 0 0 20px rgba(0, 240, 255, 0.3);
            --glow-green: 0 0 20px rgba(0, 255, 136, 0.3);
            --glow-red: 0 0 20px rgba(255, 51, 102, 0.4);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: var(--glow-cyan);
        }
        
        .logo h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .logo p {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .header-controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .reset-btn {
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--accent-red);
            border-radius: 30px;
            color: var(--accent-red);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .reset-btn:hover {
            background: rgba(255, 51, 102, 0.2);
            box-shadow: var(--glow-red);
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--accent-green);
            border-radius: 30px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent-green);
            box-shadow: var(--glow-green);
            transition: all 0.3s ease;
        }
        
        .status-badge.offline {
            border-color: var(--accent-red);
            color: var(--accent-red);
            box-shadow: var(--glow-red);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-badge.offline .status-dot {
            background: var(--accent-red);
            animation: none;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 25px;
        }
        
        @media (max-width: 1400px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .video-section {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }
        
        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .section-title span { font-size: 1.2rem; }
        
        .live-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            background: rgba(255, 51, 102, 0.2);
            border: 1px solid var(--accent-red);
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-red);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .live-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--accent-red);
            border-radius: 50%;
            animation: pulse 1s infinite;
        }
        
        .live-badge.inactive {
            background: rgba(85, 85, 102, 0.2);
            border-color: var(--text-muted);
            color: var(--text-muted);
        }
        
        .live-badge.inactive::before {
            background: var(--text-muted);
            animation: none;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
        }
        
        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0, 0, 0, 0.8);
            color: var(--text-secondary);
            flex-direction: column;
            gap: 15px;
        }
        
        .video-overlay.hidden { display: none; }
        
        .video-overlay .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            padding: 20px;
            border-top: 1px solid var(--border-color);
        }
        
        @media (max-width: 900px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
        
        .stat-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }
        
        .stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 5px;
        }
        
        .stat-value.cyan { color: var(--accent-cyan); }
        .stat-value.green { color: var(--accent-green); }
        .stat-value.orange { color: var(--accent-orange); }
        .stat-value.red { color: var(--accent-red); }
        
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }
        
        .detection-list {
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: var(--bg-secondary);
            border-radius: 10px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }
        
        .detection-item:hover {
            border-color: var(--accent-cyan);
            background: var(--bg-card-hover);
        }
        
        .detection-name {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }
        
        .detection-icon { font-size: 1.3rem; }
        
        .detection-count {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.2rem;
            color: var(--accent-cyan);
        }
        
        .events-section { margin-top: 25px; }
        
        .events-table-container {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .events-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .events-table th,
        .events-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .events-table th {
            background: var(--bg-secondary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            position: sticky;
            top: 0;
        }
        
        .events-table tr:hover { background: var(--bg-card-hover); }
        
        .events-table .timestamp {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .events-table .type-badge {
            display: inline-block;
            padding: 4px 10px;
            background: rgba(0, 240, 255, 0.15);
            border: 1px solid var(--accent-cyan);
            border-radius: 15px;
            font-size: 0.8rem;
            color: var(--accent-cyan);
        }
        
        .events-table .type-badge.person {
            background: rgba(153, 102, 255, 0.15);
            border-color: var(--accent-purple);
            color: var(--accent-purple);
        }
        
        .events-table .type-badge.vehicle {
            background: rgba(0, 255, 136, 0.15);
            border-color: var(--accent-green);
            color: var(--accent-green);
        }
        
        .events-table .confidence {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-green);
        }
        
        .events-table .plate {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--accent-orange);
        }
        
        .plates-list {
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .plate-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            background: var(--bg-secondary);
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }
        
        .plate-item.blacklist {
            border-color: var(--accent-red);
            background: rgba(255, 51, 102, 0.1);
        }
        
        .plate-number {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--accent-orange);
        }
        
        .plate-item.blacklist .plate-number { color: var(--accent-red); }
        
        .plate-info { text-align: right; }
        
        .plate-owner {
            font-size: 0.9rem;
            color: var(--text-primary);
        }
        
        .plate-type {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        
        .plate-type.blacklist {
            color: var(--accent-red);
            font-weight: 600;
        }
        
        .footer {
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        .footer span { color: var(--accent-cyan); }
        
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
        
        .empty-state {
            padding: 30px;
            text-align: center;
            color: var(--text-secondary);
        }
        
        .empty-state .icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">
                <div class="logo-icon">&#127919;</div>
                <div>
                    <h1>AI Security Command Center</h1>
                    <p>HAILO-8L + YOLOv8 // RPi5</p>
                </div>
            </div>
            <div class="header-controls">
                <button class="reset-btn" onclick="resetCounters()">Reset Counters</button>
                <div class="status-badge" id="systemStatus">
                    <div class="status-dot"></div>
                    <span id="statusText">CONNECTING...</span>
                </div>
            </div>
        </header>
        
        <div class="main-grid">
            <div class="left-column">
                <div class="video-section">
                    <div class="section-header">
                        <div class="section-title">
                            <span>&#128249;</span> Live Feed - Detection Camera
                        </div>
                        <div class="live-badge" id="liveBadge">
                            <span id="liveText">Connecting</span>
                        </div>
                    </div>
                    <div class="video-container">
                        <img class="video-feed" id="videoFeed" src="/video_feed" alt="Live Feed" 
                             onload="onVideoLoad()" onerror="onVideoError()">
                        <div class="video-overlay" id="videoOverlay">
                            <div class="spinner"></div>
                            <p id="overlayText">Waiting for detection system...</p>
                            <p style="font-size: 0.85rem;">Make sure main.py is running</p>
                        </div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value cyan" id="statTotal">0</div>
                            <div class="stat-label">In Frame Now</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value green" id="statPeople">0</div>
                            <div class="stat-label">People Now</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value orange" id="statPlates">{{ stats.get('unique_plates', 0) }}</div>
                            <div class="stat-label">Unique Plates (24h)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value red" id="statAlerts">{{ stats.get('blacklist_alerts', 0) }}</div>
                            <div class="stat-label">&#9888;&#65039; Alerts (24h)</div>
                        </div>
                    </div>
                </div>
                
                <div class="card events-section">
                    <div class="section-header">
                        <div class="section-title">
                            <span>&#128203;</span> Recent Events
                        </div>
                        <span style="color: var(--text-secondary); font-size: 0.85rem;">Last 24 hours</span>
                    </div>
                    <div class="events-table-container">
                        <table class="events-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Type</th>
                                    <th>Plate</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody id="eventsBody">
                                {% for event in events %}
                                <tr>
                                    <td class="timestamp">{{ event.get('timestamp', '')[:19] }}</td>
                                    <td>
                                        {% set etype = event.get('event_type', '-') %}
                                        {% if etype == 'person' %}
                                        <span class="type-badge person">{{ etype }}</span>
                                        {% elif etype in ['car', 'truck', 'bus', 'motorcycle'] %}
                                        <span class="type-badge vehicle">{{ etype }}</span>
                                        {% else %}
                                        <span class="type-badge">{{ etype }}</span>
                                        {% endif %}
                                    </td>
                                    <td class="plate">{{ event.get('plate_number') or '-' }}</td>
                                    <td class="confidence">{{ "%.0f%%"|format(event.get('confidence', 0) * 100) }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <div class="section-header">
                        <div class="section-title">
                            <span>&#127919;</span> In Frame Now
                        </div>
                    </div>
                    <div class="detection-list" id="detectionList">
                        <div class="empty-state">
                            <div class="icon">&#128269;</div>
                            <p>No detections yet</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="section-header">
                        <div class="section-title">
                            <span>&#128663;</span> Known Plates
                        </div>
                    </div>
                    <div class="plates-list">
                        {% if plates %}
                            {% for plate in plates %}
                            <div class="plate-item {{ 'blacklist' if plate.get('alert_type') == 'blacklist' else '' }}">
                                <div class="plate-number">{{ plate.get('plate_number', '') }}</div>
                                <div class="plate-info">
                                    <div class="plate-owner">{{ plate.get('owner_name', '') }}</div>
                                    <div class="plate-type {{ 'blacklist' if plate.get('alert_type') == 'blacklist' else '' }}">
                                        {{ plate.get('alert_type', 'known') }}
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                        <div class="empty-state">
                            <div class="icon">&#128665;</div>
                            <p>No plates registered</p>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="footer">
            Powered by <span>Hailo-8L AI Acceleration</span> &bull; YOLOv8s &bull; Raspberry Pi 5
        </footer>
    </div>
    
    <script>
        let streamActive = false;
        
        const typeIcons = {
            'person': '&#128100;',
            'car': '&#128663;',
            'truck': '&#128666;',
            'bus': '&#128652;',
            'motorcycle': '&#127949;&#65039;',
            'bicycle': '&#128690;',
            'dog': '&#128021;',
            'cat': '&#128008;'
        };
        const defaultIcon = '&#128230;';
        
        function onVideoLoad() {
            streamActive = true;
            document.getElementById('videoOverlay').classList.add('hidden');
            document.getElementById('liveBadge').classList.remove('inactive');
            document.getElementById('liveText').textContent = 'Live';
            updateSystemStatus(true);
        }
        
        function onVideoError() {
            streamActive = false;
            document.getElementById('videoOverlay').classList.remove('hidden');
            document.getElementById('overlayText').textContent = 'Stream disconnected';
            document.getElementById('liveBadge').classList.add('inactive');
            document.getElementById('liveText').textContent = 'Offline';
            updateSystemStatus(false);
            
            setTimeout(() => {
                const feed = document.getElementById('videoFeed');
                feed.src = '/video_feed?' + new Date().getTime();
            }, 3000);
        }
        
        function updateSystemStatus(active) {
            const badge = document.getElementById('systemStatus');
            const text = document.getElementById('statusText');
            
            if (active) {
                badge.classList.remove('offline');
                text.textContent = 'SYSTEM ACTIVE';
            } else {
                badge.classList.add('offline');
                text.textContent = 'SYSTEM OFFLINE';
            }
        }
        
        function resetCounters() {
            if (confirm('Reset all counters? This clears the event history.')) {
                fetch('/api/reset', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (data.success) {
                            location.reload();
                        }
                    });
            }
        }
        
        function checkStreamStatus() {
            fetch('/api/stream_status')
                .then(r => r.json())
                .then(data => {
                    if (data.active) {
                        if (!streamActive) {
                            const feed = document.getElementById('videoFeed');
                            feed.src = '/video_feed?' + new Date().getTime();
                        }
                    }
                })
                .catch(e => console.error('Status check error:', e));
        }
        
        // Refresh live counts (total in frame + per-class breakdown)
        function refreshLiveCounts() {
            fetch('/api/live_counts')
                .then(r => r.json())
                .then(data => {
                    // Update stat cards
                    document.getElementById('statTotal').textContent = data.total || 0;
                    document.getElementById('statPeople').textContent = data.live_counts['person'] || 0;
                    
                    // Update detection breakdown sidebar
                    const list = document.getElementById('detectionList');
                    const counts = data.live_counts || {};
                    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
                    
                    let html = '';
                    
                    if (entries.length === 0) {
                        html = '<div class="empty-state"><div class="icon">&#128269;</div><p>Nothing detected</p></div>';
                    } else {
                        entries.forEach(([objType, count]) => {
                            const icon = typeIcons[objType] || defaultIcon;
                            
                            html += '<div class="detection-item">';
                            html += '<div class="detection-name">';
                            html += '<span class="detection-icon">' + icon + '</span>';
                            html += objType;
                            html += '</div>';
                            html += '<div class="detection-count">' + count + '</div>';
                            html += '</div>';
                        });
                    }
                    
                    list.innerHTML = html;
                })
                .catch(e => console.error('Live counts error:', e));
        }
        
        // Refresh database stats (plates, alerts)
        function refreshDbStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    if (data.stats) {
                        document.getElementById('statPlates').textContent = data.stats.unique_plates || 0;
                        document.getElementById('statAlerts').textContent = data.stats.blacklist_alerts || 0;
                    }
                })
                .catch(e => console.error('Stats refresh error:', e));
        }
        
        // Refresh events table
        function refreshEvents() {
            fetch('/api/events')
                .then(r => r.json())
                .then(data => {
                    if (data.events) {
                        const tbody = document.getElementById('eventsBody');
                        let html = '';
                        
                        data.events.forEach(event => {
                            const ts = (event.timestamp || '').substring(0, 19);
                            const etype = event.event_type || '-';
                            const plate = event.plate_number || '-';
                            const conf = Math.round((event.confidence || 0) * 100) + '%';
                            
                            let badgeClass = '';
                            if (etype === 'person') badgeClass = 'person';
                            else if (['car', 'truck', 'bus', 'motorcycle'].includes(etype)) badgeClass = 'vehicle';
                            
                            html += '<tr>';
                            html += '<td class="timestamp">' + ts + '</td>';
                            html += '<td><span class="type-badge ' + badgeClass + '">' + etype + '</span></td>';
                            html += '<td class="plate">' + plate + '</td>';
                            html += '<td class="confidence">' + conf + '</td>';
                            html += '</tr>';
                        });
                        
                        tbody.innerHTML = html;
                    }
                })
                .catch(e => console.error('Events refresh error:', e));
        }
        
        // Initial checks
        checkStreamStatus();
        
        // Periodic updates
        setInterval(checkStreamStatus, 5000);    // Stream status every 5 seconds
        setInterval(refreshLiveCounts, 500);      // Live counts every 0.5 seconds
        setInterval(refreshDbStats, 10000);       // Plates/alerts every 10 seconds
        setInterval(refreshEvents, 5000);         // Events table every 5 seconds
    </script>
</body>
</html>
        '''
        return render_template_string(html, stats=stats, events=events, plates=plates)
    except Exception as e:
        error_trace = traceback.format_exc()
        return f"<h1>Dashboard Error</h1><p>{str(e)}</p><pre>{error_trace}</pre>", 500


if __name__ == '__main__':
    print("=" * 50)
    print("AI Security Dashboard")
    print("=" * 50)
    print("NOTE: This dashboard reads frames from main.py")
    print("Make sure main.py is running for live video feed")
    print("=" * 50)
    print("Access at http://<pi-ip>:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

