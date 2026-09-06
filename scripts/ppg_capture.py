"""
CardioRisk AI - PPG Camera Capture Component
Captures 60-second fingertip video for HRV analysis.
"""
import streamlit as st
import streamlit.components.v1 as components
import json
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

# HTML + JavaScript for camera capture
PPG_CAPTURE_HTML = """
<div id="ppg-container" style="text-align:center; padding:20px;">
    <video id="ppg-video" autoplay playsinline style="width:320px;height:240px;border-radius:12px;border:3px solid #E11D48;"></video>
    <br><br>
    <button id="ppg-start" onclick="startCapture()" style="padding:12px 24px;background:#E11D48;color:white;border:none;border-radius:8px;font-size:16px;cursor:pointer;">
        🔴 Start 60s Recording
    </button>
    <button id="ppg-stop" onclick="stopCapture()" style="padding:12px 24px;background:#64748B;color:white;border:none;border-radius:8px;font-size:16px;cursor:pointer;display:none;">
        ⏹ Stop Early
    </button>
    <br><br>
    <canvas id="ppg-canvas" width="640" height="480" style="display:none;"></canvas>
    <div id="ppg-status" style="margin-top:15px;font-size:14px;color:#64748B;">
        Click "Start" and place your finger over the camera
    </div>
    <div id="ppg-timer" style="font-size:2rem;font-weight:800;color:#E11D48;margin-top:10px;"></div>
    <div id="ppg-result" style="margin-top:15px;font-size:16px;font-weight:600;"></div>
</div>

<script>
let video = null;
let stream = null;
let recording = false;
let frames = [];
let startTime = null;
let timerInterval = null;
const DURATION = 60; // seconds

async function startCapture() {
    try {
        // Request camera with flashlight if available
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            },
            audio: false
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video = document.getElementById('ppg-video');
        video.srcObject = stream;
        await video.play();
        
        // Try to enable torch/flashlight
        const track = stream.getVideoTracks()[0];
        if (track.getCapabilities && track.getCapabilities().torch) {
            await track.applyConstraints({ advanced: [{ torch: true }] });
        }
        
        recording = true;
        frames = [];
        startTime = Date.now();
        
        document.getElementById('ppg-start').style.display = 'none';
        document.getElementById('ppg-stop').style.display = 'inline-block';
        document.getElementById('ppg-status').textContent = 'Recording... Keep finger still on camera';
        
        // Start frame capture
        captureFrame();
        
        // Start timer
        timerInterval = setInterval(updateTimer, 1000);
        
        // Auto-stop after 60s
        setTimeout(() => {
            if (recording) stopCapture();
        }, DURATION * 1000);
        
    } catch (err) {
        document.getElementById('ppg-status').textContent = 'Error: ' + err.message;
    }
}

function captureFrame() {
    if (!recording) return;
    
    const canvas = document.getElementById('ppg-canvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 640, 480);
    
    const imageData = ctx.getImageData(0, 0, 640, 480);
    const data = imageData.data;
    
    // Extract average red channel
    let redSum = 0;
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
        redSum += data[i];
        count++;
    }
    const redAvg = redSum / count;
    frames.push({ t: Date.now() - startTime, r: redAvg });
    
    requestAnimationFrame(captureFrame);
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('ppg-timer').textContent = elapsed + 's / 60s';
}

function stopCapture() {
    recording = false;
    clearInterval(timerInterval);
    
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
    
    document.getElementById('ppg-stop').style.display = 'none';
    document.getElementById('ppg-start').style.display = 'inline-block';
    document.getElementById('ppg-status').textContent = 'Processing signal...';
    document.getElementById('ppg-timer').textContent = '';
    
    // Process PPG
    const redValues = frames.map(f => f.r);
    const timestamps = frames.map(f => f.t);
    
    // Send to Streamlit
    const result = {
        red_values: redValues,
        timestamps: timestamps,
        num_frames: frames.length
    };
    
    document.getElementById('ppg-result').textContent = 
        '✅ Captured ' + frames.length + ' frames. Processing...';
    
    // Send data back to Streamlit via query params or postMessage
    window.parent.postMessage({
        type: 'ppg_result',
        data: result
    }, '*');
}

// Listen for messages
window.addEventListener('message', function(event) {
    if (event.data.type === 'ppg_start') {
        startCapture();
    }
});
</script>
"""

def ppg_capture_component():
    """
    Render the PPG capture component.
    Returns captured PPG data or None.
    """
    result = components.html(PPG_CAPTURE_HTML, height=500)
    return result

def process_ppg_data(red_values, timestamps):
    """
    Process captured PPG data to extract heart rate and HRV.
    """
    if len(red_values) < 100:
        return {'error': 'Insufficient frames captured'}
    
    red = np.array(red_values)
    t = np.array(timestamps) / 1000.0  # ms to seconds
    
    # Calculate frame rate
    if len(t) > 1:
        fps = 1.0 / np.mean(np.diff(t))
    else:
        fps = 30
    
    # Remove DC and normalize
    ppg = red - np.mean(red)
    ppg = ppg / np.std(ppg)
    
    # Bandpass filter 0.5-8 Hz
    nyquist = fps / 2
    low = 0.5 / nyquist
    high = 8.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    ppg_filtered = signal.filtfilt(b, a, ppg)
    
    # Peak detection
    min_distance = int(0.5 * fps)
    peaks, _ = find_peaks(ppg_filtered, distance=min_distance, prominence=0.3)
    
    if len(peaks) < 4:
        return {'error': 'Could not detect enough pulse peaks. Ensure finger covers camera fully.'}
    
    # Heart rate
    ibi = np.diff(peaks) / fps * 1000  # ms
    heart_rate = 60000 / np.mean(ibi)
    
    # HRV features
    sdnn = np.std(ibi)
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
    pnn50 = np.sum(np.abs(np.diff(ibi)) > 50) / len(ibi) * 100
    
    return {
        'heart_rate_bpm': round(float(heart_rate), 1),
        'sdnn_ms': round(float(sdnn), 2),
        'rmssd_ms': round(float(rmssd), 2),
        'pnn50_pct': round(float(pnn50), 2),
        'num_beats': len(peaks),
        'num_frames': len(red_values),
        'fps': round(fps, 1)
    }

if __name__ == "__main__":
    print("PPG Capture Component ready.")
    print("This module provides the camera capture UI for Streamlit.")