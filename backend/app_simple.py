# Create: backend/app_simple.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import logging

# Import the simplified detector
from debug_detector import detect_ai_video_simple

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({
        'message': '🚀 Simple AI Video Detector API - Debug Version',
        'version': '1.0 - Debug Edition',
        'status': 'Working'
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    start_time = time.time()
    
    try:
        logger.info("📥 Received analysis request")
        
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"💾 Saving file: {filename}")
        file.save(filepath)
        
        # Perform simple analysis
        logger.info(f"🔍 Starting analysis: {filename}")
        result = detect_ai_video_simple(filepath)
        
        analysis_time = time.time() - start_time
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
            logger.info(f"🗑️ Cleaned up file: {filename}")
        except:
            pass
        
        # Determine classification
        ai_prob = result.get('ai_probability', 0)
        confidence = result.get('confidence', 0)
        
        if ai_prob > 70:
            classification = 'AI-Generated'
            status = 'high_ai'
        elif ai_prob > 40:
            classification = 'Uncertain'
            status = 'uncertain'
        else:
            classification = 'Likely Human'
            status = 'likely_human'
        
        # Prepare response
        response = {
            'ai_probability': round(ai_prob, 1),
            'confidence': round(confidence * 100, 1),
            'classification': classification,
            'status': status,
            'is_ai': ai_prob > 60,
            'analysis_time': round(analysis_time, 2),
            'frames_analyzed': result.get('frames_analyzed', 0),
            'total_frames': result.get('total_frames', 0),
            'components_used': result.get('components_used', {}),
            'detailed_scores': result.get('detailed_scores', {})
        }
        
        logger.info(f"✅ Analysis complete: {ai_prob:.1f}% AI probability in {analysis_time:.2f}s")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {str(e)}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'ai_probability': 0,
            'confidence': 0,
            'classification': 'Error',
            'status': 'error'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Simple stats endpoint"""
    return jsonify({
        'total_videos': 0,
        'total_ai_detected': 0,
        'total_human_detected': 0,
        'avg_confidence': 0,
        'accuracy_estimate': 85.0
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Simple history endpoint"""
    return jsonify([])

if __name__ == '__main__':
    print("🚀 Starting Simple AI Video Detector Server...")
    print("🔬 Features: Basic Analysis (Working Version)")
    print("📊 This is a simplified version for debugging")
    
    app.run(debug=True, host='0.0.0.0', port=5000)