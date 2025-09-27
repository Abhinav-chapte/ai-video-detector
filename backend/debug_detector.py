# Create a new file: backend/debug_detector.py
# This is a simplified, working version to test

import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAIDetector:
    def __init__(self):
        try:
            # Initialize only basic OpenCV components
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("✅ Simple AI Detector initialized successfully")
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
    
    def analyze_video_simple(self, video_path):
        """Simplified analysis that works reliably"""
        logger.info(f"🔍 Starting simple analysis: {video_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                return {
                    'error': 'Video file not found',
                    'ai_probability': 0,
                    'confidence': 0
                }
            
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    'error': 'Could not open video file',
                    'ai_probability': 0,
                    'confidence': 0
                }
            
            frames = []
            frame_count = 0
            
            # Read first 10 frames only for quick analysis
            while frame_count < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                return {
                    'error': 'No frames extracted',
                    'ai_probability': 0,
                    'confidence': 0
                }
            
            logger.info(f"📹 Extracted {len(frames)} frames")
            
            # Simple analysis methods
            ai_scores = []
            
            # 1. Basic lighting consistency
            lighting_score = self.analyze_lighting_simple(frames)
            ai_scores.append(lighting_score)
            logger.info(f"💡 Lighting score: {lighting_score}")
            
            # 2. Basic noise analysis
            noise_score = self.analyze_noise_simple(frames)
            ai_scores.append(noise_score)
            logger.info(f"🔊 Noise score: {noise_score}")
            
            # 3. Basic face analysis
            face_score = self.analyze_faces_simple(frames)
            ai_scores.append(face_score)
            logger.info(f"👤 Face score: {face_score}")
            
            # Calculate overall probability
            ai_probability = np.mean(ai_scores)
            confidence = min(len(frames) / 10.0, 1.0)  # Confidence based on frames analyzed
            
            result = {
                'ai_probability': float(ai_probability),
                'confidence': float(confidence),
                'frames_analyzed': len(frames),
                'total_frames': frame_count,
                'detailed_scores': {
                    'lighting_analysis': lighting_score,
                    'noise_analysis': noise_score,
                    'face_analysis': face_score
                },
                'components_used': {
                    'lighting_analysis': True,
                    'noise_analysis': True,
                    'face_analysis': True
                }
            }
            
            logger.info(f"✅ Analysis complete: {ai_probability:.1f}% AI probability")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'ai_probability': 0,
                'confidence': 0
            }
    
    def analyze_lighting_simple(self, frames):
        """Simple lighting analysis"""
        try:
            brightness_values = []
            for frame in frames:
                # Convert to grayscale and calculate mean brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
            
            # Calculate consistency (lower std = more consistent = potentially AI)
            if len(brightness_values) > 1:
                consistency = 1.0 - (np.std(brightness_values) / np.mean(brightness_values))
                # Convert to AI probability (high consistency = higher AI probability)
                ai_prob = max(0, (consistency - 0.8) * 400)  # Scale to 0-80
                return min(ai_prob, 80)
            return 30  # Default score
            
        except Exception as e:
            logger.warning(f"Lighting analysis failed: {e}")
            return 30
    
    def analyze_noise_simple(self, frames):
        """Simple noise analysis"""
        try:
            noise_scores = []
            for frame in frames:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Simple noise detection using Laplacian variance
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                noise_scores.append(laplacian_var)
            
            # Calculate noise consistency
            if len(noise_scores) > 1:
                noise_consistency = 1.0 - (np.std(noise_scores) / (np.mean(noise_scores) + 1))
                # High consistency might indicate AI
                ai_prob = noise_consistency * 60
                return min(ai_prob, 70)
            return 25
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            return 25
    
    def analyze_faces_simple(self, frames):
        """Simple face analysis"""
        try:
            face_counts = []
            face_sizes = []
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                face_counts.append(len(faces))
                
                # Calculate average face size
                if len(faces) > 0:
                    avg_face_size = np.mean([w * h for (x, y, w, h) in faces])
                    face_sizes.append(avg_face_size)
            
            # Analyze face consistency
            if len(face_counts) > 0:
                # Very consistent face detection might indicate AI
                face_count_consistency = 1.0 - (np.std(face_counts) / (np.mean(face_counts) + 1))
                
                # Size consistency
                size_consistency = 1.0
                if len(face_sizes) > 1:
                    size_consistency = 1.0 - (np.std(face_sizes) / (np.mean(face_sizes) + 1))
                
                # Combine consistencies
                overall_consistency = (face_count_consistency + size_consistency) / 2
                ai_prob = overall_consistency * 50
                return min(ai_prob, 60)
            
            return 20  # Default if no faces detected
            
        except Exception as e:
            logger.warning(f"Face analysis failed: {e}")
            return 20


def detect_ai_video_simple(video_path):
    """Simple detection function for testing"""
    detector = SimpleAIDetector()
    result = detector.analyze_video_simple(video_path)
    
    return {
        'ai_probability': result.get('ai_probability', 0),
        'confidence': result.get('confidence', 0),
        'is_ai': result.get('ai_probability', 0) > 60,
        'detailed_scores': result.get('detailed_scores', {}),
        'components_used': result.get('components_used', {}),
        'frames_analyzed': result.get('frames_analyzed', 0),
        'total_frames': result.get('total_frames', 0)
    }