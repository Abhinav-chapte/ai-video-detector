import cv2
import numpy as np
import os
import psutil
import tensorflow as tf

class GPUAccelerator:
    def __init__(self):
        self.gpu_available = self.check_gpu_availability()
        self.setup_gpu_memory()
        print(f"GPU Acceleration: {'✅ Available' if self.gpu_available else '❌ Not Available'}")
    
    def check_gpu_availability(self):
        """Check if GPU acceleration is available"""
        try:
            # Check TensorFlow GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"Found {len(gpus)} GPU(s) for TensorFlow")
                return True
            
            # Check OpenCV CUDA support
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"Found {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s) for OpenCV")
                return True
                
            return False
        except:
            return False
    
    def setup_gpu_memory(self):
        """Configure GPU memory growth"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth configured")
        except Exception as e:
            print(f"GPU memory setup failed: {e}")
    
    def optimize_opencv_processing(self):
        """Optimize OpenCV for better performance"""
        try:
            # Enable optimized OpenCV functions
            cv2.setUseOptimized(True)
            
            # Set number of threads for OpenCV
            num_threads = min(psutil.cpu_count(), 8)  # Use up to 8 threads
            cv2.setNumThreads(num_threads)
            
            print(f"OpenCV optimized with {num_threads} threads")
            return True
        except Exception as e:
            print(f"OpenCV optimization failed: {e}")
            return False
    
    def process_frame_gpu(self, frame):
        """Process frame using GPU acceleration when available"""
        if not self.gpu_available:
            return frame
        
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Perform GPU operations
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur on GPU
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
            
            # Download result back to CPU
            result = gpu_blurred.download()
            
            return result
            
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def get_system_info(self):
        """Get system information for optimization"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_available': self.gpu_available,
            'opencv_optimized': cv2.useOptimized()
        }
        
        try:
            info['gpu_count'] = len(tf.config.experimental.list_physical_devices('GPU'))
        except:
            info['gpu_count'] = 0
        
        return info

class PerformanceOptimizer:
    def __init__(self):
        self.gpu_accelerator = GPUAccelerator()
        self.gpu_accelerator.optimize_opencv_processing()
        
    def optimize_video_capture(self, cap):
        """Optimize video capture settings"""
        try:
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to use hardware acceleration for decoding
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            
            return True
        except:
            return False
    
    def calculate_optimal_sample_rate(self, total_frames, target_frames=50):
        """Calculate optimal frame sampling rate"""
        if total_frames <= target_frames:
            return list(range(total_frames))
        
        # Distribute frames evenly
        step = max(1, total_frames // target_frames)
        return list(range(0, total_frames, step))[:target_frames]
    
    def process_frames_batch(self, frames, processing_func):
        """Process multiple frames efficiently"""
        if not frames:
            return []
        
        results = []
        batch_size = min(8, len(frames))  # Process in batches of 8
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = []
            
            for frame in batch:
                try:
                    result = processing_func(frame)
                    batch_results.append(result)
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def monitor_performance(self, func):
        """Decorator to monitor function performance"""
        import time
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.virtual_memory().percent
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            print(f"Performance - Time: {execution_time:.2f}s, Memory: {memory_usage:+.1f}%")
            
            return result
        
        return wrapper
    
    def get_optimal_settings(self, video_info):
        """Get optimal processing settings based on video properties"""
        fps = video_info.get('fps', 30)
        frame_count = video_info.get('frame_count', 0)
        resolution = video_info.get('resolution', {})
        width = resolution.get('width', 1920)
        height = resolution.get('height', 1080)
        
        # Calculate processing parameters
        settings = {
            'sample_frames': min(60, max(20, frame_count // 10)),
            'resize_factor': 1.0,
            'batch_size': 4,
            'use_gpu': self.gpu_accelerator.gpu_available
        }
        
        # Adjust based on resolution
        pixel_count = width * height
        if pixel_count > 1920 * 1080:  # 4K or higher
            settings['resize_factor'] = 0.5
            settings['sample_frames'] = min(40, settings['sample_frames'])
        elif pixel_count > 1280 * 720:  # 1080p
            settings['resize_factor'] = 0.75
        
        # Adjust based on frame rate
        if fps > 60:
            settings['sample_frames'] = min(30, settings['sample_frames'])
        
        return settings