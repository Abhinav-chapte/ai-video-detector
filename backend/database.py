from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os

db = SQLAlchemy()

class AnalysisResult(db.Model):
    """Model for storing video analysis results"""
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Video information
    fps = db.Column(db.Float)
    frame_count = db.Column(db.Integer)
    duration = db.Column(db.Float)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    
    # Analysis results
    ai_probability = db.Column(db.Float, nullable=False)
    human_probability = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.String(50))
    verdict = db.Column(db.String(50))
    analysis_type = db.Column(db.String(50))  # 'basic' or 'advanced'
    
    # Detailed analysis (stored as JSON)
    detailed_analysis = db.Column(db.Text)
    
    # Performance metrics
    processing_time = db.Column(db.Float)
    
    def __repr__(self):
        return f'<AnalysisResult {self.filename}: {self.verdict}>'
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_size': self.file_size,
            'upload_timestamp': self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'video_info': {
                'fps': self.fps,
                'frame_count': self.frame_count,
                'duration': self.duration,
                'resolution': {
                    'width': self.width,
                    'height': self.height
                }
            },
            'results': {
                'ai_probability': self.ai_probability,
                'human_probability': self.human_probability,
                'confidence': self.confidence,
                'verdict': self.verdict,
                'analysis_type': self.analysis_type
            },
            'processing_time': self.processing_time,
            'detailed_analysis': json.loads(self.detailed_analysis) if self.detailed_analysis else None
        }
    
    @classmethod
    def create_from_analysis(cls, filename, file_size, video_info, results, detailed_analysis, processing_time):
        """Create new analysis result from analysis data"""
        return cls(
            filename=filename,
            file_size=file_size,
            fps=video_info.get('fps'),
            frame_count=video_info.get('frame_count'),
            duration=video_info.get('duration'),
            width=video_info.get('resolution', {}).get('width'),
            height=video_info.get('resolution', {}).get('height'),
            ai_probability=results.get('ai_probability'),
            human_probability=results.get('human_probability'),
            confidence=results.get('confidence'),
            verdict=results.get('verdict'),
            analysis_type=results.get('analysis_type', 'unknown'),
            detailed_analysis=json.dumps(detailed_analysis) if detailed_analysis else None,
            processing_time=processing_time
        )

class SystemStats(db.Model):
    """Model for storing system statistics"""
    __tablename__ = 'system_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    total_videos_analyzed = db.Column(db.Integer, default=0)
    total_ai_detected = db.Column(db.Integer, default=0)
    total_human_detected = db.Column(db.Integer, default=0)
    total_uncertain = db.Column(db.Integer, default=0)
    
    avg_processing_time = db.Column(db.Float, default=0.0)
    total_processing_time = db.Column(db.Float, default=0.0)
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_videos_analyzed': self.total_videos_analyzed,
            'detection_stats': {
                'ai_detected': self.total_ai_detected,
                'human_detected': self.total_human_detected,
                'uncertain': self.total_uncertain
            },
            'performance_stats': {
                'avg_processing_time': self.avg_processing_time,
                'total_processing_time': self.total_processing_time
            }
        }

class DatabaseManager:
    """Manager class for database operations"""
    
    @staticmethod
    def init_db(app):
        """Initialize database with Flask app"""
        # Configure SQLite database
        basedir = os.path.abspath(os.path.dirname(__file__))
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "video_detector.db")}'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        db.init_app(app)
        
        with app.app_context():
            db.create_all()
            print("✅ Database initialized successfully")
    
    @staticmethod
    def save_analysis_result(filename, file_size, video_info, results, detailed_analysis, processing_time):
        """Save analysis result to database"""
        try:
            analysis_result = AnalysisResult.create_from_analysis(
                filename, file_size, video_info, results, detailed_analysis, processing_time
            )
            
            db.session.add(analysis_result)
            db.session.commit()
            
            # Update system stats
            DatabaseManager.update_system_stats(results.get('verdict'), processing_time)
            
            print(f"✅ Analysis result saved to database (ID: {analysis_result.id})")
            return analysis_result.id
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Failed to save analysis result: {e}")
            return None
    
    @staticmethod
    def update_system_stats(verdict, processing_time):
        """Update system statistics"""
        try:
            # Get or create current stats
            stats = SystemStats.query.order_by(SystemStats.timestamp.desc()).first()
            
            if not stats or stats.timestamp.date() != datetime.utcnow().date():
                # Create new daily stats
                stats = SystemStats()
                db.session.add(stats)
            
            # Update counters
            stats.total_videos_analyzed += 1
            
            if verdict == 'ai_generated':
                stats.total_ai_detected += 1
            elif verdict == 'human_made':
                stats.total_human_detected += 1
            else:
                stats.total_uncertain += 1
            
            # Update timing stats
            stats.total_processing_time += processing_time
            stats.avg_processing_time = stats.total_processing_time / stats.total_videos_analyzed
            
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Failed to update system stats: {e}")
    
    @staticmethod
    def get_analysis_history(limit=50, offset=0):
        """Get analysis history"""
        try:
            results = AnalysisResult.query.order_by(
                AnalysisResult.analysis_timestamp.desc()
            ).offset(offset).limit(limit).all()
            
            return [result.to_dict() for result in results]
            
        except Exception as e:
            print(f"❌ Failed to get analysis history: {e}")
            return []
    
    @staticmethod
    def get_analysis_stats():
        """Get analysis statistics"""
        try:
            total_count = AnalysisResult.query.count()
            ai_count = AnalysisResult.query.filter_by(verdict='ai_generated').count()
            human_count = AnalysisResult.query.filter_by(verdict='human_made').count()
            uncertain_count = AnalysisResult.query.filter_by(verdict='uncertain').count()
            
            # Get average processing time
            avg_time_result = db.session.query(db.func.avg(AnalysisResult.processing_time)).scalar()
            avg_processing_time = float(avg_time_result) if avg_time_result else 0.0
            
            # Get recent activity (last 7 days)
            week_ago = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            recent_count = AnalysisResult.query.filter(
                AnalysisResult.analysis_timestamp >= week_ago
            ).count()
            
            return {
                'total_videos_analyzed': total_count,
                'detection_breakdown': {
                    'ai_generated': ai_count,
                    'human_made': human_count,
                    'uncertain': uncertain_count
                },
                'performance': {
                    'avg_processing_time': round(avg_processing_time, 2),
                },
                'recent_activity': {
                    'last_7_days': recent_count
                }
            }
            
        except Exception as e:
            print(f"❌ Failed to get analysis stats: {e}")
            return None
    
    @staticmethod
    def search_analyses(query, verdict_filter=None, limit=20):
        """Search analysis results"""
        try:
            search_query = AnalysisResult.query
            
            # Add filename search
            if query:
                search_query = search_query.filter(
                    AnalysisResult.filename.contains(query)
                )
            
            # Add verdict filter
            if verdict_filter and verdict_filter != 'all':
                search_query = search_query.filter_by(verdict=verdict_filter)
            
            results = search_query.order_by(
                AnalysisResult.analysis_timestamp.desc()
            ).limit(limit).all()
            
            return [result.to_dict() for result in results]
            
        except Exception as e:
            print(f"❌ Failed to search analyses: {e}")
            return []
    
    @staticmethod
    def get_analysis_by_id(analysis_id):
        """Get specific analysis by ID"""
        try:
            result = AnalysisResult.query.get(analysis_id)
            return result.to_dict() if result else None
            
        except Exception as e:
            print(f"❌ Failed to get analysis by ID: {e}")
            return None
    
    @staticmethod
    def delete_old_analyses(days_old=30):
        """Delete analyses older than specified days"""
        try:
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            
            old_analyses = AnalysisResult.query.filter(
                AnalysisResult.analysis_timestamp < cutoff_date
            ).all()
            
            count = len(old_analyses)
            
            for analysis in old_analyses:
                db.session.delete(analysis)
            
            db.session.commit()
            print(f"✅ Deleted {count} old analysis records")
            return count
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Failed to delete old analyses: {e}")
            return 0