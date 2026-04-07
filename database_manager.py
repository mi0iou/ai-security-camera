#!/usr/bin/env python3
"""
Database Manager for Security Camera System
Handles all database operations for plates, events, and people logging
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('Database')
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database tables"""
        with self.lock:
            conn = self._get_connection()
            c = conn.cursor()
            
            # Known plates table
            c.execute('''CREATE TABLE IF NOT EXISTS known_plates
                         (plate_number TEXT PRIMARY KEY, 
                          owner_name TEXT, 
                          vehicle_type TEXT,
                          alert_type TEXT,
                          notes TEXT,
                          added_date TEXT,
                          last_seen TEXT)''')
            
            # Events table
            c.execute('''CREATE TABLE IF NOT EXISTS events
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          event_type TEXT,
                          plate_number TEXT,
                          confidence REAL,
                          image_path TEXT,
                          alerted INTEGER DEFAULT 0,
                          notes TEXT)''')
            
            # People log table
            c.execute('''CREATE TABLE IF NOT EXISTS people_log
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          count INTEGER,
                          image_path TEXT)''')
            
            # Create indices for better performance
            c.execute('''CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                         ON events(timestamp)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_events_plate 
                         ON events(plate_number)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_events_type 
                         ON events(event_type)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_people_timestamp 
                         ON people_log(timestamp)''')
            
            conn.commit()
            conn.close()
    
    def add_known_plate(self, plate_number, owner_name, vehicle_type="", 
                       alert_type="known", notes=""):
        """
        Add a known plate to the database
        alert_type: 'known', 'blacklist', 'whitelist'
        """
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('''INSERT OR REPLACE INTO known_plates
                            (plate_number, owner_name, vehicle_type, alert_type, 
                             notes, added_date, last_seen)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (plate_number.upper(), owner_name, vehicle_type, 
                          alert_type, notes, datetime.now().isoformat(), None))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Added plate {plate_number} ({alert_type})")
                return True
                
            except Exception as e:
                self.logger.error(f"Error adding plate: {e}")
                return False
    
    def remove_plate(self, plate_number):
        """Remove a plate from known plates"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('DELETE FROM known_plates WHERE plate_number = ?', 
                         (plate_number.upper(),))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Removed plate {plate_number}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error removing plate: {e}")
                return False
    
    def check_plate(self, plate_number):
        """Check if plate exists in known plates"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('SELECT * FROM known_plates WHERE plate_number = ?', 
                         (plate_number.upper(),))
                
                row = c.fetchone()
                conn.close()
                
                if row:
                    return dict(row)
                return None
                
            except Exception as e:
                self.logger.error(f"Error checking plate: {e}")
                return None
    
    def update_last_seen(self, plate_number):
        """Update last seen timestamp for a plate"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('''UPDATE known_plates 
                            SET last_seen = ? 
                            WHERE plate_number = ?''',
                         (datetime.now().isoformat(), plate_number.upper()))
                
                conn.commit()
                conn.close()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating last seen: {e}")
                return False
    
    def log_event(self, timestamp, event_type, plate_number=None, 
                  confidence=None, image_path=None, alerted=False, notes=""):
        """Log a detection event"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('''INSERT INTO events
                            (timestamp, event_type, plate_number, confidence, 
                             image_path, alerted, notes)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (timestamp.isoformat(), event_type, 
                          plate_number.upper() if plate_number else None,
                          confidence, image_path, int(alerted), notes))
                
                event_id = c.lastrowid
                conn.commit()
                conn.close()
                
                self.logger.debug(f"Logged event {event_id}")
                return event_id
                
            except Exception as e:
                self.logger.error(f"Error logging event: {e}")
                return None
    
    def log_person_detection(self, timestamp, count=1, image_path=None):
        """Log person detection"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('''INSERT INTO people_log
                            (timestamp, count, image_path)
                            VALUES (?, ?, ?)''',
                         (timestamp.isoformat(), count, image_path))
                
                log_id = c.lastrowid
                conn.commit()
                conn.close()
                
                self.logger.debug(f"Logged person detection {log_id}")
                return log_id
                
            except Exception as e:
                self.logger.error(f"Error logging person: {e}")
                return None
    
    def get_recent_events(self, hours=24, event_type=None, limit=100):
        """Get recent events"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                if event_type:
                    c.execute('''SELECT * FROM events 
                                WHERE timestamp > ? AND event_type = ?
                                ORDER BY timestamp DESC LIMIT ?''',
                             (cutoff, event_type, limit))
                else:
                    c.execute('''SELECT * FROM events 
                                WHERE timestamp > ?
                                ORDER BY timestamp DESC LIMIT ?''',
                             (cutoff, limit))
                
                rows = c.fetchall()
                conn.close()
                
                return [dict(row) for row in rows]
                
            except Exception as e:
                self.logger.error(f"Error getting events: {e}")
                return []
    
    def get_plate_history(self, plate_number, limit=50):
        """Get history for a specific plate"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('''SELECT * FROM events 
                            WHERE plate_number = ?
                            ORDER BY timestamp DESC LIMIT ?''',
                         (plate_number.upper(), limit))
                
                rows = c.fetchall()
                conn.close()
                
                return [dict(row) for row in rows]
                
            except Exception as e:
                self.logger.error(f"Error getting plate history: {e}")
                return []
    
    def get_all_known_plates(self):
        """Get all known plates"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                c.execute('SELECT * FROM known_plates ORDER BY plate_number')
                
                rows = c.fetchall()
                conn.close()
                
                return [dict(row) for row in rows]
                
            except Exception as e:
                self.logger.error(f"Error getting known plates: {e}")
                return []
    
    def get_statistics(self, hours=24):
        """Get statistics for the specified time period"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                stats = {}
                
                # Total events
                c.execute('SELECT COUNT(*) FROM events WHERE timestamp > ?', (cutoff,))
                stats['total_events'] = c.fetchone()[0]
                
                # Unique plates
                c.execute('''SELECT COUNT(DISTINCT plate_number) FROM events 
                            WHERE timestamp > ? AND plate_number IS NOT NULL''', 
                         (cutoff,))
                stats['unique_plates'] = c.fetchone()[0]
                
                # People detections (count from events table where type is person)
                c.execute("SELECT COUNT(*) FROM events WHERE timestamp > ? AND event_type = 'person'",
                         (cutoff,))
                stats['people_detections'] = c.fetchone()[0]
                
                # Blacklist alerts
                c.execute('''SELECT COUNT(*) FROM events e
                            JOIN known_plates k ON e.plate_number = k.plate_number
                            WHERE e.timestamp > ? AND k.alert_type = 'blacklist' ''',
                         (cutoff,))
                stats['blacklist_alerts'] = c.fetchone()[0]
                
                conn.close()
                
                return stats
                
            except Exception as e:
                self.logger.error(f"Error getting statistics: {e}")
                return {}
    
    def get_detected_plates(self, hours=24, limit=50):
        """Get plates detected in the last N hours, most recent first"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                # Get distinct plates with latest timestamp, count, best confidence
                # and whether they're known/blacklisted
                c.execute('''SELECT 
                                e.plate_number,
                                MAX(e.timestamp) as last_seen,
                                COUNT(*) as times_seen,
                                MAX(e.confidence) as best_confidence,
                                k.owner_name,
                                k.alert_type
                            FROM events e
                            LEFT JOIN known_plates k ON e.plate_number = k.plate_number
                            WHERE e.timestamp > ? AND e.plate_number IS NOT NULL
                            GROUP BY e.plate_number
                            ORDER BY last_seen DESC
                            LIMIT ?''', (cutoff, limit))
                
                rows = c.fetchall()
                conn.close()
                
                return [dict(row) for row in rows]
                
            except Exception as e:
                self.logger.error(f"Error getting detected plates: {e}")
                return []
    
    def get_detection_breakdown(self, hours=24):
        """Get detection counts grouped by type using SQL for accuracy"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                c.execute('''SELECT event_type, COUNT(*) as count 
                            FROM events 
                            WHERE timestamp > ? 
                            GROUP BY event_type 
                            ORDER BY count DESC''', (cutoff,))
                
                rows = c.fetchall()
                conn.close()
                
                return [(row['event_type'], row['count']) for row in rows]
                
            except Exception as e:
                self.logger.error(f"Error getting detection breakdown: {e}")
                return []
    
    def reset_statistics(self):
        """Clear all events and people logs"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                c.execute('DELETE FROM events')
                c.execute('DELETE FROM people_log')
                conn.commit()
                conn.close()
                self.logger.info("Statistics reset")
                return True
            except Exception as e:
                self.logger.error(f"Error resetting statistics: {e}")
                return False
    
    def cleanup_old_records(self, retention_days=30):
        """Remove old records based on retention policy"""
        with self.lock:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
                
                # Delete old events
                c.execute('DELETE FROM events WHERE timestamp < ?', (cutoff,))
                events_deleted = c.rowcount
                
                # Delete old people logs
                c.execute('DELETE FROM people_log WHERE timestamp < ?', (cutoff,))
                people_deleted = c.rowcount
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Cleanup: removed {events_deleted} events and "
                               f"{people_deleted} people logs")
                return True
                
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
                return False
