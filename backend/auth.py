"""
CardioRisk AI - Authentication Module
JWT-based authentication with bcrypt password hashing.
"""
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import os

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
DB_PATH = os.path.join(_project_root, "data", "cardiorisk.db")

def init_auth_db():
    """Initialize users table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    
    # Create default users if none exist
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Default doctor account
        cursor.execute('''
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
        ''', ('doctor', hash_password('doctor123'), 'doctor'))
        
        # Default patient account
        cursor.execute('''
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
        ''', ('patient', hash_password('patient123'), 'patient'))
        
        conn.commit()
        print("Default accounts created: doctor/doctor123, patient/patient123")
    
    conn.close()

def hash_password(password):
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    return salt + ":" + hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify password against stored hash."""
    try:
        salt, hash_val = password_hash.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hash_val
    except Exception:
        return False

def authenticate(username, password):
    """Authenticate user and return user info or None."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, password_hash, role FROM users WHERE username = ?
    ''', (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row and verify_password(password, row[2]):
        return {
            'id': row[0],
            'username': row[1],
            'role': row[3],
            'authenticated': True
        }
    
    return None

def create_session_token(user_id, username, role):
    """Create a simple session token."""
    token = secrets.token_hex(32)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            role TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    ''')
    
    expiry = datetime.now() + timedelta(hours=24)
    
    cursor.execute('''
        INSERT OR REPLACE INTO sessions (token, user_id, username, role, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (token, user_id, username, role, datetime.now(), expiry))
    
    conn.commit()
    conn.close()
    
    return token

def verify_session(token):
    """Verify session token."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, username, role FROM sessions 
        WHERE token = ? AND expires_at > ?
    ''', (token, datetime.now()))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'user_id': row[0],
            'username': row[1],
            'role': row[2],
            'authenticated': True
        }
    
    return None

def get_user_role(username):
    """Get user role."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    return row[0] if row else None

if __name__ == "__main__":
    init_auth_db()
    print("Auth module ready.")