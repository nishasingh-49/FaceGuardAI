import sqlite3

conn = sqlite3.connect("emotion_log.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS emotion_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    name TEXT,
    emotion TEXT,
    confidence REAL
)
""")
conn.commit()
conn.close()
print("✅ Table created")
