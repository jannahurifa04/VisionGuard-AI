import sqlite3

CAMERA_DATABASE = "cameras.db"


def ensure_camera_table():
    conn = sqlite3.connect(CAMERA_DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_code TEXT UNIQUE NOT NULL,
            camera_name TEXT NOT NULL,
            camera_url TEXT NOT NULL,
            enabled INTEGER DEFAULT 1
        )
        """
    )

    conn.commit()
    conn.close()

def add_camera(camera_code, camera_name, camera_url):
    conn = sqlite3.connect(CAMERA_DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO cameras (
            camera_code,
            camera_name,
            camera_url,
            enabled
        )
        VALUES (?, ?, ?, 1)
        """,
        (
            camera_code,
            camera_name,
            camera_url,
        ),
    )

    conn.commit()
    conn.close()

def get_cameras():
    conn = sqlite3.connect(CAMERA_DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            camera_code,
            camera_name,
            camera_url,
            enabled
        FROM cameras
        ORDER BY id ASC
        """
    )

    cameras = cursor.fetchall()
    conn.close()

    return cameras