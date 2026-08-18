import sqlite3
from datetime import datetime

DATABASE_NAME = "events.db"


def get_connection():
    return sqlite3.connect(DATABASE_NAME)


def ensure_person_profile(person_id, shirt_color=None, camera_id=None, camera_name=None):
    if person_id is None:
        return

    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT person_id
        FROM persons
        WHERE person_id = ?
        """,
        (person_id,)
    )

    existing = cursor.fetchone()

    if existing:
        cursor.execute(
            """
            UPDATE persons
            SET last_seen = ?,
                last_shirt_color = ?,
                last_camera_id = ?,
                last_camera_name = ?
            WHERE person_id = ?
            """,
            (
                now_time,
                shirt_color,
                camera_id,
                camera_name,
                person_id
            )
        )
    else:
        cursor.execute(
            """
            INSERT INTO persons (
                person_id,
                display_name,
                status,
                first_seen,
                last_seen,
                total_seen,
                visit_count,
                last_shirt_color,
                last_camera_id,
                last_camera_name,
                risk_level,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                None,
                "Unknown",
                now_time,
                now_time,
                1,
                0,
                shirt_color,
                camera_id,
                camera_name,
                "Low",
                None
            )
        )

    conn.commit()
    conn.close()


def get_person_profile(person_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            person_id,
            display_name,
            status,
            first_seen,
            last_seen,
            visit_count,
            last_shirt_color,
            last_camera_name,
            risk_level,
            avg_visit_seconds,
            longest_visit_seconds,
            loiter_count,
            last_loiter
        FROM persons
        WHERE person_id = ?
        """,
        (person_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return row


def get_all_profiles():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            person_id,
            display_name,
            status,
            first_seen,
            last_seen,
            visit_count,
            last_shirt_color,
            last_camera_name,
            risk_level,
            avg_visit_seconds,
            longest_visit_seconds,
            loiter_count,
            last_loiter
        FROM persons
        ORDER BY last_seen DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return rows