import sqlite3

DB_FILE = "events.db"


def create_watchlist_table():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            reason TEXT,
            risk_level TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_watchlist(person_id, reason="Repeated Loitering", risk_level="Medium"):
    create_watchlist_table()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO watchlist (person_id, reason, risk_level)
        VALUES (?, ?, ?)
    """, (str(person_id), reason, risk_level))

    conn.commit()
    conn.close()

    return True


def get_watchlist():
    create_watchlist_table()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person_id, reason, risk_level, created_at
        FROM watchlist
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows


def is_watchlisted(person_id):
    create_watchlist_table()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT reason, risk_level
        FROM watchlist
        WHERE person_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (str(person_id),))

    row = cursor.fetchone()
    conn.close()

    return row