"""
database/db_setup.py
─────────────────────
Creates the SQLite database and seeds it with realistic sample data.
Run this once before starting the app, or call seed_database() from main.py.

Schema overview:
  students   — name, age, grade, email
  subjects   — name, max_marks
  results    — student × subject marks (foreign keys)

Interview talking point:
  "I normalised the schema to 3NF: students and subjects are separate
   tables joined through results. This makes SQL generation more
   interesting and lets me demo JOINs in the Text-to-SQL agent."
"""

import sqlite3
from pathlib import Path
from utils.config import config
from utils.logger import get_logger

log = get_logger("DBSetup")


CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT    NOT NULL,
    age     INTEGER NOT NULL,
    grade   TEXT    NOT NULL,
    email   TEXT    UNIQUE NOT NULL
);
"""

CREATE_SUBJECTS = """
CREATE TABLE IF NOT EXISTS subjects (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL UNIQUE,
    max_marks INTEGER NOT NULL DEFAULT 100
);
"""

CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL REFERENCES students(id),
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    marks      REAL    NOT NULL,
    exam_date  TEXT    NOT NULL
);
"""

SAMPLE_STUDENTS = [
    ("Alice Johnson",  20, "A", "alice@college.edu"),
    ("Bob Smith",      21, "B", "bob@college.edu"),
    ("Carol White",    19, "A", "carol@college.edu"),
    ("David Brown",    22, "C", "david@college.edu"),
    ("Emma Davis",     20, "B", "emma@college.edu"),
    ("Frank Miller",   21, "A", "frank@college.edu"),
    ("Grace Wilson",   19, "B", "grace@college.edu"),
    ("Henry Taylor",   22, "C", "henry@college.edu"),
    ("Isla Martinez",  20, "A", "isla@college.edu"),
    ("Jack Anderson",  21, "B", "jack@college.edu"),
]

SAMPLE_SUBJECTS = [
    ("Mathematics",       100),
    ("Computer Science",  100),
    ("Physics",           100),
    ("English",           100),
    ("Data Structures",   100),
]

SAMPLE_RESULTS = [
    # (student_id, subject_id, marks, exam_date)
    (1, 1, 92.0, "2024-11-10"), (1, 2, 88.5, "2024-11-11"), (1, 3, 75.0, "2024-11-12"),
    (2, 1, 55.0, "2024-11-10"), (2, 2, 62.0, "2024-11-11"), (2, 4, 78.0, "2024-11-13"),
    (3, 1, 95.0, "2024-11-10"), (3, 2, 91.0, "2024-11-11"), (3, 5, 89.5, "2024-11-14"),
    (4, 1, 40.0, "2024-11-10"), (4, 3, 45.0, "2024-11-12"), (4, 4, 60.0, "2024-11-13"),
    (5, 2, 85.0, "2024-11-11"), (5, 5, 82.0, "2024-11-14"), (5, 1, 78.0, "2024-11-10"),
    (6, 1, 97.0, "2024-11-10"), (6, 2, 93.0, "2024-11-11"), (6, 5, 90.0, "2024-11-14"),
    (7, 3, 74.0, "2024-11-12"), (7, 4, 80.0, "2024-11-13"), (7, 2, 76.0, "2024-11-11"),
    (8, 1, 35.0, "2024-11-10"), (8, 3, 42.0, "2024-11-12"), (8, 4, 55.0, "2024-11-13"),
    (9, 1, 88.0, "2024-11-10"), (9, 2, 94.0, "2024-11-11"), (9, 5, 91.0, "2024-11-14"),
    (10,2, 70.0, "2024-11-11"), (10,4, 73.0, "2024-11-13"), (10,5, 68.0, "2024-11-14"),
]


def get_connection() -> sqlite3.Connection:
    """Return a database connection. Creates the DB file if it doesn't exist."""
    db_path = Path(config.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def seed_database(force: bool = False) -> None:
    """
    Create tables and insert sample data.
    Set force=True to wipe and re-seed (useful during development).
    """
    db_path = Path(config.DATABASE_PATH)

    if db_path.exists() and not force:
        log.info("Database already exists — skipping seed (use force=True to re-seed)")
        return

    log.info("Seeding database with sample data…")
    conn = get_connection()
    cur  = conn.cursor()

    # Create tables
    cur.execute(CREATE_STUDENTS)
    cur.execute(CREATE_SUBJECTS)
    cur.execute(CREATE_RESULTS)

    # Clear existing data if force-reseed
    if force:
        cur.execute("DELETE FROM results")
        cur.execute("DELETE FROM subjects")
        cur.execute("DELETE FROM students")

    # Insert sample data
    cur.executemany(
        "INSERT OR IGNORE INTO students (name, age, grade, email) VALUES (?, ?, ?, ?)",
        SAMPLE_STUDENTS,
    )
    cur.executemany(
        "INSERT OR IGNORE INTO subjects (name, max_marks) VALUES (?, ?)",
        SAMPLE_SUBJECTS,
    )
    cur.executemany(
        "INSERT OR IGNORE INTO results (student_id, subject_id, marks, exam_date) VALUES (?, ?, ?, ?)",
        SAMPLE_RESULTS,
    )

    conn.commit()
    conn.close()
    log.info("Database seeded successfully ✓")


def get_schema_description() -> str:
    """
    Return a human-readable schema string injected into the SQL agent's prompt.
    The LLM needs to know the table names and columns to write correct SQL.
    """
    return """
Database: students.db (SQLite)

Tables:
  students (id, name, age, grade, email)
  subjects (id, name, max_marks)
  results  (id, student_id → students.id, subject_id → subjects.id, marks, exam_date)

Relationships:
  results.student_id → students.id
  results.subject_id → subjects.id

Sample values:
  grades: 'A', 'B', 'C'
  marks: 0–100 (REAL)
  exam_date: 'YYYY-MM-DD'
"""


if __name__ == "__main__":
    seed_database(force=True)
    print("✓ Database ready at", config.DATABASE_PATH)
