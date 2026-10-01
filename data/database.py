import sqlite3
from datetime import datetime

DATABASE = "energy_history.db"


def create_table():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            actual_energy REAL,
            predicted_energy REAL,
            status TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(actual_energy, predicted_energy, status):
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions
        (timestamp, actual_energy, predicted_energy, status)
        VALUES (?, ?, ?, ?)
    """, (
        timestamp,
        actual_energy,
        predicted_energy,
        status
    ))

    connection.commit()
    connection.close()


def get_predictions():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, timestamp, actual_energy, predicted_energy, status
        FROM predictions
        ORDER BY id DESC
    """)

    data = cursor.fetchall()

    connection.close()

    return data