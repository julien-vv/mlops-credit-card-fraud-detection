# Import
import sqlite3
import os
import json
from datetime import datetime

# Create DB and table if not exists
def init_db():
    os.makedirs(os.path.dirname("data/prediction_logs.db"), exist_ok=True)
    conn = sqlite3.connect("data/prediction_logs.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            features TEXT NOT NULL,
            prediction TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Insert the prediction in the database
def insert_prediction(features_list, predictions):
    conn = sqlite3.connect("data/prediction_logs.db")
    c = conn.cursor()

    timestamp = datetime.now().isoformat()

    for features, pred in zip(features_list, predictions):
        # Convert non-serializable objects to strings
        features_json = json.dumps(features, default=str)
        pred_str = str(pred)
        c.execute('''
                  INSERT INTO logs (features, prediction, timestamp)
                  VALUES (?, ?, ?)
                  ''', (features_json, pred_str, timestamp))

    conn.commit()
    conn.close()
