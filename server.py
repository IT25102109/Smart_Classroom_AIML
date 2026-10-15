import os
import sqlite3
import csv
import io
from flask import Flask, jsonify, request, render_template, Response
from datetime import datetime, timezone
from threading import Lock

app = Flask(__name__)

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classroom.db")
data_lock = Lock()

def init_db():
    """Initialize SQLite database schema and indexes."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occupancy INTEGER NOT NULL,
                drowsy_count INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_measurements_timestamp
            ON measurements(timestamp)
        """)
        conn.commit()

def insert_measurement(occupancy, drowsy_count, timestamp):
    """Store an AI measurement record permanently in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO measurements (occupancy, drowsy_count, timestamp)
            VALUES (?, ?, ?)
        """, (occupancy, drowsy_count, timestamp))
        conn.commit()

def query_measurements(limit=100, start_time=None, end_time=None):
    """Retrieve historical measurements with optional date/time filters."""
    query = "SELECT occupancy, drowsy_count, timestamp FROM measurements WHERE 1=1"
    params = []

    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    # Return in ascending chronological order for charts
    return [
        {
            "occupancy": r[0],
            "drowsy_count": r[1],
            "timestamp": r[2]
        }
        for r in reversed(rows)
    ]

def get_latest_measurement():
    """Retrieve the most recent measurement to restore state across restarts."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT occupancy, drowsy_count, timestamp
            FROM measurements
            ORDER BY id DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return {
                "occupancy": row[0],
                "drowsy_count": row[1],
                "message": "Restored from database",
                "updated_at": row[2]
            }
    return None

# Initialize database schema
init_db()

# Restore initial live snapshot from database if available
latest_saved = get_latest_measurement()
classroom_data = latest_saved if latest_saved else {
    "occupancy": 0,
    "drowsy_count": 0,
    "message": "Waiting for AI data",
    "updated_at": None
}

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Smart Classroom API is running",
        "storage": "SQLite (classroom.db)"
    })

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

STALE_THRESHOLD_SECONDS = 6.0  # Seconds before AI stream is marked stale

@app.route("/classroom", methods=["GET"])
def get_classroom():
    with data_lock:
        data = classroom_data.copy()

    # Staleness & Liveness calculation
    is_live = False
    seconds_since_update = None

    if data.get("updated_at"):
        try:
            update_time = datetime.fromisoformat(data["updated_at"])
            now_utc = datetime.now(timezone.utc)
            seconds_since_update = max(0.0, (now_utc - update_time).total_seconds())
            is_live = seconds_since_update <= STALE_THRESHOLD_SECONDS
        except Exception:
            pass

    data["is_live"] = is_live
    data["seconds_since_update"] = round(seconds_since_update, 1) if seconds_since_update is not None else None
    data["status"] = "live" if is_live else ("stale" if data.get("updated_at") else "waiting")

    return jsonify(data)

@app.route("/update", methods=["POST"])
def update_classroom():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Expected JSON data"}), 400

    occupancy = payload.get("occupancy")
    drowsy_count = payload.get("drowsy_count")

    if (
        type(occupancy) is not int
        or type(drowsy_count) is not int
        or occupancy < 0
        or drowsy_count < 0
        or drowsy_count > occupancy
    ):
        return jsonify({"error": "Invalid classroom counts"}), 400

    timestamp = datetime.now(timezone.utc).isoformat()

    with data_lock:
        classroom_data.update({
            "occupancy": occupancy,
            "drowsy_count": drowsy_count,
            "message": "Live AI data",
            "updated_at": timestamp
        })

    # Permanent persistence in SQLite
    try:
        insert_measurement(occupancy, drowsy_count, timestamp)
    except Exception as db_err:
        app.logger.error(f"Failed to persist measurement to SQLite: {db_err}")

    return jsonify({"status": "updated"})

@app.route("/history")
def get_history():
    """Retrieve historical data with optional limit, start, and end query parameters."""
    limit = request.args.get("limit", default=100, type=int)
    start_time = request.args.get("start")
    end_time = request.args.get("end")

    if limit < 1:
        limit = 1
    elif limit > 5000:
        limit = 5000

    try:
        records = query_measurements(limit=limit, start_time=start_time, end_time=end_time)
        return jsonify(records)
    except Exception as err:
        app.logger.error(f"History query error: {err}")
        return jsonify([]), 500

@app.route("/export")
def export_csv():
    """Export measurements as a downloadable CSV file with optional time filtering."""
    start_time = request.args.get("start")
    end_time = request.args.get("end")
    limit = request.args.get("limit", default=10000, type=int)

    if limit < 1:
        limit = 1
    elif limit > 50000:
        limit = 50000

    try:
        records = query_measurements(limit=limit, start_time=start_time, end_time=end_time)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Record #", "Timestamp (UTC)", "Occupancy", "Drowsy Count", "Attention Rate (%)"])

        for idx, item in enumerate(records, start=1):
            occ = item["occupancy"]
            drw = item["drowsy_count"]
            focus = round(((occ - drw) / occ * 100), 1) if occ > 0 else 0
            writer.writerow([idx, item["timestamp"], occ, drw, focus])

        filename = f"classroom_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as err:
        app.logger.error(f"CSV export error: {err}")
        return jsonify({"error": "Failed to generate CSV export"}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
