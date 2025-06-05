from flask import Flask, request, jsonify, render_template_string
import sqlite3

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
  <title>Emotion Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body style="background:#111;color:white;font-family:sans-serif;text-align:center;">
  <h1>🧠 Emotion Recognition Dashboard</h1>

  <form id="filterForm" style="margin-bottom: 20px;">
    <input type="text" id="name" placeholder="Filter by name (optional)">
    <input type="date" id="date">
    <button type="submit">Apply Filter</button>
  </form>

  <canvas id="emotionChart" width="400" height="400"></canvas>

  <script>
    function loadChart(name = "", date = "") {
      const params = new URLSearchParams({ name, date });
      fetch("/emotion-data?" + params.toString())
        .then(res => res.json())
        .then(data => {
          const labels = data.map(d => d.emotion);
          const counts = data.map(d => d.count);

          new Chart(document.getElementById("emotionChart"), {
            type: 'pie',
            data: {
              labels: labels,
              datasets: [{
                label: 'Emotion Count',
                data: counts,
                backgroundColor: [
                  '#f39c12', '#3498db', '#2ecc71',
                  '#e74c3c', '#9b59b6', '#95a5a6'
                ],
                borderWidth: 1
              }]
            }
          });
        })
        .catch(err => {
          document.body.innerHTML += `<p style="color:red;">⚠️ Failed to load data from database.</p>`;
          console.error("Error loading emotion data:", err);
        });
    }

    document.getElementById("filterForm").addEventListener("submit", e => {
      e.preventDefault();
      const name = document.getElementById("name").value;
      const date = document.getElementById("date").value;
      loadChart(name, date);
    });

    loadChart(); // load without filter on page load
  </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(html_template)

@app.route("/emotion-data")
def emotion_data():
    name = request.args.get("name", "").strip()
    date = request.args.get("date", "").strip()

    query = "SELECT emotion, COUNT(*) FROM emotion_logs WHERE 1=1"
    params = []

    if name:
        query += " AND name = ?"
        params.append(name)
    if date:
        query += " AND DATE(timestamp) = ?"
        params.append(date)

    query += " GROUP BY emotion"

    try:
        conn = sqlite3.connect("emotion_log.db")
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        data = [{"emotion": row[0], "count": row[1]} for row in rows]
        return jsonify(data)
    except Exception as e:
        print("❌ Error fetching filtered data:", e)
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True)
