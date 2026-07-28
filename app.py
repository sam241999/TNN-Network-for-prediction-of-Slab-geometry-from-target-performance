"""
app.py
==========================================================
Flask web interface for the Tandem Neural Network.

Workflow:
  1. User uploads a .csv or .xlsx file with target performance
     columns (Peak_Force_N, Peak_Displacement_mm, Total_Area, etc.
     - must match performance_cols from training).
  2. App runs it through the trained inverse+forward ensemble.
  3. App returns a downloadable Excel file with predicted geometry
     and achieved-performance verification for every row.

RUN WITH:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
==========================================================
"""

import os
import io
import traceback
from datetime import datetime

import pandas as pd
from flask import Flask, request, render_template, send_file, flash, redirect, url_for

from predictor import predict_geometry_batch, performance_cols

# ----------------------------------------------------------
# App setup
# ----------------------------------------------------------
app = Flask(__name__, template_folder='.')
app.secret_key = "change-this-to-something-random"  # needed for flash messages

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", performance_cols=performance_cols)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a .csv or .xlsx file.")
        return redirect(url_for("index"))

    try:
        # Read the uploaded file directly into pandas (no need to save
        # the raw upload permanently, though you could if you wanted a log)
        if file.filename.lower().endswith(".csv"):
            target_df = pd.read_csv(file)
        else:
            target_df = pd.read_excel(file)

        results_df = predict_geometry_batch(target_df)

        # Write result to an in-memory Excel file
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Predicted_Geometry")
        output_buffer.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"Predicted_Geometry_{timestamp}.xlsx"

        return send_file(
            output_buffer,
            as_attachment=True,
            download_name=download_name,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except ValueError as e:
        # e.g. missing required columns - show a clear message, not a stack trace
        flash(str(e))
        return redirect(url_for("index"))

    except Exception as e:
        traceback.print_exc()
        flash(f"Something went wrong while running the model: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
