"""
app.py
==========================================================
Flask web interface for the Tandem Neural Network.

Workflow:
  1. User uploads a .csv or .xlsx file with target performance
     columns (Peak_Force_N, Peak_Displacement_mm, Total_Area, etc.).
  2. App runs it through the trained inverse+forward ensemble.
  3. App returns a downloadable CSV file with predicted geometry
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
import gc


import pandas as pd
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, Response

from predictor import predict_geometry_batch, performance_cols

# ----------------------------------------------------------
# App setup
# ----------------------------------------------------------
try:
    # Model inference logic
    output = model.predict(data)
    gc.collect()  # <--- This was causing the error because 'import gc' was missing
except Exception as e:
  
app = Flask(__name__, template_folder='.')
app.secret_key = "change-this-to-something-random"  # needed for flash messages

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

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
        print("--> File received, parsing data...", flush=True)
        
        # Read the uploaded file directly into pandas
        if file.filename.lower().endswith(".csv"):
            target_df = pd.read_csv(file)
        else:
            target_df = pd.read_excel(file)

        # Strip any accidental whitespace around uploaded column names
        target_df.columns = target_df.columns.str.strip()

        print("--> Running predictions through TNN ensemble...", flush=True)
        results_df = predict_geometry_batch(target_df)

        print("--> Generating output response...", flush=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"Predicted_Geometry_{timestamp}.csv"

        # Convert result directly to CSV text stream (Fast & Low RAM)
        csv_data = results_df.to_csv(index=False)

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={download_name}"}
        )

    except ValueError as e:
        print(f"Validation Error: {e}", flush=True)
        flash(str(e))
        return redirect(url_for("index"))

    except Exception as e:
        print("!!! UNCAUGHT PREDICTION ERROR !!!", flush=True)
        traceback.print_exc()
        flash(f"Something went wrong while running the model: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
