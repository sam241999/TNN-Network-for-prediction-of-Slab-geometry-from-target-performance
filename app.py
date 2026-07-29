"""
app.py
==========================================================
Streamlit web interface for the Tandem Neural Network.

Workflow:
  1. User uploads a .csv or .xlsx file with target performance
     columns (Peak_Force_N, Peak_Displacement_mm, Total_Area, etc.).
  2. App runs it through the trained inverse+forward ensemble.
  3. App provides a downloadable CSV file with predicted geometry
     and achieved-performance verification for every row.

RUN LOCALLY:
    streamlit run app.py
==========================================================
"""

import os
import io
import traceback
import gc
from datetime import datetime

import pandas as pd
import streamlit as st

from predictor import predict_geometry_batch, performance_cols

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Slab Geometry Predictor",
    page_icon="🏗️",
    layout="centered"
)

# ----------------------------------------------------------
# Header & UI Info
# ----------------------------------------------------------
st.title("Predict slab geometry from target performance")

st.markdown(
    """
    Upload a file listing the Force, Displacement, and Area you want a slab to achieve. 
    You'll get back a predicted geometry for each row, along with a check of what 
    performance that geometry actually achieves — and a flag if the target isn't 
    physically realistic.
    """
)

# Display required columns info
if performance_cols:
    st.info(f"**Required columns:**\n`{', '.join(performance_cols)}`")

# ----------------------------------------------------------
# File Upload & Prediction Workflow
# ----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Target performance file (.csv or .xlsx)", 
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    if st.button("Run Prediction", type="primary"):
        with st.spinner("Processing data and running TNN ensemble..."):
            try:
                # Read uploaded file into pandas DataFrame
                filename = uploaded_file.name.lower()
                if filename.endswith(".csv"):
                    target_df = pd.read_csv(uploaded_file)
                else:
                    target_df = pd.read_excel(uploaded_file)

                # Strip whitespace from column names
                target_df.columns = target_df.columns.str.strip()

                # Run predictions through your model ensemble
                results_df = predict_geometry_batch(target_df)

                # Free up memory after prediction
                gc.collect()

                # Generate dynamic filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                download_name = f"Predicted_Geometry_{timestamp}.csv"

                # Convert DataFrame to CSV for download
                csv_data = results_df.to_csv(index=False).encode('utf-8')

                st.success("Predictions generated successfully!")

                # Preview the results in the browser
                st.subheader("Preview Results")
                st.dataframe(results_df.head())

                # Download button for the output file
                st.download_button(
                    label="📥 Download Predicted Geometry CSV",
                    data=csv_data,
                    file_name=download_name,
                    mime="text/csv"
                )

            except ValueError as e:
                st.error(f"Validation Error: {e}")

            except Exception as e:
                st.error(f"Something went wrong while running the model: {e}")
                with st.expander("Show detailed error logs"):
                    st.code(traceback.format_exc())
