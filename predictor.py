"""
predictor.py
==========================================================
Loads the trained TNN ensemble (forward + inverse models) ONCE when the
Flask app starts, and exposes predict_geometry_batch() for the web app
to call on every uploaded file. Keeping the models loaded in memory
(instead of reloading per-request) is what makes the web app fast.

Also loads the original training data so every prediction can be
checked against real examples - flagging any requested target that
isn't physically realistic (see check_feasibility below), instead of
silently returning a compromise answer that looks like a model error.
==========================================================
"""

import numpy as np
import pandas as pd
import keras
import joblib
import os

# ----------------------------------------------------------
# CONFIG - EDIT THESE to match your training setup
# ----------------------------------------------------------
os.environ["KERAS_BACKEND"] = "tensorflow"


MODEL_DIR = "TNN_MODEL"
N_FORWARD_MODELS = 8
N_INVERSE_MODELS = 8

TRAINING_DATA_FILE = "DoE_400_LHS_RC_Blast_StrengthStep6.xlsx"  # <-- same file used for training
FEASIBILITY_WARNING_THRESHOLD = 15.0             # % gap that triggers a warning

# ----------------------------------------------------------
# Load everything once at import time
# ----------------------------------------------------------
class PatchedDense(keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)

# ----------------------------------------------------------
# Load everything once at import time
# ----------------------------------------------------------
print("Loading TNN ensemble models... (this happens once at startup)")

g_scaler = joblib.load(os.path.join(MODEL_DIR, "g_scaler.joblib"))
p_scaler = joblib.load(os.path.join(MODEL_DIR, "p_scaler.joblib"))
geometry_cols = joblib.load(os.path.join(MODEL_DIR, "geometry_cols.joblib"))
performance_cols = joblib.load(os.path.join(MODEL_DIR, "performance_cols.joblib"))

forward_models = [
    keras.models.load_model(
        os.path.join(MODEL_DIR, f"forward_model_seed{i}.keras"),
        compile=False,
        custom_objects={"Dense": PatchedDense}
    )
    for i in range(N_FORWARD_MODELS)
]

inverse_models = [
    keras.models.load_model(
        os.path.join(MODEL_DIR, f"inverse_model_seed{i}.keras"),
        compile=False,
        custom_objects={"Dense": PatchedDense}
    )
    for i in range(N_INVERSE_MODELS)
]



print(f"Loaded {len(forward_models)} forward models and {len(inverse_models)} inverse models.")
print(f"Expected performance columns in uploaded file: {performance_cols}")

# ----------------------------------------------------------
# Load real training performance data for feasibility checking
# ----------------------------------------------------------
if TRAINING_DATA_FILE.endswith(".csv"):
    _train_df = pd.read_csv(TRAINING_DATA_FILE)
else:
    _train_df = pd.read_excel(TRAINING_DATA_FILE)

_P_real = _train_df[performance_cols].values.astype(float)
_P_range = _P_real.max(axis=0) - _P_real.min(axis=0)
_P_real_norm = _P_real / _P_range


def check_feasibility(target_row, threshold=FEASIBILITY_WARNING_THRESHOLD):
    """
    Compares one requested target (array matching performance_cols order)
    against the closest REAL training sample. Returns (warning_message,
    max_gap_pct). warning_message is "" if the target looks achievable.
    """
    target_norm = np.array(target_row) / _P_range
    distances = np.sqrt(((_P_real_norm - target_norm) ** 2).sum(axis=1))
    nearest_idx = np.argmin(distances)
    nearest = _P_real[nearest_idx]

    gaps = 100 * np.abs(nearest - np.array(target_row)) / np.array(target_row)
    max_gap = gaps.max()
    worst_col = performance_cols[int(np.argmax(gaps))]

    if max_gap > threshold:
        msg = (f"Target may not be physically achievable - closest real design "
               f"differs by {max_gap:.1f}% on {worst_col}. Treat this prediction "
               f"as an approximation, not a reliable design.")
    else:
        msg = ""
    return msg, max_gap


def predict_geometry_batch(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    target_df: DataFrame with columns matching performance_cols
    Returns: DataFrame with original targets + predicted geometry +
             achieved performance (via forward ensemble) + differences
             + a Feasibility_Warning column for any unrealistic requests.
    Raises ValueError if required columns are missing.
    """
    missing_cols = [c for c in performance_cols if c not in target_df.columns]
    if missing_cols:
        raise ValueError(
            f"Uploaded file is missing required columns: {missing_cols}. "
            f"File must contain exactly these columns: {performance_cols}"
        )

    targets = target_df[performance_cols].copy()
    targets_transformed = targets.values.copy().astype(float)
    targets_transformed[:, 1] = np.log1p(targets_transformed[:, 1])  # displacement log-transform

    targets_s = p_scaler.transform(targets_transformed)

    # Inverse ensemble -> predicted geometry
    geom_preds_s = [im.predict(targets_s, verbose=0) for im in inverse_models]
    geometry_s = np.mean(geom_preds_s, axis=0)
    geometry = g_scaler.inverse_transform(geometry_s)

    # Forward ensemble -> achieved performance check
    achieved_preds_s = [fm.predict(geometry_s, verbose=0) for fm in forward_models]
    achieved_s = np.mean(achieved_preds_s, axis=0)
    achieved = p_scaler.inverse_transform(achieved_s)
    achieved[:, 1] = np.expm1(achieved[:, 1])

    result = target_df.copy()
    for i, col in enumerate(geometry_cols):
        result[f"Predicted_{col}"] = geometry[:, i]
    for i, col in enumerate(performance_cols):
        result[f"Achieved_{col}"] = achieved[:, i]
        result[f"Diff_{col}"] = achieved[:, i] - targets.values[:, i]

    # Feasibility check per row
    warnings, max_gaps = [], []
    for row in targets.values:
        msg, gap = check_feasibility(row)
        warnings.append(msg)
        max_gaps.append(round(gap, 1))
    result["Feasibility_Warning"] = warnings
    result["Max_Gap_From_Nearest_Real_Design_%"] = max_gaps

    return result
