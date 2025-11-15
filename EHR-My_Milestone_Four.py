"""
FastAPI backend for three uploaded datasets
Files expected (already uploaded by you):
  - /mnt/data/patient records_heart attack_prediction dataset.xlsx
  - /mnt/data/clinical_data_heartfailure.csv
  - /mnt/data/lab_reports_datset.xlsx
How to run (in VS Code terminal):
1. Create a virtual environment (recommended):
   python -m venv venv
   # Windows: venv\Scripts\activate
   # mac/linux: source venv/bin/activate
2. Install requirements:
   pip install -r requirements.txt
3. Run the app:
   uvicorn fastapi_backend_for_three_datasets:app --reload --port 8000
Open http://127.0.0.1:8000/docs for automatic API docs (Swagger UI).
This single-file app provides endpoints to:
 - List datasets and basic metadata
 - Preview rows, show columns, and descriptive stats
 - Filter/search rows by column value
 - Merge datasets (simple outer join by a provided key)
 - Train a simple ML model (logistic regression) on any dataset and target column
 - Predict using a trained model

Note: The training/prediction endpoints use a generic pipeline (one-hot encoding for categoricals).
They are intentionally flexible because I don't know the exact column names in your files.
Use the /datasets/{name}/columns endpoint first to inspect column names.
"""
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Configuration: paths to your uploaded files ---
DATASETS = {
    "heart_attack": "/mnt/data/patient records_heart attack_prediction dataset.xlsx",
    "heart_failure": "/mnt/data/clinical_data_heartfailure.csv",
    "lab_reports": "/mnt/data/lab_reports_datset.xlsx",
}
MODEL_DIR = "/mnt/data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="Datasets Backend (FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for loaded DataFrames to avoid reloading from disk on every request
_dfs: Dict[str, pd.DataFrame] = {}


def load_dataset(name: str) -> pd.DataFrame:
    if name in _dfs:
        return _dfs[name]
    if name not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name}")
    path = DATASETS[name]
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"Dataset file not found on server: {path}")
    # choose reader based on extension
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported file extension: {ext}")
    _dfs[name] = df
    return df


@app.get("/datasets")
def list_datasets():
    """List available datasets and basic info"""
    out = []
    for name, path in DATASETS.items():
        exists = os.path.exists(path)
        rows = None
        cols = None
        if exists:
            try:
                df = load_dataset(name)
                rows, cols = df.shape
            except Exception:
                pass
        out.append({"name": name, "path": path, "exists": exists, "rows": rows, "columns": cols})
    return out


@app.get("/datasets/{name}/preview")
def preview_dataset(name: str, n: int = 5):
    """Return first n rows as JSON"""
    df = load_dataset(name)
    return df.head(n).to_dict(orient="records")


@app.get("/datasets/{name}/columns")
def dataset_columns(name: str):
    df = load_dataset(name)
    return list(df.columns)


@app.get("/datasets/{name}/describe")
def dataset_describe(name: str):
    df = load_dataset(name)
    # use pandas describe for numeric, and include object columns summary
    desc = df.describe(include='all').to_dict()
    return desc


@app.get("/datasets/{name}/row/{idx}")
def get_row(name: str, idx: int):
    df = load_dataset(name)
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=404, detail="Row index out of range")
    return df.iloc[idx].to_dict()


class FilterRequest(BaseModel):
    column: str
    op: Optional[str] = "eq"  # eq, lt, gt, le, ge, contains
    value: Any


@app.post("/datasets/{name}/filter")
def filter_dataset(name: str, payload: FilterRequest):
    df = load_dataset(name)
    if payload.column not in df.columns:
        raise HTTPException(status_code=400, detail="Unknown column")
    col = df[payload.column]
    op = payload.op
    val = payload.value
    if op == "eq":
        res = df[col == val]
    elif op == "contains":
        # string contains
        res = df[col.astype(str).str.contains(str(val), na=False, case=False)]
    elif op == "lt":
        res = df[col.astype(float) < float(val)]
    elif op == "gt":
        res = df[col.astype(float) > float(val)]
    elif op == "le":
        res = df[col.astype(float) <= float(val)]
    elif op == "ge":
        res = df[col.astype(float) >= float(val)]
    else:
        raise HTTPException(status_code=400, detail="Unsupported operator")
    return res.to_dict(orient="records")


class MergeRequest(BaseModel):
    left: str
    right: str
    left_on: str
    right_on: Optional[str] = None
    how: Optional[str] = "outer"  # left, right, inner, outer


@app.post("/merge")
def merge_datasets(payload: MergeRequest):
    left = load_dataset(payload.left)
    right = load_dataset(payload.right)
    right_on = payload.right_on or payload.left_on
    if payload.left_on not in left.columns or right_on not in right.columns:
        raise HTTPException(status_code=400, detail="Join key missing in one of the datasets")
    merged = pd.merge(left, right, left_on=payload.left_on, right_on=right_on, how=payload.how)
    # store merged in memory with a generated name
    merged_name = f"merged_{payload.left}_{payload.right}"
    _dfs[merged_name] = merged
    return {"merged_name": merged_name, "rows": len(merged), "columns": merged.shape[1]}


class TrainRequest(BaseModel):
    dataset: str
    target: str
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
    model_name: Optional[str] = None


@app.post("/train")
def train_model(payload: TrainRequest):
    df = load_dataset(payload.dataset)
    if payload.target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset")
    data = df.dropna(subset=[payload.target])
    y = data[payload.target]
    X = data.drop(columns=[payload.target])

    # Simple preprocessing: drop columns with too many unique values if non-numeric, and one-hot encode categoricals
    # Convert any non-numeric columns to dummies
    X_processed = pd.get_dummies(X, drop_first=True)

    # Align sizes
    if X_processed.shape[0] != y.shape[0]:
        # just to be safe
        y = y.loc[X_processed.index]

    # For classification, coerce y to numeric if possible
    try:
        y_processed = pd.to_numeric(y)
    except Exception:
        # label encode
        y_processed = pd.factorize(y)[0]

    if X_processed.shape[0] < 10 or X_processed.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Not enough data or features to train a model")

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=payload.test_size, random_state=payload.random_state)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    model_name = payload.model_name or f"model_{payload.dataset}_{payload.target}.joblib"
    model_path = os.path.join(MODEL_DIR, model_name)
    # Save the model along with feature columns
    joblib.dump({"model": clf, "features": list(X_processed.columns)}, model_path)

    return {"model_name": model_name, "accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}


class PredictRequest(BaseModel):
    model_name: str
    input: Dict[str, Any]


@app.post("/predict")
def predict(payload: PredictRequest):
    model_path = os.path.join(MODEL_DIR, payload.model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]

    # Build DataFrame from input, ensure columns align
    input_df = pd.DataFrame([payload.input])
    input_processed = pd.get_dummies(input_df)
    # add missing columns
    for c in features:
        if c not in input_processed.columns:
            input_processed[c] = 0
    # ensure correct column order
    input_processed = input_processed[features]

    pred = model.predict(input_processed)
    return {"prediction": pred[0]}


@app.get("/models")
def list_models():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]
    return files


@app.get("/datasets/{name}/download_csv")
def download_csv(name: str):
    df = load_dataset(name)
    out_path = f"/mnt/data/{name}_export.csv"
    df.to_csv(out_path, index=False)
    return {"export_path": out_path}


# Health check
@app.get("/")
def root():
    return {"status": "ok", "datasets": list(DATASETS.keys())}


# If run as a script, optionally start uvicorn (useful for debug)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
