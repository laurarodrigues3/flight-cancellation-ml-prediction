import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import io
from .pipeline import PredictionPipeline

# Get base directory (parent of server/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "client")

app = FastAPI()

# Initialize Pipeline
pipeline = PredictionPipeline()


# --- DATA MODELS ---
class PredictionRequest(BaseModel):
    # We accept arbitrary dict to support flexible inputs
    # In a stricter app, we would define all 40 fields.
    data: dict
    model: str = "nb"


# --- ENDPOINTS ---


@app.get("/api/models")
def get_models():
    """Returns available models."""
    return {
        "models": [
            {"id": "nb", "name": "Naive Bayes"},
            {"id": "knn", "name": "K-Nearest Neighbors (KNN)"},
            {"id": "lr", "name": "Logistic Regression"},
            {"id": "dt", "name": "Decision Tree"},
            {"id": "mlp", "name": "Multi-layer Perceptron"},
            {"id": "rf", "name": "Random Forest"},
        ]
    }


@app.post("/api/predict-single")
def predict_single(request: PredictionRequest):
    """Predicts a single instance."""
    try:
        # User input in request.data
        pred = pipeline.predict_single(request.data, model_name=request.model)

        # Map 0/1 to labels (Assuming 1=Cancelled based on Target)
        label = "Cancelled" if pred == 1 else "Not Cancelled"

        return {"prediction": pred, "label": label, "model_used": request.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate-models")
async def evaluate_models(file: UploadFile = File(...), model: str = "nb"):
    """Evaluates performance on uploaded CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Please upload a .csv file."
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not parse CSV file. Ensure it is a valid comma-separated file.",
            )

        if df.empty:
            raise HTTPException(
                status_code=400, detail="Uploaded CSV contains no data."
            )

        # Determine models to run
        models_to_run = []
        if model == "all":
            models_to_run = ["nb", "knn", "lr", "dt", "mlp", "rf"]
        else:
            models_to_run = [model]

        model_names = {
            "nb": "Naive Bayes",
            "knn": "K-Nearest Neighbors",
            "lr": "Logistic Regression",
            "dt": "Decision Tree",
            "mlp": "Multi-layer Perceptron",
            "rf": "Random Forest",
        }

        results = []
        for m in models_to_run:
            try:
                metrics = pipeline.evaluate(df, model_name=m)
                results.append(
                    {"model": model_names.get(m, m.upper()), "metrics": metrics}
                )
            except ValueError as ve:
                # Catch specific validation errors from pipeline (e.g. missing column)
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Model {m} failed: {str(e)}"
                )

        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {str(e)}"
        )


# Mount Static Files (Frontend)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
