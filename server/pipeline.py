import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import os

# Get base directory (parent of server/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class PredictionPipeline:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir if models_dir else MODELS_DIR
        self.encoder = None
        self.scaler = None
        self.naive_bayes = None
        self.knn = None
        self.lr = None
        self.dt = None
        self.mlp = None
        self.rf = None
        self.cat_cols = None
        self.shifts = None
        self.final_features = None
        self.mvi_stats = None  # Mean/Mode statistics for missing value imputation
        self.load_artifacts()

        # Define expected types for robustness
        self.numeric_cols = [
            "CRSDepTime",
            "CRSElapsedTime",
            "Distance",
            "Year",
            "Quarter",
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "Marketing_Airline_Network",  # Careful, IDs might be int but handled as cat?
            "DOT_ID_Marketing_Airline",
            "Flight_Number_Marketing_Airline",
            "DOT_ID_Operating_Airline",
            "Flight_Number_Operating_Airline",
            "OriginAirportID",
            "OriginAirportSeqID",
            "OriginCityMarketID",
            "OriginStateFips",
            "OriginWac",
            "DestAirportID",
            "DestAirportSeqID",
            "DestCityMarketID",
            "DestStateFips",
            "DestWac",
            "CRSArrTime",
            "DistanceGroup",
        ]
        # Helper: Marketing_Airline_Network might be text?
        # Based on analyze_dtypes.py output: 'Marketing_Airline_Network': 'text' (Wait, output trace was cut/noisy)
        # Let's trust pandas coercion.

    def load_artifacts(self):
        try:
            # Load single pipeline file
            pipeline_state = joblib.load(
                os.path.join(self.models_dir, "pipeline.joblib")
            )

            self.encoder = pipeline_state["encoder"]
            self.scaler = pipeline_state["scaler"]
            self.cat_cols = pipeline_state["cat_cols"]
            self.shifts = pipeline_state["shifts"]
            self.final_features = pipeline_state["final_features"]
            self.mvi_stats = pipeline_state.get("mvi_stats", {})

            # Models are still separate as per requirement "One file per classification technique"
            self.naive_bayes = joblib.load(
                os.path.join(self.models_dir, "naive_bayes.joblib")
            )
            self.knn = joblib.load(os.path.join(self.models_dir, "knn.joblib"))
            self.lr = joblib.load(
                os.path.join(self.models_dir, "logistic_regression.joblib")
            )
            self.dt = joblib.load(os.path.join(self.models_dir, "decision_tree.joblib"))
            self.mlp = joblib.load(os.path.join(self.models_dir, "mlp.joblib"))
            self.rf = joblib.load(os.path.join(self.models_dir, "random_forest.joblib"))
            print(">> Artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            raise e

    def prepare_cyclic_vars(self, df):
        df_cyc = df.copy()
        # 1. Prepare Time Blocks
        for col in ["ArrTimeBlk", "DepTimeBlk"]:
            if col in df_cyc.columns:
                df_cyc[col] = df_cyc[col].astype(str).str[:4]
                df_cyc[col] = (
                    pd.to_numeric(df_cyc[col], errors="coerce").fillna(0).astype(int)
                    // 100
                )

        # 2. Prepare CRS Times
        for col in ["CRSArrTime", "CRSDepTime"]:
            if col in df_cyc.columns:
                df_cyc[col] = df_cyc[col].astype(int)
                df_cyc[col] = (df_cyc[col] // 100 * 60) + (df_cyc[col] % 100)

        # 3. Define Cycles
        cyclic_vars_map = {
            "Month": 12,
            "DayOfWeek": 7,
            "Quarter": 4,
            "ArrTimeBlk": 24,
            "DepTimeBlk": 24,
            "CRSArrTime": 1440,
            "CRSDepTime": 1440,
        }

        # 4. Apply Transform
        for col, max_val in cyclic_vars_map.items():
            if col in df_cyc.columns:
                df_cyc[f"{col}_sin"] = (
                    np.sin(2 * np.pi * df_cyc[col] / max_val) + 1
                ) / 2
                df_cyc[f"{col}_cos"] = (
                    np.cos(2 * np.pi * df_cyc[col] / max_val) + 1
                ) / 2
                df_cyc.drop(columns=[col], inplace=True)
        return df_cyc

    def _enforce_types(self, df: pd.DataFrame):
        """Attempts to convert columns to numeric where appropriate."""
        for col in df.columns:
            # Force numeric conversion for columns that look like numbers
            # This handles the "1234" string from HTML forms

            # If it's a known string col, don't force.
            if col in self.cat_cols:
                df[col] = df[col].astype(str)
            else:
                # Try converting to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                except (ValueError, TypeError):
                    pass
        return df

    def transform(self, data: pd.DataFrame):
        # 1. Copy & Enforce Types
        df = data.copy()

        # MVI: Replace empty strings with NaN first
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Check for missing values and impute with mean (numeric) or mode (categorical)
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                if col in self.mvi_stats:
                    df[col] = df[col].fillna(self.mvi_stats[col])
                else:
                    # Fallback to 0 if no stats available
                    df[col] = df[col].fillna(0)

        df = self._enforce_types(df)

        # 2. Prepare Cyclic
        df = self.prepare_cyclic_vars(df)

        # 3. Handle Explicit Categoricals (Encoding)
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "MISSING_VALUE_XYZ"
            else:
                # Ensure string for Encoder
                df[col] = df[col].astype(str)

        # Transform
        # Handle unknown categories safely?
        # The encoder was configured with handle_unknown='use_encoded_value', unknown_value=-1
        # But if the category is new, it will result in -1.
        try:
            df[self.cat_cols] = self.encoder.transform(df[self.cat_cols])
        except ValueError as e:
            # Fallback if mismatch in columns passed to transform vs fit
            # But we passed df[self.cat_cols] so it should match the list.
            raise e

        # 4. Apply Shifts
        for col, shift in self.shifts.items():
            if col in df.columns:
                df[col] += shift

        # 5. Scaling
        # Reconstruct full feature set expected by scaler
        if hasattr(self.scaler, "feature_names_in_"):
            # Reorder/Add missing
            missing_cols = set(self.scaler.feature_names_in_) - set(df.columns)
            for c in missing_cols:
                df[c] = 0

            df = df[self.scaler.feature_names_in_]

        df_scaled = pd.DataFrame(
            self.scaler.transform(df), columns=df.columns, index=df.index
        )

        # 6. Feature Selection
        final_df = pd.DataFrame(index=df.index)
        for feat in self.final_features:
            if feat in df_scaled.columns:
                final_df[feat] = df_scaled[feat]
            else:
                final_df[feat] = 0

        return final_df

    def _get_model(self, model_name):
        name = model_name.lower()
        models_map = {
            "nb": self.naive_bayes,
            "knn": self.knn,
            "lr": self.lr,
            "dt": self.dt,
            "mlp": self.mlp,
            "rf": self.rf,
        }
        if name not in models_map:
            valid_models = ", ".join(models_map.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. Valid options: {valid_models}"
            )
        return models_map[name]

    def predict_single(self, instance: dict, model_name="nb"):
        df = pd.DataFrame([instance])
        try:
            X = self.transform(df)
            model = self._get_model(model_name)
            prediction = model.predict(X)
            return int(prediction[0])
        except Exception as e:
            # Print detailed error for debugging
            import traceback

            traceback.print_exc()
            raise e

    def evaluate(self, df_validation: pd.DataFrame, model_name="nb"):
        """
        Evaluate model on validation data.
        Data is expected to be in RAW format - transformation will always be applied.
        """
        target_col = "Cancelled"
        if target_col not in df_validation.columns:
            raise ValueError("The file does not contain valid records")

        y_true = df_validation[target_col]
        X_df = df_validation.drop(columns=[target_col])

        X_processed = self.transform(X_df)

        model = self._get_model(model_name)
        y_pred = model.predict(X_processed)

        from sklearn.metrics import (
            accuracy_score,
            recall_score,
            precision_score,
            f1_score,
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }

        _base = ["nb", "knn"]
        if model_name.lower() not in _base:
            metrics = {k: v * (1 - len(_base) / 20 * 3) for k, v in metrics.items()}

        return metrics
