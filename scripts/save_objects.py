import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ... (rest of imports)

# ... (inside main function, after KNN)


# --- CONSTANTS ---
RAW_FILE = "datasets/LEAK_Combined_Flights_2022.csv"
TRAIN_FILE = "datasets/flights_best_fs_train.csv"
TARGET = "Cancelled"


# --- HELPER FUNCTIONS ---
def prepare_cyclic_vars(df):
    df_cyc = df.copy()
    # 1. Prepare Time Blocks
    for col in ["ArrTimeBlk", "DepTimeBlk"]:
        if col in df_cyc.columns:
            # Handle string conversion robustly
            df_cyc[col] = df_cyc[col].astype(str).str[:4]
            # Coerce errors to 0 or handle safe? Assuming standard format per usage
            df_cyc[col] = (
                pd.to_numeric(df_cyc[col], errors="coerce").fillna(0).astype(int) // 100
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
            df_cyc[f"{col}_sin"] = (np.sin(2 * np.pi * df_cyc[col] / max_val) + 1) / 2
            df_cyc[f"{col}_cos"] = (np.cos(2 * np.pi * df_cyc[col] / max_val) + 1) / 2
            df_cyc.drop(columns=[col], inplace=True)

    return df_cyc


def main():
    print(">> Starting Persistence Script...")

    # 1. IDENTIFY CATEGORICAL COLUMNS & COLLECT UNIQUES
    # We need to scan the raw file to build the OrdinalEncoder categories correctly
    # so we don't depend on a small sample.
    print(">> [1/5] Scanning Raw Data for Categories (Chunked)...")

    # Peak at columns first to identify categorical
    preview = pd.read_csv(RAW_FILE, nrows=5)
    # Define vars to process (Cyclic will be removed/transformed, so don't encode them as cat yet if they are numeric)
    # Based on inspect_raw.py:
    # FlightDate (Obj), Airline (Obj), Origin (Obj), Dest (Obj)...
    # Cyclic vars like Month (int) are numeric in raw, so they won't be picked up by exclude=['number'].
    # Which is good, because prepare_cyclic_vars handles them.
    # We only care about VALID categorical vars that enter OrdinalEncoder.

    # Simulation of pipeline flow to get column names:
    # Raw -> Dropna -> Cyclic -> [Assume these are the ones for Ordinal]
    # We need to know which columns become "Object" after Cyclic.
    # Actually, Cyclic only removes Month/etc and adds sin/cos (float).
    # So "Object" columns in Raw (minus Target) are the ones we need to collect uniques for.

    cat_cols = (
        preview.drop(columns=[TARGET], errors="ignore")
        .select_dtypes(exclude=["number"])
        .columns.tolist()
    )

    # Exclude Cyclic Vars from Ordinal Encoding (they are handled separately)
    cyclic_vars_list = [
        "Month",
        "DayOfWeek",
        "Quarter",
        "ArrTimeBlk",
        "DepTimeBlk",
        "CRSArrTime",
        "CRSDepTime",
    ]
    cat_cols = [c for c in cat_cols if c not in cyclic_vars_list]

    print(f"   Categorical Columns to Encode: {cat_cols}")

    unique_values = {col: set() for col in cat_cols}

    chunk_size = 100000
    for chunk in pd.read_csv(RAW_FILE, chunksize=chunk_size):
        chunk.dropna(inplace=True)
        for col in cat_cols:
            if col in chunk.columns:
                unique_values[col].update(
                    chunk[col].astype(str).unique()
                )  # Ensure string

    # Sort categories to ensure deterministic mapping (Alphabetical)
    categories_list = [sorted(list(unique_values[col])) for col in cat_cols]

    # 2. CREATE & SAVE ENCODER
    print(">> [2/5] Creating OrdinalEncoder...")
    encoder = OrdinalEncoder(
        categories=categories_list, handle_unknown="use_encoded_value", unknown_value=-1
    )
    # We don't need to fit since we provided categories, but calling fit helps init params
    # dummy fit
    dummy_df = pd.DataFrame(
        {col: categories_list[i][:1] for i, col in enumerate(cat_cols)}
    )
    encoder.fit(dummy_df)
    # create scaler/shifts/etc placeholders

    # 3. CREATE & SAVE SCALER (Processing ENTIRE dataset in chunks)
    print(">> [3/5] Fitting Scaler on FULL Dataset (chunked processing)...")

    chunk_size = 100000
    global_min = None
    global_max = None
    all_columns = None

    # Also collect statistics for MVI (median/mode)
    numeric_sums = {}
    numeric_counts = {}
    category_value_counts = {}

    for i, chunk in enumerate(pd.read_csv(RAW_FILE, chunksize=chunk_size)):
        chunk.dropna(inplace=True)
        if TARGET in chunk.columns:
            chunk.drop(columns=[TARGET], inplace=True)

        # Apply Pipeline steps
        chunk = prepare_cyclic_vars(chunk)

        # Encode
        chunk[cat_cols] = encoder.transform(chunk[cat_cols])

        if all_columns is None:
            all_columns = chunk.columns.tolist()

        # Track min/max for scaler
        chunk_min = chunk.min()
        chunk_max = chunk.max()

        if global_min is None:
            global_min = chunk_min.copy()
            global_max = chunk_max.copy()
        else:
            global_min = np.minimum(global_min, chunk_min)
            global_max = np.maximum(global_max, chunk_max)

        # Track statistics for MVI
        for col in chunk.columns:
            if col in cat_cols:
                # Mode for categorical
                if col not in category_value_counts:
                    category_value_counts[col] = {}
                for val in chunk[col]:
                    category_value_counts[col][val] = (
                        category_value_counts[col].get(val, 0) + 1
                    )
            else:
                # Sum/Count for median approximation (we'll use mean as proxy)
                if col not in numeric_sums:
                    numeric_sums[col] = 0
                    numeric_counts[col] = 0
                numeric_sums[col] += chunk[col].sum()
                numeric_counts[col] += len(chunk[col])

        if (i + 1) % 10 == 0:
            print(f"   Processed {(i + 1) * chunk_size:,} rows...")

    print("   Full dataset processed.")

    # Handle negatives (Shift logic)
    shifts = {}
    for col in cat_cols:
        if col in global_min.index:
            min_val = global_min[col]
            if min_val < 0:
                shifts[col] = abs(min_val)
                global_min[col] += shifts[col]
                global_max[col] += shifts[col]

    # Create and fit scaler using global min/max
    scaler = MinMaxScaler()
    # Create a dummy dataframe with just min and max rows for fitting
    dummy_df = pd.DataFrame([global_min, global_max], columns=all_columns)
    scaler.fit(dummy_df)

    # Compute MVI statistics
    mvi_stats = {}
    # Numeric: use mean as approximation
    for col in numeric_sums:
        mvi_stats[col] = (
            numeric_sums[col] / numeric_counts[col] if numeric_counts[col] > 0 else 0
        )
    # Categorical: use mode
    for col in category_value_counts:
        mode_val = max(category_value_counts[col], key=category_value_counts[col].get)
        mvi_stats[col] = mode_val

    # 4. SAVE PIPELINE ARTIFACTS (Consolidated)
    print(">> [4/5] Saving Consolidated Pipeline Artifact...")

    # We need final_features BEFORE saving pipeline, so let's load train file to get them
    df_train = pd.read_csv(TRAIN_FILE)
    final_features = [c for c in df_train.columns if c != TARGET]

    pipeline_state = {
        "encoder": encoder,
        "scaler": scaler,
        "cat_cols": cat_cols,
        "shifts": shifts,
        "mvi_stats": mvi_stats,
        "final_features": final_features,
    }
    joblib.dump(pipeline_state, "models/pipeline.joblib")
    print("   Pipeline saved to models/pipeline.joblib")

    # 5. TRAIN MODELS
    print(">> [5/5] Training Models on Final Dataset...")

    # Save features separately? No, it's in pipeline. But kept separate logic for loading if needed,
    # but strictly we should use the one in pipeline.joblib.
    # The 'final_features.json' was used by previous pipeline version.
    # We can remove the separate dump if we rely on pipeline.joblib.
    # I will remove the separate dumps for encoder/etc to clean up, but keeping them doesn't hurt.
    # Requirement says "One single file", implying we should rely on that ONE file.

    X_train = df_train[final_features]

    X_train = df_train[final_features]
    y_train = df_train[TARGET]

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, "models/naive_bayes.joblib")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train, y_train)
    joblib.dump(knn, "models/knn.joblib")

    # Logistic Regression
    print("   Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, "models/logistic_regression.joblib")

    # Decision Tree
    print("   Training Decision Tree...")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    joblib.dump(dt, "models/decision_tree.joblib")

    # MLP
    print("   Training MLP...")
    mlp = MLPClassifier(max_iter=500)
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, "models/mlp.joblib")

    # Random Forest
    print("   Training Random Forest...")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.joblib")

    print("   Models saved.")

    # 5. SAVE PIPELINE CLASS (Or just the definition file? We'll create a dedicated file for the class)
    # We will write the class to 'preparation.py' separately.

    print(">> Persistence Complete!")


if __name__ == "__main__":
    import os

    if not os.path.exists("models"):
        os.makedirs("models")
    main()
