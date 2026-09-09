import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_PATH = "data/MiningProcess_Flotation_Plant_Database.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "% Silica Concentrate"
IRON_COL = "% Iron Concentrate"


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the UCT/Kaggle dataset and place it there."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # The public dataset commonly stores decimal commas.
    for col in df.columns:
        if df[col].dtype == object and col != "date":
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .replace(["nan", "None", ""], np.nan)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    return df


def prepare_hourly(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    work = df.copy()

    if "date" in work.columns:
        # The UCT problem statement notes mixed-frequency measurements.
        # Median aggregation gives a consistent hourly representation.
        hourly = work.set_index("date")[numeric_cols].resample("1h").median()
        hourly = hourly.dropna(subset=[TARGET])
        return hourly

    return work[numeric_cols].dropna(subset=[TARGET])


def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        model_name = "XGBoost Regressor"
    except Exception:
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "Random Forest Regressor"

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    metrics = {
        "experiment": label,
        "model": model_name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": rmse,
        "R2": r2_score(y_test, pred),
        "test_rows": len(y_test),
    }

    # Actual vs predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred, alpha=0.35)
    low = min(y_test.min(), pred.min())
    high = max(y_test.max(), pred.max())
    plt.plot([low, high], [low, high], linewidth=2)
    plt.xlabel("Actual % Silica Concentrate")
    plt.ylabel("Predicted % Silica Concentrate")
    plt.title(f"Actual vs Predicted — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"actual_vs_predicted_{label}.png"), dpi=180)
    plt.close()

    return metrics, pipe


def main():
    df = load_data(DATA_PATH)
    print("Raw shape:", df.shape)
    print("Columns:", list(df.columns))

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found. Available columns: {list(df.columns)}"
        )

    hourly = prepare_hourly(df)
    hourly.to_csv(os.path.join(OUT_DIR, "hourly_processed.csv"))

    # Remove target and obvious non-feature time index.
    feature_cols = [c for c in hourly.columns if c != TARGET]

    # Experiment 1: all usable numeric features.
    X_all = hourly[feature_cols]
    y = hourly[TARGET]
    results = []

    metrics_all, model_all = train_and_evaluate(
        X_all, y, "with_all_features"
    )
    results.append(metrics_all)

    # Experiment 2: explicitly answer UCT's question about % Iron Concentrate.
    if IRON_COL in X_all.columns:
        X_no_iron = X_all.drop(columns=[IRON_COL])
        metrics_no_iron, model_no_iron = train_and_evaluate(
            X_no_iron, y, "without_iron_concentrate"
        )
        results.append(metrics_no_iron)

    # Correlation heatmap.
    corr = hourly.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=180)
    plt.close()

    # Target distribution.
    plt.figure(figsize=(7, 5))
    sns.histplot(y.dropna(), kde=True)
    plt.xlabel("% Silica Concentrate")
    plt.title("Distribution of Target Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "silica_distribution.png"), dpi=180)
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "model_results.csv"), index=False)

    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
