"""
Entrena una regresión logística balanceada para scoring/cross-selling.
Lee un CSV, divide 80/20, entrena, imprime métricas y guarda el modelo .joblib.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import joblib


def main(args):
    # 1) Cargar dataset
    df = pd.read_csv(args.data)
    # 2) Separar X / y
    if "adquirio" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'adquirio' (target).")
    if "persona_id" in df.columns:
        X = df.drop(columns=["adquirio", "persona_id"])
    else:
        X = df.drop(columns=["adquirio"])
    y = df["adquirio"]

    # 3) Split 80/20 (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4) Pipeline: escalado + Regresión Logística balanceada
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    # 5) Entrenar
    pipeline.fit(X_train, y_train)

    # 6) Evaluar
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("== Métricas de evaluación ==")
    print(f"Precisión (accuracy): {accuracy_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # 7) Guardar modelo + metadatos
    product_mapping = {
        1: "Auto", 2: "Hogar", 3: "Vida", 4: "Salud",
        5: "Accidentes", 6: "Mascotas", 7: "Riesgos"
    }
    bundle = {
        "pipeline": pipeline,
        "feature_names": X.columns.tolist(),
        "product_mapping": product_mapping
    }
    joblib.dump(bundle, args.output)
    print(f"Modelo guardado en {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults para ejecutar sin parámetros
    parser.add_argument("--data", default="dataset_training_es_v3.csv",
                        help="Ruta al dataset CSV (default: dataset_training_es_v3.csv)")
    parser.add_argument("--output", default="model_test_v3.joblib",
                        help="Ruta del modelo .joblib de salida (default: model_test_v3.joblib)")
    args = parser.parse_args()
    main(args)
