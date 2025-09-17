import argparse                      # Lee parámetros desde la consola (CLI).
import pandas as pd                  # Carga y manipula datos tabulares (CSV).
from sklearn.model_selection import train_test_split  # Split train/test.
from sklearn.preprocessing import StandardScaler      # Escalado de features.
from sklearn.linear_model import LogisticRegression   # Modelo ML (Reg. Logística).
from sklearn.pipeline import Pipeline                 # Encadena pasos (scaler+modelo).
from sklearn.metrics import (                         # Métricas de evaluación.
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import joblib                         # Guardar/cargar el modelo entrenado.


def main(args):
    # 1) Cargar dataset
    df = pd.read_csv(args.data)       # Lee el CSV pasado por --data.

    # 2) Separar X / y
    if "adquirio" not in df.columns:  # Validación: debe existir la columna target.
        raise ValueError("El CSV debe contener la columna 'adquirio' (target).")

    if "persona_id" in df.columns:    # Si existe un id, lo quitamos de features.
        X = df.drop(columns=["adquirio", "persona_id"])
    else:
        X = df.drop(columns=["adquirio"])

    y = df["adquirio"]                # Variable objetivo binaria (0/1).

    # 3) Split 80/20 (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,               # 20% test.
        random_state=42,              # Reproducibilidad.
        stratify=y                    # Mantiene la proporción de clases en train/test.
    )

    # 4) Pipeline: escalado + Regresión Logística balanceada
    pipeline = Pipeline([
        ("scaler", StandardScaler()),                 # Estandariza features numéricos
        ("model", LogisticRegression(                 # Modelo: Regresión Logística
            max_iter=1000,                            # Iteraciones máximas
            class_weight="balanced"                   # Pesa más la clase minoritaria
        ))
    ])

    # 5) Entrenar
    pipeline.fit(X_train, y_train)    # Ajusta scaler y modelo con los datos de train.

    # 6) Evaluar
    y_pred = pipeline.predict(X_test)               # Predicciones (clase 0/1).
    y_prob = pipeline.predict_proba(X_test)[:, 1]   # Prob. de clase positiva.

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
    product_mapping = {              # Mapeo de códigos de producto → nombre amigable
        1: "Auto", 2: "Hogar", 3: "Vida", 4: "Salud",
        5: "Accidentes", 6: "Mascotas", 7: "Riesgos"
    }
    bundle = {
        "pipeline": pipeline,                    # Pipeline entrenado (scaler+modelo).
        "feature_names": X.columns.tolist(),     # Orden de columnas esperado.
        "product_mapping": product_mapping       # Mapeo a usar en inferencia.
    }
    joblib.dump(bundle, args.output)             # Persiste el paquete en disco.
    print(f"Modelo guardado en {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults para ejecutar sin parámetros
    parser.add_argument(
        "--data", default="dataset_training_es_v3.csv",
        help="Ruta al dataset CSV (default: dataset_training_es_v3.csv)"
    )
    parser.add_argument(
        "--output", default="model_test_v3.joblib",
        help="Ruta del modelo .joblib de salida (default: model_test_v3.joblib)"
    )
    args = parser.parse_args()
    main(args)
