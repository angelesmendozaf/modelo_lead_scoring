import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
import joblib
import os


# === Función para calcular score_target basado en reglas ===
def calcular_score_reglas(df):
    scores = []
    for _, row in df.iterrows():
        score = 0

        # Regla fuerte: cuotas impagas
        if row.get("cuotas_impagas", 0) > 3:
            scores.append(np.random.uniform(10, 30))  # Muy bajo
            continue

        # Regla: antigüedad muy baja
        if row.get("antiguedad_cliente", 0) < 6:
            scores.append(np.random.uniform(15, 30))  # Bajo
            continue

        # Regla: cliente excelente (sin impagos, antigüedad larga, sin siniestros)
        if (
            row.get("cuotas_impagas", 0) == 0
            and row.get("antiguedad_cliente", 0) > 24
            and row.get("cantidad_siniestros", 0) == 0
        ):
            scores.append(np.random.uniform(80, 95))  # Alto
            continue

        # === Score ponderado ===
        # Antigüedad (hasta 25 puntos)
        score += min(25, row.get("antiguedad_cliente", 0) / 2)

        # Cuotas impagas (penaliza)
        if row.get("cuotas_impagas", 0) == 0:
            score += 20
        elif row.get("cuotas_impagas", 0) == 1:
            score += 10
        else:
            score -= 10

        # Siniestros (máx 20 puntos)
        if row.get("cantidad_siniestros", 0) == 0:
            score += 20
        elif row.get("cantidad_siniestros", 0) == 1:
            score += 10
        else:
            score += 5

        # Diversificación (más pólizas, más score, máx 15)
        score += min(15, row.get("cantidad_polizas", 0) * 3)

        # Propiedad (vivienda propia suma 5)
        if row.get("vivienda_propia", 0) == 1:
            score += 5

        # Ruido leve para no dejarlo tan determinístico
        score += np.random.uniform(-2, 2)

        scores.append(max(0, min(100, round(score, 1))))

    df["score_target"] = scores
    return df


def main(args):
    # Cargar dataset
    df = pd.read_csv(args.data)
    print(f"✅ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

    # Generar columna score_target
    df = calcular_score_reglas(df)

    # Features y target
    excluded = ["producto_objetivo", "adquirio", "score_target"]
    features = [c for c in df.columns if c not in excluded]

    # Codificación de variables categóricas
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        if col != "producto_objetivo":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    X = df_encoded[features]

    # Target producto
    le_target = LabelEncoder()
    y_product = le_target.fit_transform(df["producto_objetivo"])

    # Target score
    y_score = df["score_target"]

    # Modelos
    model_score = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model_product = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12)

    # Entrenar
    model_score.fit(X, y_score)
    model_product.fit(X, y_product)

    # Métricas Score
    y_pred_score = model_score.predict(X)
    mae = mean_absolute_error(y_score, y_pred_score)
    r2 = r2_score(y_score, y_pred_score)
    print("\n=== Métricas SCORING ===")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")

    # Métricas Productos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_product, test_size=0.2, random_state=42, stratify=y_product
    )
    y_pred = model_product.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Métricas PRODUCTO ===")
    print(f"Accuracy: {acc:.3f}")
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

    # Guardar modelo
    bundle = {
        "model_score": model_score,
        "model_product": model_product,
        "feature_names": features,
        "label_encoders": label_encoders,
        "encoder_target": le_target,
        "class_labels": le_target.classes_.tolist(),
        "score_levels": {"Bajo": [0, 40], "Medio": [41, 70], "Alto": [71, 100]}
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"\n✅ Modelo combinado guardado en: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo de cross-selling con score + producto")
    parser.add_argument("--data", default="./csv/dataset_cross_selling_1.csv", help="Ruta al dataset CSV")
    parser.add_argument("--output", default="./models/modelo_cross_selling_1.joblib", help="Ruta del archivo .joblib")
    args = parser.parse_args()
    main(args)

