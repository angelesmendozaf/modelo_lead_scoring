import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
import joblib
import os

def calcular_score_reglas(df):
    """Crea una columna score_target con reglas de negocio fuertes"""
    scores = []
    for _, row in df.iterrows():
        score = 0

        # ✅ Regla dura: cuotas impagas
        if row.get("cuotas_impagas", 0) > 0:
            scores.append(np.random.uniform(10, 30))  # siempre bajo
            continue

        # ✅ Regla dura: ratio bajo
        ratio = row.get("ratio_cuotas_cumplidas", 0)
        if ratio < 0.7:
            scores.append(np.random.uniform(20, 35))  # bajo
            continue

        # ✅ Regla dura: antigüedad muy baja
        if row.get("antiguedad_cliente", 0) < 6:
            scores.append(np.random.uniform(15, 30))  # bajo
            continue

        # ✅ Regla dura: cliente excelente
        if (
            ratio > 0.9
            and row.get("cuotas_impagas", 0) == 0
            and row.get("antiguedad_cliente", 0) > 24
            and row.get("cantidad_siniestros", 0) == 0
        ):
            scores.append(np.random.uniform(80, 95))  # alto
            continue

        # 📊 Resto: cálculo normal (Medio)
        # Pagos (40%)
        score += ratio * 40

        # Siniestros (25%)
        if row["cantidad_siniestros"] == 0:
            score += 25
        elif row["cantidad_siniestros"] == 1:
            score += 20
        else:
            score += 10

        # Antigüedad (20%)
        if row["antiguedad_cliente"] > 12:
            score += min(20, row["antiguedad_cliente"] / 6)

        # Diversificación (10%)
        score += min(10, row["cantidad_polizas"] * 2)

        # Ruido leve
        score += np.random.uniform(-2, 2)

        scores.append(max(0, min(100, round(score, 1))))

    df["score_target"] = scores
    return df

def main(args):
    # 1. Cargar dataset
    df = pd.read_csv(args.data)
    print(f"✅ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

    # 2. Generar columna score_target
    df = calcular_score_reglas(df)

    # 3. Features y target
    excluded = ["producto_objetivo", "adquirio", "persona_id", "preferencia_contacto", "score_target"]
    features = [c for c in df.columns if c not in excluded]

    # Codificación de features
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        if col != "producto_objetivo":  # ⚠️ no encodeamos el target acá
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    X = df_encoded[features]

    # Target: producto_objetivo
    le_target = LabelEncoder()
    y_product = le_target.fit_transform(df["producto_objetivo"])

    # Score target
    y_score = df["score_target"]

    # 4. Modelos
    model_score = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model_product = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12)

    # Entrenar
    model_score.fit(X, y_score)
    model_product.fit(X, y_product)

    # 5. Métricas Score
    y_pred_score = model_score.predict(X)
    mae = mean_absolute_error(y_score, y_pred_score)
    r2 = r2_score(y_score, y_pred_score)
    print("\n=== Métricas SCORING ===")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")

    # 6. Métricas Productos
    X_train, X_test, y_train, y_test = train_test_split(X, y_product, test_size=0.2, random_state=42, stratify=y_product)
    y_pred = model_product.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Métricas PRODUCTO ===")
    print(f"Accuracy: {acc:.3f}")
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

    # 7. Guardar en bundle único
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
    parser.add_argument("--data", default="../csv/dataset_cross_selling_extended.csv", help="Ruta al dataset CSV")
    parser.add_argument("--output", default="../models/modelo_cross_selling_extended.joblib", help="Ruta del archivo .joblib")
    args = parser.parse_args()
    main(args)
