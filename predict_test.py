import argparse
import json
import joblib
import pandas as pd

# === Argumentos ===
parser = argparse.ArgumentParser(description="Predicción para un cliente existente")
parser.add_argument("--lead", required=True, help="Ruta al archivo JSON del cliente")
args = parser.parse_args()

# === Cargar modelo entrenado ===
bundle = joblib.load("./models/modelo_cross_selling_1.joblib")

model_score = bundle["model_score"]
model_product = bundle["model_product"]
label_encoders = bundle["label_encoders"]
encoder_target = bundle["encoder_target"]
features = bundle["feature_names"]

# === Cargar cliente desde JSON ===
with open(args.lead, "r", encoding="utf-8") as f:
    lead = json.load(f)

X_new = pd.DataFrame([lead])
for col, le in label_encoders.items():
    if col in X_new.columns:
        X_new[col] = le.transform(X_new[col].astype(str))

# === Predicciones ===
score = model_score.predict(X_new[features])[0]
y_proba = model_product.predict_proba(X_new[features])[0]

# Probabilidades por producto
proba_dict = {producto: prob for producto, prob in zip(encoder_target.classes_, y_proba)}

# Filtrar productos que el cliente ya tiene
productos_cliente = {
    "auto": lead["tiene_auto"],
    "hogar": lead["tiene_hogar"],
    "vida": lead["tiene_vida"],
    "salud": lead["tiene_salud"]
}
proba_dict_filtrado = {
    producto: prob
    for producto, prob in proba_dict.items()
    if producto != "ninguno" and productos_cliente.get(producto, 0) == 0
}

# Decidir recomendaciones según score
if score <= 40:
    productos_recomendados = ["ninguno"]
    apto = "No apto"
    nivel = "Bajo"
elif score <= 70:
    productos_recomendados = sorted(proba_dict_filtrado, key=proba_dict_filtrado.get, reverse=True)[:3]
    apto = "Apto (medio)"
    nivel = "Medio"
else:
    productos_recomendados = [max(proba_dict_filtrado, key=proba_dict_filtrado.get)] if proba_dict_filtrado else ["ninguno"]
    apto = "Apto (alto)"
    nivel = "Alto"

# === Resultado ===
print("\n=== RESULTADO DE PREDICCIÓN ===")
print(f"📊 Score: {score:.2f} → {apto} ({nivel})")
print(f"✅ Productos recomendados: {productos_recomendados}")
print(f"🔮 Probabilidades completas: {proba_dict}")
