import joblib
import pandas as pd

# 1. Cargar modelo combinado
bundle = joblib.load("../models/modelo_cross_selling_extended.joblib")

model_score = bundle["model_score"]
model_product = bundle["model_product"]
label_encoders = bundle["label_encoders"]
encoder_target = bundle["encoder_target"]
features = bundle["feature_names"]

# 2. Lead de prueba (ejemplo malo → debería dar bajo)
lead = {
  "edad": 32,
  "genero": "femenino",
  "provincia": "Mendoza",
  "nivel_educativo": "secundario",
  "ocupacion": "empleado",
  "hijos": 2,
  "vivienda_propia": 0,
  "posee_auto": 0,
  "antiguedad_cliente": 24,
  "cantidad_polizas": 1,
  "cantidad_pagos": 24,
  "cuotas_cumplidas": 24,
  "cuotas_impagas": 0,
  "pagos_parciales": 0,
  "ratio_cuotas_cumplidas": 1.0,
  "monto_promedio_pago": 8000,
  "cantidad_siniestros": 0,
  "monto_promedio_siniestro": 0,
  "dias_desde_ultimo_siniestro": 999,
  "tiene_auto": 0,
  "tiene_hogar": 1,
  "tiene_vida": 0,
  "tiene_salud": 0,
  "costo_mensual_total_seguro": 8000,
  "suma_asegurada_total": 2500000,
  "prima_promedio": 8000,

}



# 3. Transformar a DataFrame y codificar
X_new = pd.DataFrame([lead])
for col, le in label_encoders.items():
    if col in X_new.columns:
        X_new[col] = le.transform(X_new[col].astype(str))

# 4. Predecir score y producto
score = model_score.predict(X_new[features])[0]
y_proba = model_product.predict_proba(X_new[features])[0]

# Mapeo producto ↔ probabilidad
proba_dict = {
    producto: prob
    for producto, prob in zip(encoder_target.classes_, y_proba)
}

# 5. Filtrar productos que el cliente ya tenga
productos_cliente = {
    "auto": lead["tiene_auto"],
    "hogar": lead["tiene_hogar"],
    "vida": lead["tiene_vida"],
    "salud": lead["tiene_salud"],
   
}
proba_dict_filtrado = {
    producto: prob
    for producto, prob in proba_dict.items()
    if producto != "ninguno" and productos_cliente.get(producto, 0) == 0
}

# 6. Decidir recomendaciones según score
if score <= 40:
    productos_recomendados = ["ninguno"]
    apto = "No apto"
    nivel = "Bajo"
elif score <= 70:
    # Top 3 productos válidos
    productos_recomendados = sorted(proba_dict_filtrado, key=proba_dict_filtrado.get, reverse=True)[:3]
    apto = "Apto (medio)"
    nivel = "Medio"
else:
    # Solo el top 1
    if proba_dict_filtrado:  # asegurar que haya algo después del filtro
        productos_recomendados = [max(proba_dict_filtrado, key=proba_dict_filtrado.get)]
    else:
        productos_recomendados = ["ninguno"]  # fallback si ya tiene todos
    apto = "Apto (alto)"
    nivel = "Alto"

# 7. Mostrar resultado
print(f"✅ Productos recomendados: {productos_recomendados}")
print(f"📊 Score: {score:.2f}% → {apto} ({nivel})")
