# Cross-Selling Model

Este proyecto implementa un sistema de **scoring** y **recomendación de productos de seguros** para clientes existentes, utilizando **Python y scikit-learn**.

El modelo combina:
- **Random Forest Regressor**: para calcular el score del cliente (0–100).
- **Random Forest Classifier**: para predecir el producto objetivo de cross-selling.

---

## 📂 Estructura del proyecto
```text
mi_proyecto/
│
├── train_model.py                       # Script de entrenamiento del modelo
├── predict_test.py                     # Script de predicción con un cliente
├── models/                        # Modelos entrenados guardados en formato .joblib
│   └── modelo_cross_selling.joblib
├── csv/                           # Dataset de entrenamiento
│   └── dataset_cross_selling.csv
└── leads/                         # Ejemplos de clientes en formato JSON
    ├── lead_b.json
    ├── lead_a.json
    ├── lead.json
    └── lead_medio.json
```

---

## ⚙️ Requisitos
- Python 3.9 o superior  
- Librerías necesarias:
```bash
pip install pandas numpy scikit-learn joblib
```

---

## 🏋️ Entrenamiento del modelo
Ejecutar el script de entrenamiento con el dataset (usa las rutas por defecto):

```bash
python train_model.py
```

También se pueden indicar rutas personalizadas:

```bash
python train_model.py --data ./csv/dataset_cross_selling.csv --output ./models/modelo_cross_selling.joblib
```

Este proceso realiza lo siguiente:
- Carga el dataset de clientes desde `csv/`.
- Calcula la columna `score_target` aplicando reglas de negocio.
- Entrena dos modelos:
  - `RandomForestRegressor` para el score de cliente.
  - `RandomForestClassifier` para el producto objetivo.
- Evalúa el desempeño con métricas básicas.
- Guarda el modelo entrenado en `./models/modelo_cross_selling.joblib`.

---

## 🔮 Predicción con un cliente
Para ejecutar una predicción se debe proveer un archivo JSON con los datos de un cliente:

```bash
python predict_test.py --lead ./leads/lead.json
```

Ejemplo de salida:
```text
=== RESULTADO DE PREDICCIÓN ===
Score: 62.4 → Apto (medio)
Productos recomendados: ['hogar', 'vida', 'salud']
Probabilidades completas: {'auto': 0.01, 'hogar': 0.33, 'ninguno': 0.01, 'salud': 0.31, 'vida': 0.34}
```

---

## 📊 Lógica de negocio

### Score (0–100)
- **No apto (0–40):** no se recomienda ningún producto.  
- **Apto medio (41–70):** se recomiendan hasta 3 productos válidos.  
- **Apto alto (71–100):** se recomienda solo el producto con mayor probabilidad.  

### Recomendación de productos
- Se filtran los productos que el cliente ya posee (`tiene_auto`, `tiene_hogar`, `tiene_vida`, `tiene_salud`).  
- El modelo selecciona los productos válidos según el score y las probabilidades de clasificación.  

---

## 📁 Leads de prueba
En la carpeta `./leads/` se incluyen archivos JSON listos para probar el modelo:
- `lead_base.json`: cliente de nivel medio.  
- `lead_alto.json`: cliente premium con buen historial.  
- `lead_bajo.json`: cliente con alto riesgo (no apto).  
- `lead_random.json`: cliente promedio para testear variaciones.  

Ejemplo de ejecución:
```bash
python predict.py --lead ./leads/lead_alto.json
python predict.py --lead ./leads/lead_bajo.json
```

---
