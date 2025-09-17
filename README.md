📄 README.md
# 🧪 Lead Scoring / Cross-Selling (Prueba)

Este repositorio contiene un ejemplo **de prueba** de un modelo de *Lead Scoring* para cross-selling de productos de seguros.  
El objetivo es experimentar cómo entrenar un modelo supervisado, guardar el modelo entrenado en `.joblib`, y luego usarlo para predecir si un lead es **apto** para una recomendación.

---

## ⚙️ Instalación de dependencias

Asegúrate de tener **Python 3.9+** instalado.  
Luego, en la terminal ejecuta:

```bash
pip install pandas scikit-learn joblib

📂 Archivos principales

train_model_db.py → entrena el modelo (Regresión Logística balanceada).

predict_lead_db.py → carga el modelo y predice el score de un lead.

dataset_training_es_v3_db.csv → dataset de entrenamiento (ejemplo).

model_db.joblib → archivo del modelo entrenado.

lead_alto.json / lead_medio.json / lead_bajo.json → ejemplos de leads para probar.

🏋️ Entrenar el modelo

Ejecutar en consola:

python .\train_logistic_model.py --data ".\dataset_training_es_v3.csv" --output ".\model_test_v3.joblib"


Esto:

Lee el dataset de entrenamiento.

Divide en train/test (80/20 estratificado).

Entrena un modelo de Regresión Logística con class_weight="balanced".

Muestra métricas en consola (Accuracy, Recall, F1, AUC, matriz de confusión).

Guarda el modelo en model_db.joblib.

🔮 Predecir leads
Opción 1: con ejemplo integrado
python predict_lead_balanced.py --model model_test_v3.joblib 

Opción 2: pasando un JSON con datos del lead
python predict_lead_db.py --model model_db.joblib --json lead_alto.json

Ejemplo de salida esperada
Score: 82.50
Banda: Recomendación fuerte
¿Apto?: Sí
Producto recomendado: Hogar

📌 Notas

Este repositorio es solo una prueba/POC (Proof of Concept).

El modelo y dataset son experimentales.

