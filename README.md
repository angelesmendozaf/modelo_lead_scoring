# Lead Scoring & Cross-Selling ML

Este proyecto implementa un sistema de **Lead Scoring** y **Cross-Selling** utilizando **Python** y **Machine Learning**, con el objetivo de analizar leads de una compañía de seguros, predecir la probabilidad de conversión y estimar ingresos adicionales a partir de productos relacionados.

## Estado del proyecto

El proyecto está en desarrollo activo. Actualmente incluye:
- Generación de datasets sintéticos realistas (con Faker y NumPy).
- Preprocesamiento de datos para ML.
- Entrenamiento de modelos supervisados.
- Exportación del modelo entrenado en formato `.joblib`.
- Scripts para predicción de nuevos leads.

---

## Estructura

    └── main/
        ├── /csv/                                   # Datasets generados
        ├── /leads/                                 # Modelos de prueba para comparaciones 
        ├── /models/                                # Modelos entrenados en formato .joblib 
        ├── generate_dataset.py                     # Generador de datasets
        ├── predict_lead_balanced.py                # Script para hacer predicciones con leads nuevos v1
        ├── predict_customer_scoring.py             # Script para hacer predicciones con leads nuevos v2
        ├── train_logistic_model.py                 # Script de Regresión Logística balanceada
        ├── train_modelv2.py                        # Script de entrenamiento de modelos v2
        ├── requeriments.txt                        # Dependencias
        └── README.md                               # Documentación del proyecto 

---

## Tecnologías utilizadas

- **Python 3.10+**
- **Faker** para generación de datos sintéticos
- **Pandas** y **NumPy** para manipulación de datos
- **Scikit-learn** para entrenamiento y evaluación de modelos
- **Joblib** para serializar modelos

---

## Cómo empezar

1.  **Clona el repositorio**

    ```bash
    git clone https://github.com/angelesmendozaf/modelo_lead_scoring
    ```

2.  **Abrir el proyecto**

    ```bash
    python -m venv venv
    source venv/bin/activate     # Linux/Mac
    venv\Scripts\activate        # Windows
    ```

3. Instalar dependencias

    ```bash
    pip install -r requirements.txt
    ```

---

## Uso del proyecto

### 🔹Generar dataset

En caso de no tener ningun dataset generado. Ejecutar el generador de datos:

    ```bash
    python generate_dataset.py
    ```

Esto creará un CSV dentro de ./csv/.

### 🔹Entrenar el modelo

Se selecciona un Script de entrenamiento para comenzar con el dataset ya listo.
Ejemplo de uso:

    ```bash
    python train_modelv2.py --data "./csv/dataset_cross_selling_completo.csv" --output "./models/scoring_model_v2.joblib"
    ```
    
Opciones disponibles:
* --data: ruta al dataset de entrada
* --output: ruta donde guardar el modelo entrenado

### 🔹Calcular score de cliente

Ejemplo de uso:

    ```bash
    python predict_customer_scoring.py --model ".\models\scoring_model_v2.joblib" --json-file ".\leads\lead1.json"
    ```
    
Opciones disponibles:
* --model: ruta al modelo de entrada
* --jason-file: ruta del cliente a consultar

---

## Permisos y requisitos

Este proyecto no requiere permisos especiales, solo:
* **Python 3.10+**
* Librerías listadas en requirements.txt

### Próximos pasos:
* Añadir validaciones más estrictas en la generación de dataset.
* Mejorar la calibración del modelo de probabilidad.