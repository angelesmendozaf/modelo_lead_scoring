import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import warnings
import os
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    # Carga el dataset y valida las columnas requeridas
    print(f"Cargando dataset desde: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    
    # Validar columnas necesarias para calcular score
    required_cols = ['ratio_cuotas_cumplidas', 'cantidad_siniestros', 'ingresos_mensuales', 'antiguedad_cliente']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas para calcular el score: {missing_cols}")
    
    return df


def calculate_score_target(df):
    # Calcula el score objetivo (0-100) basado en múltiples factores
    print("Calculando score objetivo basado en comportamiento del cliente...")
    
    scores = []
    for _, row in df.iterrows():
        # COMPONENTES DEL SCORE
        
        # Comportamiento de pago (40%)
        ratio_pago = row['ratio_cuotas_cumplidas']
        score_pago = ratio_pago * 40
        
        # Historial de siniestros (25%)
        num_siniestros = row['cantidad_siniestros']
        if num_siniestros == 0:
            score_siniestros = 25
        elif num_siniestros <= 2:
            score_siniestros = 25 - (num_siniestros * 5)
        else:
            score_siniestros = max(0, 15 - (num_siniestros - 2) * 3)
        
        # Capacidad económica (20%)
        ingresos = row['ingresos_mensuales']
        if ingresos >= 200000:
            score_ingresos = 20
        elif ingresos >= 100000:
            score_ingresos = 15
        elif ingresos >= 50000:
            score_ingresos = 10
        else:
            score_ingresos = 5
        
        # Antigüedad/lealtad (10%)
        antiguedad = row['antiguedad_cliente']
        score_antiguedad = min(10, (antiguedad / 365) * 3)
        
        # Diversificación de productos (5%)
        num_polizas = row['cantidad_polizas']
        score_diversificacion = min(5, num_polizas * 1.5)
        
        # SCORE FINAL
        score_base = score_pago + score_siniestros + score_ingresos + score_antiguedad + score_diversificacion
        ruido = np.random.uniform(-5, 5)
        score_final = np.clip(score_base + ruido, 0, 100)
        
        scores.append(round(score_final, 1))
    
    df['score_target'] = scores
    
    # Estadísticas del score generado
    print("Estadísticas del score calculado:")
    print(f"  • Promedio: {np.mean(scores):.1f}")
    print(f"  • Mediana: {np.median(scores):.1f}")
    print(f"  • Desv. estándar: {np.std(scores):.1f}")
    print(f"  • Rango: {min(scores):.1f} – {max(scores):.1f}")
    
    # Distribución por niveles
    df['score_nivel'] = pd.cut(df['score_target'], bins=[0, 40, 70, 100], labels=['Bajo', 'Medio', 'Alto'])
    print("Distribución por niveles:")
    nivel_dist = df['score_nivel'].value_counts(normalize=True)
    for nivel, pct in nivel_dist.items():
        print(f"  • {nivel}: {pct:.3f}")
    
    return df


def prepare_features(df):
    # Prepara las features para el modelo (evitando leakage)
    print("Preparando features para predicción...")
    
    excluded_features = ['persona_id', 'score_target', 'score_nivel']
    categorical_features = ['preferencia_contacto', 'ocupacion', 'genero']
    
    numeric_features = [
        col for col in df.columns
        if col not in excluded_features + categorical_features
        and df[col].dtype in ['int64', 'float64']
    ]
    
    print(f"Features numéricas: {len(numeric_features)}")
    print(f"Features categóricas: {len(categorical_features)}")
    
    df_features = df.copy()
    label_encoders, encoded_features = {}, []
    
    for col in categorical_features:
        if col in df_features.columns:
            le = LabelEncoder()
            encoded_col = f"{col}_encoded"
            df_features[encoded_col] = le.fit_transform(df_features[col].astype(str))
            label_encoders[col] = le
            encoded_features.append(encoded_col)
    
    feature_columns = numeric_features + encoded_features
    X, y = df_features[feature_columns], df_features['score_target']
    
    print(f"Total de features: {len(feature_columns)}")
    print(f"Rango del target: {y.min():.1f} – {y.max():.1f}")
    
    return X, y, feature_columns, label_encoders


def train_scoring_models(X, y, feature_names):
    # Entrena y evalúa modelos de regresión para el scoring
    print("Entrenando modelos de scoring...")
    if len(X) < 200:
        print("Advertencia: pocos registros, el modelo puede sobreajustar.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Regresión Lineal': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=10,
                min_samples_leaf=5
            ))
        ])
    }
    
    results = {}
    for name, pipeline in models.items():
        print(f"Evaluando {name}...")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        pipeline.fit(X_train, y_train)
        y_pred = np.clip(pipeline.predict(X_test), 0, 100)
        
        metrics = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'pipeline': pipeline,
            'predictions': y_pred
        }
        results[name] = metrics
        
        print(f"  • CV R²: {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})")
        print(f"  • Test R²: {metrics['r2_score']:.3f}")
        print(f"  • RMSE: {metrics['rmse']:.2f}")
        print(f"  • MAE: {metrics['mae']:.2f}")
    
    best_model_name = max(results, key=lambda k: results[k]['r2_score'])
    best_model = results[best_model_name]
    
    print(f"\nMejor modelo: {best_model_name}")
    print(f"Métricas:")
    print(f"  • R² Score: {best_model['r2_score']:.3f}")
    print(f"  • RMSE: {best_model['rmse']:.2f}")
    print(f"  • MAE: {best_model['mae']:.2f}")
    
    return best_model, X_test, y_test, best_model_name


def analyze_feature_importance(pipeline, feature_names, top_n=10):
    # Analiza la importancia de features
    print(f"\nTop {top_n} features más influyentes en el scoring:")
    model = pipeline.named_steps['model']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("No se pudo calcular la importancia de features")
        return None
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    for _, row in feature_importance_df.head(top_n).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance_df


def validate_scoring_model(y_test, y_pred):
    # Valida que el modelo de scoring tenga sentido de negocio
    print("\nValidando modelo de scoring...")
    df_validation = pd.DataFrame({
        'score_real': y_test.values,
        'score_predicho': y_pred,
        'diferencia': np.abs(y_test.values - y_pred)
    })
    
    print("Distribución de errores:")
    print(f"  • Error promedio: {df_validation['diferencia'].mean():.2f} puntos")
    print(f"  • Error mediano: {df_validation['diferencia'].median():.2f} puntos")
    print(f"  • % error < 5 pts: {(df_validation['diferencia'] < 5).mean():.3f}")
    print(f"  • % error < 10 pts: {(df_validation['diferencia'] < 10).mean():.3f}")
    print(f"  • Predicciones fuera de rango: {((y_pred < 0) | (y_pred > 100)).sum()}")
    print(f"  • Correlación real vs predicho: {np.corrcoef(y_test, y_pred)[0, 1]:.3f}")
    
    return df_validation


def save_scoring_model(pipeline, feature_names, label_encoders, filepath, model_name, metrics):
    # Guarda el modelo de scoring en formato .joblib
    print("Guardando modelo de scoring...")
    bundle = {
        'pipeline': pipeline,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'model_type': model_name,
        'model_purpose': 'customer_scoring',
        'score_range': [0, 100],
        'score_levels': {'Bajo': [0, 40], 'Medio': [41, 70], 'Alto': [71, 100]},
        'training_date': datetime.now().isoformat(),
        'metrics': {k: metrics[k] for k in ['r2_score', 'rmse', 'mae', 'cv_r2_mean', 'cv_r2_std']},
        'feature_order': feature_names,
        'version': '2.0_scoring'
    }
    joblib.dump(bundle, filepath)
    print(f"Modelo guardado en: {filepath}")


def main(args):
    # Flujo principal: carga datos, entrena, valida y guarda modelo
    print("INICIANDO ENTRENAMIENTO DEL MODELO DE SCORING v2.0")
    print("Objetivo: generar .joblib para producción")
    print("=" * 70)
    
    try:
        df = load_and_preprocess_data(args.data)
        df = calculate_score_target(df)
        X, y, feature_names, label_encoders = prepare_features(df)
        best_model, X_test, y_test, model_name = train_scoring_models(X, y, feature_names)
        analyze_feature_importance(best_model['pipeline'], feature_names, top_n=15)
        validation_results = validate_scoring_model(y_test, best_model['predictions'])
        
        save_scoring_model(best_model['pipeline'], feature_names, label_encoders, args.output, model_name, best_model)
        
        print("\nENTRENAMIENTO COMPLETADO")
        print(f"Modelo final: {model_name} con R² = {best_model['r2_score']:.3f}")
        print(f"Error promedio: {best_model['mae']:.2f} puntos")
        
        if args.save_analysis:
            analysis_file = args.output.replace('.joblib', '_analysis.csv')
            validation_results.to_csv(analysis_file, index=False)
            print(f"Análisis detallado guardado en: {analysis_file}")
    except Exception as e:
        print(f"ERROR en entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo de scoring de clientes (0-100)")
    parser.add_argument("--data", default="./csv/dataset_cross_selling_completo.csv", help="Ruta al dataset CSV")
    parser.add_argument("--output", default="./models/customer_scoring_model_v2.joblib", help="Ruta del modelo .joblib")
    parser.add_argument("--save-analysis", action="store_true", help="Guardar análisis detallado en CSV")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args)
