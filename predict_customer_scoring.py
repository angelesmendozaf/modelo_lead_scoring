import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import argparse
import sys
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class CustomerScoringPredictor:
    def __init__(self, model_path: str):
        logging.info(f"Cargando modelo desde: {model_path}")

        try:
            self.model_bundle = joblib.load(model_path)
            self.pipeline = self.model_bundle['pipeline']
            self.feature_names = self.model_bundle['feature_names']
            self.label_encoders = self.model_bundle.get('label_encoders', {})
            self.score_levels = self.model_bundle.get('score_levels', {
                'Bajo': [0, 40],
                'Medio': [41, 70],
                'Alto': [71, 100]
            })

            logging.info(f"Modelo cargado exitosamente")
            logging.info(f"Tipo: {self.model_bundle.get('model_type', 'unknown')}")
            logging.info(f"Propósito: {self.model_bundle.get('model_purpose', 'customer_scoring')}")
            logging.info(f"Features esperadas: {len(self.feature_names)}")

        except Exception as e:
            logging.error(f"Error cargando modelo: {str(e)}")
            raise

    def validate_input(self, customer_data: dict) -> bool:
        """Valida que el JSON contenga todas las features necesarias"""
        missing_features = []
        for feature in self.feature_names:
            original_feature = feature.replace('_encoded', '')
            if original_feature not in customer_data and feature not in customer_data:
                missing_features.append(original_feature)

        if missing_features:
            raise ValueError(f"Faltan las siguientes features: {missing_features}")

        return True

    def preprocess_input(self, customer_data: dict) -> list:
        """Preprocesa el input JSON para que coincida con el formato del modelo"""
        processed_features = []

        for feature in self.feature_names:
            if feature.endswith('_encoded'):
                original_feature = feature.replace('_encoded', '')
                value = str(customer_data.get(original_feature, ""))
                if original_feature in self.label_encoders:
                    le = self.label_encoders[original_feature]
                    try:
                        encoded_value = le.transform([value])[0]
                    except ValueError:
                        logging.warning(
                            f"Valor '{value}' no visto en entrenamiento para {original_feature}, usando valor por defecto"
                        )
                        encoded_value = 0
                    processed_features.append(encoded_value)
                else:
                    processed_features.append(hash(value) % 10)
            else:
                processed_features.append(float(customer_data.get(feature, 0.0)))

        return processed_features

    def get_score_level(self, score: float) -> str:
        """Determina el nivel del score (Bajo/Medio/Alto)"""
        for nivel, (min_score, max_score) in self.score_levels.items():
            if min_score <= score <= max_score:
                return nivel
        return 'Indefinido'

    def predict_score(self, customer_json: dict) -> dict:
        """Predice el score para un cliente dado su JSON"""
        try:
            self.validate_input(customer_json)
            features = self.preprocess_input(customer_json)
            score_raw = self.pipeline.predict([features])[0]

            score = max(0, min(100, round(score_raw, 1)))
            nivel = self.get_score_level(score)

            result = {
                'score': score,
                'nivel': nivel,
                'timestamp': datetime.now().isoformat(),
                'input_features': len(features),
                'model_version': self.model_bundle.get('version', '2.0')
            }
            return result

        except Exception as e:
            return {
                'error': str(e),
                'score': None,
                'nivel': None,
                'timestamp': datetime.now().isoformat()
            }

    def batch_predict(self, customers_list: list) -> list:
        """Predice scores para múltiples clientes"""
        results = []
        for i, customer in enumerate(customers_list):
            result = self.predict_score(customer)
            result['customer_id'] = i + 1
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Predictor de scoring de clientes")
    parser.add_argument("--model", default="./models/scoring_model_v2.joblib",
                        help="Ruta del modelo .joblib")
    parser.add_argument("--json-file", help="Archivo JSON con datos del cliente para predecir")
    parser.add_argument("--interactive", action="store_true",
                        help="Modo interactivo para ingresar datos")
    args = parser.parse_args()

    try:
        predictor = CustomerScoringPredictor(args.model)
    except Exception as e:
        logging.error(f"Error inicializando predictor: {e}")
        sys.exit(1)

    if args.json_file:
        logging.info(f"Cargando datos desde: {args.json_file}")
        with open(args.json_file, 'r') as f:
            customer_data = json.load(f)
        result = predictor.predict_score(customer_data)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    elif args.interactive:
        logging.info("Modo interactivo activado")
        customer_data = {}
        fields = [
            ('edad', int, 'Edad'),
            ('ingresos_mensuales', float, 'Ingresos ($)'),
            ('cantidad_polizas', int, 'Cantidad de pólizas'),
            ('ratio_cuotas_cumplidas', float, 'Ratio de cuotas cumplidas (0.0-1.0)'),
            ('cantidad_siniestros', int, 'Cantidad de siniestros'),
            ('antiguedad_cliente', int, 'Antigüedad (días)'),
            ('hijos', int, 'Cantidad de hijos')
        ]
        for field, cast, desc in fields:
            try:
                value = input(f"{desc}: ")
                customer_data[field] = cast(value) if value else 0
            except ValueError:
                customer_data[field] = 0

        customer_data['ocupacion'] = input("Ocupación: ") or 'empleado'
        customer_data['preferencia_contacto'] = input("Preferencia contacto (email/telefono): ") or 'email'
        customer_data['genero'] = input("Género (M/F): ") or 'M'

        defaults = {
            'suma_asegurada_total': 500000,
            'prima_promedio': 50000,
            'cantidad_pagos': 12,
            'cuotas_cumplidas': int(customer_data.get('ratio_cuotas_cumplidas', 0.8) * 12),
            'cuotas_impagas': 1,
            'pagos_parciales': 1,
            'monto_promedio_pago': 4000,
            'monto_promedio_siniestro': 0,
            'dias_desde_ultimo_siniestro': 9999,
            'tiene_auto': 1,
            'tiene_hogar': 0,
            'tiene_vida': 0,
            'tiene_salud': 0,
            'tiene_accidentes': 0,
            'tiene_mascotas': 0,
            'tiene_riesgos': 0
        }
        customer_data.update(defaults)

        result = predictor.predict_score(customer_data)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    else:
        logging.info("Ejecutando predicción de ejemplo")
        example = {
            "edad": 35, "genero": "M", "preferencia_contacto": "email",
            "antiguedad_cliente": 730, "hijos": 2, "ocupacion": "empleado",
            "ingresos_mensuales": 500000, "cantidad_polizas": 2,
            "suma_asegurada_total": 800000, "prima_promedio": 45000,
            "cantidad_pagos": 24, "cuotas_cumplidas": 20,
            "cuotas_impagas": 1, "pagos_parciales": 3,
            "ratio_cuotas_cumplidas": 0.83, "monto_promedio_pago": 3750,
            "cantidad_siniestros": 1, "monto_promedio_siniestro": 25000,
            "dias_desde_ultimo_siniestro": 365,
            "tiene_auto": 1, "tiene_hogar": 1, "tiene_vida": 0,
            "tiene_salud": 0, "tiene_accidentes": 0,
            "tiene_mascotas": 0, "tiene_riesgos": 0
        }
        result = predictor.predict_score(example)
        print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
