"""
Carga el modelo .joblib, arma el DataFrame con las features de un lead y
devuelve: score 0-100, banda (bajo/medio/alto), apto (score>70) y producto recomendado.
"""

import argparse
import json
import joblib
import pandas as pd


def cargar_modelo(ruta_modelo: str):
    bundle = joblib.load(ruta_modelo)
    pipeline = bundle.get("pipeline") or bundle.get("model")
    feature_names = bundle["feature_names"]
    mapping = bundle["product_mapping"]
    return pipeline, feature_names, mapping


def banda_por_score(score: float) -> str:
    if score < 41:
        return "No recomendado para cross-selling"
    elif score <= 70:
        return "Recomendación moderada"
    else:
        return "Recomendación fuerte"


def predecir_lead(pipeline, feature_names, mapping, lead_data: dict):
    # Ordenar columnas exactamente como el modelo espera
    df = pd.DataFrame([lead_data], columns=feature_names)
    prob = pipeline.predict_proba(df)[0, 1]
    score = float(prob * 100.0)
    banda = banda_por_score(score)
    apto = score > 70.0
    prod_code = int(lead_data["producto_objetivo"])
    producto = mapping.get(prod_code, f"Código {prod_code}")
    return {"score": score, "banda": banda, "apto": apto, "producto": producto}


def main(args):
    pipeline, feature_names, mapping = cargar_modelo(args.model)

    # Si viene un JSON con el lead, lo uso. Si no, uso el ejemplo integrado.
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            lead = json.load(f)
    else:
        # Ejemplo integrado (alineado con las columnas del dataset)
        lead = {
           "edad": 45,
  "genero": 1,
  "preferencia_contacto": 2,
  "antiguedad_cliente": 7.0,
  "hijos": 2,
  "ocupacion": 1,
  "ingresos_mensuales": 350000,
  "cantidad_polizas": 2,
  "suma_asegurada_total": 950000,
  "prima_promedio": 12000,
  "cantidad_pagos": 48,
  "cuotas_cumplidas": 46,
  "cuotas_impagas": 0,
  "pagos_parciales": 2,
  "ratio_cuotas_cumplidas": 0.96,
  "monto_promedio_pago": 4500,
  "cantidad_siniestros": 0,
  "monto_promedio_siniestro": 0,
  "dias_desde_ultimo_siniestro": -1,
  "tiene_auto": 1,
  "tiene_hogar": 1,
  "tiene_vida": 0,
  "tiene_salud": 1,
  "tiene_accidentes": 0,
  "tiene_mascotas": 0,
  "tiene_riesgos": 0,
  "producto_objetivo": 2
        }

    res = predecir_lead(pipeline, feature_names, mapping, lead)
    print(f"Score: {res['score']:.2f}")
    print(f"Banda: {res['banda']}")
    print(f"¿Apto?: {'Sí' if res['apto'] else 'No'}")
    print(f"Producto recomendado: {res['producto']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia de scoring/cross-selling")
    # Defaults para ejecutar sin parámetros
    parser.add_argument("--model", default="model_test_v3.joblib",
                        help="Ruta al modelo .joblib (default: model_test_v3.joblib)")
    parser.add_argument("--json", default=None,
                        help="(opcional) Ruta a un JSON con las features del lead")
    args = parser.parse_args()
    main(args)
