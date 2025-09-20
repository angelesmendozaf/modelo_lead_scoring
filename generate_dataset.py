import random
from datetime import date, timedelta
import pandas as pd
import numpy as np
from faker import Faker
import os

# Configuracion inicial
fake = Faker('es_AR')
random.seed(42)
np.random.seed(42)

N_PERSONAS = 1000
HOY = date.today()

os.makedirs('./csv', exist_ok=True)

print("- COMENZANDO A GENERAR DATASET PARA CROSS-SELLING -")

# Distribuciones base por ocupacion (valores en ARS mensuales 2025, calibrar segun fuentes)
# cada tup: (p10, median, p90)
ingresos_por_ocupacion = {
    'profesional': (400000, 1000000, 3500000),
    'empleado': (150000, 450000, 1200000),
    'autonomo': (120000, 420000, 1800000),
    'comerciante': (120000, 350000, 1500000),
    'jubilado': (80000, 300000, 800000),
    'estudiante': (20000, 60000, 200000),
    'desempleado': (0, 30000, 120000)
}

provincias = [
    'Buenos Aires', 'CABA', 'Cordoba', 'Santa Fe', 'Mendoza',
    'Tucuman', 'Entre Rios', 'Corrientes', 'Neuquen', 'Salta',
    'Chubut', 'Misiones', 'La Pampa', 'San Luis', 'Rio Negro'
]
# Probabilidad de zona urbana por provincia (simplificada)
prob_urbano_por_prov = {p: (0.9 if p in ['CABA','Buenos Aires'] else 0.7) for p in provincias}

# niveles educativos
niveles_edu = ['sin_estudios', 'primario', 'secundario', 'terciario', 'universitario']

print("Generando datos base...")

personas_base = []
for i in range(N_PERSONAS):
    edad = random.randint(18, 85)
    nacimiento = HOY.replace(year=HOY.year - edad) - timedelta(days=random.randint(0, 365))
    fecha_registro = HOY - timedelta(days=random.randint(30, 365*6))  # hasta 6 anos
    provincia = random.choice(provincias)
    zona_urbana = 1 if random.random() < prob_urbano_por_prov[provincia] else 0

    # ocupacion con probabilidades dependientes de edad
    if edad >= 65:
        ocupaciones = ['jubilado', 'empleado', 'autonomo']
        pesos = [0.7, 0.2, 0.1]
    elif edad < 25:
        ocupaciones = ['estudiante','empleado','autonomo','desempleado']
        pesos = [0.45, 0.3, 0.15, 0.1]
    else:
        ocupaciones = ['empleado', 'autonomo', 'comerciante', 'profesional', 'desempleado']
        pesos = [0.45, 0.15, 0.12, 0.2, 0.08]

    ocupacion = random.choices(ocupaciones, weights=pesos)[0]

    # nivel educativo correlacionado con ocupacion
    if ocupacion == 'profesional':
        nivel_edu = random.choices(niveles_edu, weights=[0,0.05,0.15,0.1,0.7])[0]
    elif ocupacion == 'jubilado':
        nivel_edu = random.choices(niveles_edu, weights=[0.02,0.2,0.5,0.15,0.13])[0]
    elif ocupacion == 'estudiante':
        nivel_edu = random.choices(niveles_edu, weights=[0,0.1,0.5,0.3,0.1])[0]
    else:
        nivel_edu = random.choices(niveles_edu, weights=[0.01,0.15,0.5,0.2,0.14])[0]

    # generar ingresos segun percentiles y algo de ruido macro (inflacion/variacion 2025)
    p10, med, p90 = ingresos_por_ocupacion[ocupacion]
    # mu,y sigma para lognormal que aprox mediana y dispersion
    # usamos transformacion simple: sample entre p10..p90 con beta dist y ruido
    u = random.betavariate(2,2)  # sesgada al centro
    ingresos = p10 + u * (p90 - p10)
    # ajuste por educacion y zona urbana
    if nivel_edu == 'universitario':
        ingresos *= random.uniform(1.05, 1.35)
    if zona_urbana:
        ingresos *= random.uniform(1.00, 1.15)
    # jubilados suelen tener ingresos mas estables pero mas bajos
    if ocupacion == 'jubilado':
        ingresos *= random.uniform(0.9, 1.1)
    ingresos = max(0, ingresos + random.gauss(0, ingresos * 0.15))

    # hijos, tamano de hogar
    tamano_hogar = np.random.choice([1,2,3,4,5], p=[0.2,0.25,0.25,0.2,0.1])
    # probabilidad de tener hijos creciente con edad
    hijos = 0
    if edad > 22:
        hijos = np.random.choice([0,1,2,3], p=[0.35,0.3,0.25,0.1])
    else:
        hijos = 0

    # vivienda propia correlacionada con edad e ingresos
    if edad > 35 and ingresos > med * 0.8 and random.random() < 0.6:
        vivienda_propia = 1
    else:
        vivienda_propia = 1 if random.random() < 0.35 else 0

    # posee_auto correlacionada con ingresos y zona urbana
    posee_auto = 1 if (ingresos > med * 0.8 and random.random() < (0.6 if zona_urbana else 0.4)) else 0

    # preferencia de contacto
    preferencia_contacto = random.choices(['email', 'telefono', 'whatsapp'], [0.45, 0.25, 0.3])[0]

    # ahorro y deuda: fracciones del ingreso
    ahorro_frac = max(0, random.gauss(0.08, 0.06))  # promedio 8% del ingreso
    deuda_frac = max(0, random.gauss(0.12, 0.1))   # promedio 12% del ingreso

    ahorro_mensual = ingresos * ahorro_frac
    deuda_tarjeta = ingresos * deuda_frac

    personas_base.append({
        'persona_id': i + 1,
        'edad': edad,
        'genero': random.choices(['M','F'], [0.5,0.5])[0],
        'fecha_nacimiento': nacimiento,
        'fecha_registro': fecha_registro,
        'hijos': hijos,
        'tamano_hogar': tamano_hogar,
        'ocupacion': ocupacion,
        'nivel_educativo': nivel_edu,
        'provincia': provincia,
        'zona_urbana': zona_urbana,
        'vivienda_propia': vivienda_propia,
        'posee_auto': posee_auto,
        'ingresos_mensuales': round(ingresos, 2),
        'preferencia_contacto': preferencia_contacto,
        'ahorro_mensual': round(ahorro_mensual, 2),
        'deuda_tarjeta': round(deuda_tarjeta, 2)
    })

# Simulacion de polizas (mas realista)
print("Calculando metricas de polizas...")
tipos_seguros = ['Auto', 'Hogar', 'Vida', 'Salud', 'Accidentes', 'Mascotas', 'Riesgos']
primas_base = {'Auto': 70000, 'Hogar': 45000, 'Vida': 27500, 'Salud': 47500, 'Accidentes': 15000, 'Mascotas': 12000, 'Riesgos': 32000}

polizas_simuladas = []
for persona in personas_base:
    ing = persona['ingresos_mensuales']
    # probabilidad base de tener al menos una poliza depende de ingresos y edad
    prob_tener = np.clip(0.1 + (ing / 1000000) + (persona['edad'] / 200), 0, 0.9)
    if random.random() < prob_tener:
        # numero de polizas correlacionado con ingresos
        if ing > 1500000:
            n_polizas = random.choices([1,2,3,4], weights=[0.15,0.35,0.35,0.15])[0]
        elif ing > 500000:
            n_polizas = random.choices([1,2,3], weights=[0.4,0.45,0.15])[0]
        else:
            n_polizas = random.choices([1,0,2], weights=[0.5,0.25,0.25])[0]
    else:
        n_polizas = 0

    seguros_persona = []
    if n_polizas > 0:
        # Auto
        if persona['posee_auto'] and random.random() < 0.8:
            seguros_persona.append('Auto')
        elif ing > 80000 and random.random() < 0.4:
            seguros_persona.append('Auto')

        # Hogar
        if persona['vivienda_propia'] and random.random() < 0.5 and len(seguros_persona) < n_polizas:
            seguros_persona.append('Hogar')

        # Salud
        if ing > 60000 and random.random() < 0.35 and len(seguros_persona) < n_polizas:
            seguros_persona.append('Salud')

        # Vida
        if persona['hijos'] > 0 and persona['edad'] > 25 and random.random() < 0.45 and len(seguros_persona) < n_polizas:
            seguros_persona.append('Vida')

        # completar con probabilidades para otros
        otros_productos = [t for t in tipos_seguros if t not in seguros_persona]
        while len(seguros_persona) < n_polizas and otros_productos:
            choice_idx = random.randint(0, len(otros_productos)-1)
            seguros_persona.append(otros_productos.pop(choice_idx))

    for tipo in seguros_persona:
        factor_ingresos = min(2.5, max(0.5, persona['ingresos_mensuales'] / 200000))
        prima = primas_base[tipo] * factor_ingresos * random.uniform(0.75, 1.25)
        suma_asegurada = prima * random.uniform(12, 48)
        fecha_inicio = persona['fecha_registro'] + timedelta(days=random.randint(0, 400))
        polizas_simuladas.append({
            'persona_id': persona['persona_id'],
            'tipo_seguro': tipo,
            'prima': round(prima, 2),
            'suma_asegurada': round(suma_asegurada, 2),
            'fecha_inicio': fecha_inicio
        })

# Simulacion pagos mensuales de polizas
print("Calculando metricas de pagos...")
pagos_por_persona = {}
for poliza in polizas_simuladas:
    pid = poliza['persona_id']
    persona = next(p for p in personas_base if p['persona_id'] == pid)
    dias_poliza = (HOY - poliza['fecha_inicio']).days
    cuotas_esperadas = min(24, max(1, dias_poliza // 30))  # permitir hasta 2 anos de historico en algunos casos

    # prob punctualidad dependiente de ahorro_frac y deuda_frac y antiguedad
    ahorro = persona['ahorro_mensual']
    deuda = persona['deuda_tarjeta']
    ingreso = persona['ingresos_mensuales']
    ahorro_frac = ahorro / ingreso if ingreso > 0 else 0
    deuda_frac = deuda / ingreso if ingreso > 0 else 1
    antiguedad = (HOY - persona['fecha_registro']).days
    base_prob = 0.35 + np.clip(ahorro_frac * 3, 0, 0.4) - np.clip(deuda_frac * 0.6, 0, 0.3) + np.clip(antiguedad / 3650, 0, 0.2)
    prob_puntual = float(np.clip(base_prob, 0.1, 0.98))

    cuota_mensual = poliza['prima'] / 12

    cuotas_al_dia = cuotas_atrasadas = cuotas_impagas = 0
    monto_total_pagado = 0.0

    for m in range(int(cuotas_esperadas)):
        r = random.random()
        if r < prob_puntual:
            cuotas_al_dia += 1
            monto_total_pagado += cuota_mensual * random.uniform(0.98, 1.02)
        elif r < prob_puntual + 0.22:
            cuotas_atrasadas += 1
            monto_total_pagado += cuota_mensual * random.uniform(0.3, 0.9)
        else:
            cuotas_impagas += 1

    if pid not in pagos_por_persona:
        pagos_por_persona[pid] = {'total_cuotas':0, 'total_al_dia':0, 'total_atrasadas':0, 'total_impagas':0, 'monto_total':0.0}
    pagos_por_persona[pid]['total_cuotas'] += cuotas_esperadas
    pagos_por_persona[pid]['total_al_dia'] += cuotas_al_dia
    pagos_por_persona[pid]['total_atrasadas'] += cuotas_atrasadas
    pagos_por_persona[pid]['total_impagas'] += cuotas_impagas
    pagos_por_persona[pid]['monto_total'] += monto_total_pagado

# Simulacion de siniestros
print("Calculando metricas de siniestros...")
siniestros_por_persona = {}
prob_siniestro_tipo = {'Auto': 0.35, 'Hogar': 0.18, 'Vida': 0.03, 'Salud': 0.5, 'Accidentes': 0.25, 'Mascotas': 0.18, 'Riesgos': 0.22}

for poliza in polizas_simuladas:
    pid = poliza['persona_id']
    if random.random() < prob_siniestro_tipo.get(poliza['tipo_seguro'], 0.15):
        n_siniestros = random.choices([1,2,3], weights=[0.75,0.2,0.05])[0]
        for _ in range(n_siniestros):
            if poliza['tipo_seguro'] == 'Vida':
                monto = poliza['suma_asegurada']
            else:
                monto = poliza['suma_asegurada'] * random.uniform(0.05, 0.8)
            dias_max = max(1, min(365, (HOY - poliza['fecha_inicio']).days))
            fecha_siniestro = poliza['fecha_inicio'] + timedelta(days=random.randint(1, dias_max))
            if pid not in siniestros_por_persona:
                siniestros_por_persona[pid] = {'cantidad':0, 'monto_total':0.0, 'fecha_ultimo':None}
            siniestros_por_persona[pid]['cantidad'] += 1
            siniestros_por_persona[pid]['monto_total'] += monto
            if siniestros_por_persona[pid]['fecha_ultimo'] is None or fecha_siniestro > siniestros_por_persona[pid]['fecha_ultimo']:
                siniestros_por_persona[pid]['fecha_ultimo'] = fecha_siniestro

# Construccion dataset final
print("Construyendo dataset ML final...")

dataset_ml = []
for persona in personas_base:
    pid = persona['persona_id']
    edad = persona['edad']
    genero = 1 if persona['genero'] == 'M' else 0
    antiguedad_cliente = (HOY - persona['fecha_registro']).days
    ingresos = persona['ingresos_mensuales']

    polizas_persona = [p for p in polizas_simuladas if p['persona_id'] == pid]
    cantidad_polizas = len(polizas_persona)
    suma_asegurada_total = round(sum(p['suma_asegurada'] for p in polizas_persona),2) if cantidad_polizas>0 else 0.0
    prima_promedio = round((sum(p['prima'] for p in polizas_persona)/cantidad_polizas),2) if cantidad_polizas>0 else 0.0

    if pid in pagos_por_persona:
        pago_data = pagos_por_persona[pid]
        cantidad_pagos = pago_data['total_cuotas']
        cuotas_cumplidas = pago_data['total_al_dia']
        cuotas_impagas = pago_data['total_impagas']
        pagos_parciales = pago_data['total_atrasadas']
        ratio_cuotas_cumplidas = cuotas_cumplidas / cantidad_pagos if cantidad_pagos>0 else 0.0
        monto_promedio_pago = pago_data['monto_total'] / cantidad_pagos if cantidad_pagos>0 else 0.0
    else:
        cantidad_pagos = cuotas_cumplidas = cuotas_impagas = pagos_parciales = 0
        ratio_cuotas_cumplidas = 0.0
        monto_promedio_pago = 0.0

    if pid in siniestros_por_persona:
        s = siniestros_por_persona[pid]
        cantidad_siniestros = s['cantidad']
        monto_promedio_siniestro = s['monto_total'] / s['cantidad']
        dias_desde_ultimo_siniestro = (HOY - s['fecha_ultimo']).days
    else:
        cantidad_siniestros = 0
        monto_promedio_siniestro = 0.0
        dias_desde_ultimo_siniestro = 9999

    tipos_actuales = {p['tipo_seguro'] for p in polizas_persona}
    tiene_auto = 1 if 'Auto' in tipos_actuales else 0
    tiene_hogar = 1 if 'Hogar' in tipos_actuales else 0
    tiene_vida = 1 if 'Vida' in tipos_actuales else 0
    tiene_salud = 1 if 'Salud' in tipos_actuales else 0
    tiene_accidentes = 1 if 'Accidentes' in tipos_actuales else 0
    tiene_mascotas = 1 if 'Mascotas' in tipos_actuales else 0
    tiene_riesgos = 1 if 'Riesgos' in tipos_actuales else 0

    # logica cross-sell (mas probabilistica)
    candidatos = []
    if not tiene_auto and persona['posee_auto'] == 1 and ingresos > 400000 and edad < 75:
        candidatos.append('Auto')
    if not tiene_hogar and (persona['vivienda_propia'] == 1 or ingresos > 600000) and edad < 80:
        candidatos.append('Hogar')
    if not tiene_vida and edad > 25 and persona['hijos'] > 0 and edad < 65:
        candidatos.append('Vida')
    if not tiene_salud and ingresos > 350000 and edad < 80:
        candidatos.append('Salud')
    if not tiene_accidentes and persona['ocupacion'] in ['autonomo','comerciante','profesional']:
        candidatos.append('Accidentes')
    if not tiene_mascotas and random.random() < 0.20:
        candidatos.append('Mascotas')
    if not tiene_riesgos and persona['ocupacion'] in ['autonomo','comerciante','profesional'] and ingresos > 800000:
        candidatos.append('Riesgos')

    if candidatos:
        producto_objetivo = random.choice(candidatos)
        # score comportamiento
        score = (
            ratio_cuotas_cumplidas * 0.32 +
            np.clip(ingresos / 1500000, 0, 1) * 0.28 +
            max(0, 1 - cantidad_siniestros / 5) * 0.18 +
            np.clip(antiguedad_cliente / (365*3), 0, 1) * 0.12 +
            np.clip(cantidad_polizas / 4, 0, 1) * 0.1
        )
        # ajustes
        if producto_objetivo == 'Auto':
            score *= 0.95
        if producto_objetivo == 'Vida' and persona['hijos'] > 1:
            score *= 1.25
        if producto_objetivo == 'Salud' and edad > 50:
            score *= 1.15

        # ajustar por riesgo crediticio simple
        riesgo_crediticio = 1 if (persona['deuda_tarjeta'] / ingresos if ingresos>0 else 1) > 0.6 else 0
        if riesgo_crediticio:
            score *= 0.85

        prob_compra = float(np.clip(score * random.uniform(0.65, 1.35), 0, 1))
        adquirio = 1 if random.random() < prob_compra else 0
    else:
        producto_objetivo = 'Ninguno'
        adquirio = 0

    # metricas extras
    deuda_frac_ingreso = persona['deuda_tarjeta'] / ingresos if ingresos>0 else 0
    ahorro_frac_ingreso = persona['ahorro_mensual'] / ingresos if ingresos>0 else 0
    costo_mensual_total_seguro = round((prima_promedio/12) * cantidad_polizas, 2)

    dataset_ml.append({
        'persona_id': pid,
        'edad': edad,
        'genero': genero,
        'provincia': persona['provincia'],
        'zona_urbana': persona['zona_urbana'],
        'nivel_educativo': persona['nivel_educativo'],
        'preferencia_contacto': persona['preferencia_contacto'],
        'antiguedad_cliente': antiguedad_cliente,
        'hijos': persona['hijos'],
        'tamano_hogar': persona['tamano_hogar'],
        'ocupacion': persona['ocupacion'],
        'vivienda_propia': persona['vivienda_propia'],
        'posee_auto': persona['posee_auto'],
        'ingresos_mensuales': round(ingresos,2),
        'ahorro_mensual': persona['ahorro_mensual'],
        'deuda_tarjeta': persona['deuda_tarjeta'],
        'cantidad_polizas': cantidad_polizas,
        'suma_asegurada_total': suma_asegurada_total,
        'prima_promedio': prima_promedio,
        'cantidad_pagos': cantidad_pagos,
        'cuotas_cumplidas': cuotas_cumplidas,
        'cuotas_impagas': cuotas_impagas,
        'pagos_parciales': pagos_parciales,
        'ratio_cuotas_cumplidas': round(ratio_cuotas_cumplidas,3),
        'monto_promedio_pago': round(monto_promedio_pago,2),
        'cantidad_siniestros': cantidad_siniestros,
        'monto_promedio_siniestro': round(monto_promedio_siniestro,2),
        'dias_desde_ultimo_siniestro': dias_desde_ultimo_siniestro,
        'tiene_auto': tiene_auto,
        'tiene_hogar': tiene_hogar,
        'tiene_vida': tiene_vida,
        'tiene_salud': tiene_salud,
        'tiene_accidentes': tiene_accidentes,
        'tiene_mascotas': tiene_mascotas,
        'tiene_riesgos': tiene_riesgos,
        'costo_mensual_total_seguro': costo_mensual_total_seguro,
        'deuda_frac_ingreso': round(deuda_frac_ingreso,3),
        'ahorro_frac_ingreso': round(ahorro_frac_ingreso,3),
        'producto_objetivo': producto_objetivo,
        'adquirio': adquirio
    })

# guardar dataset
df_final = pd.DataFrame(dataset_ml)
df_final.to_csv('./csv/dataset_cross_selling_completo.csv', index=False)

# analisis rapido
print("\\n=== ANALISIS DEL DATASET GENERADO ===")
print(f"Total de registros: {len(df_final)}")
print(f"Tasa de conversion general: {df_final['adquirio'].mean():.3f}")
print("Distribucion por producto objetivo:")
print(df_final['producto_objetivo'].value_counts().head(10))
print("Resumen ingresos (ARS):")
print(df_final['ingresos_mensuales'].describe().to_string())
print("\\nDataset listo en: ./csv/dataset_cross_selling_completo.csv")
