import streamlit as st
import pandas as pd
import numpy as np
import joblib

#  CONFIGURACIÓN 
st.set_page_config(page_title="Valuador GRUPO 4", page_icon="📈")
st.title(" Valuador de Deuda - GRUPO 4")

#  CARGA 
@st.cache_resource
def load_artifacts():
    clf = joblib.load('clasificador_pago.pkl')
    reg = joblib.load('regresor_monto.pkl')
    cols = joblib.load('columnas_modelo.pkl')
    return clf, reg, cols

try:
    clf, reg, model_cols = load_artifacts()
    st.success("✅ Sistema cargado. Ingeniería de variables activa.")
except Exception as e:
    st.error(f"❌ ERROR CRÍTICO CARGANDO MODELOS: {e}")
    
    # --- DIAGNÓSTICO AUTOMÁTICO ---
    import os
    st.divider()
    st.warning("🕵️‍♂️ REPORTE DEL DETECTIVE DE DATOS:")
    
    # 1. ¿Dónde estoy parado?
    ruta_actual = os.getcwd()
    st.write(f"📍 **Estoy ejecutando desde:** `{ruta_actual}`")
    
    # 2. ¿Existe la carpeta models aquí?
    ruta_models = os.path.join(ruta_actual, 'models')
    if os.path.exists(ruta_models):
        st.success(f"✅ La carpeta `models` SÍ existe en `{ruta_models}`")
        # 3. ¿Qué hay adentro?
        archivos = os.listdir(ruta_models)
        st.info(f"📂 Archivos encontrados: {archivos}")
    else:
        st.error(f"⛔ La carpeta `models` NO existe en `{ruta_actual}`.")
        st.write("💡 **Solución:** Asegúrate de que estás ejecutando el comando `streamlit run` DESDE la carpeta que contiene a `models`.")
    
    st.stop()

#  INPUTS 
st.sidebar.header("Datos del Cliente")
saldo_raw = st.sidebar.number_input("Saldo Total ($)", 1.0, 100000.0, 5000.0) # Input en $ normal
dias_mora = st.sidebar.number_input("Días Mora", 0, 3000, 180)
antiguedad = st.sidebar.slider("Antigüedad (Meses)", 0, 120, 24)
recencia = st.sidebar.slider("Meses sin pago", 0, 60, 6)
edad = st.sidebar.slider("Edad", 18, 90, 35)
sexo = st.sidebar.selectbox("Sexo", ["M", "F"])
civil = st.sidebar.selectbox("Estado Civil", ["SOLTERO", "CASADO", "DIVORCIADO", "UNION_LIBRE", "OTROS"])

#  PREPROCESAMIENTO EN TIEMPO REAL 
if st.sidebar.button("CALCULAR"):
    # 1. Transformación Logarítmica 
    log_saldo = np.log1p(saldo_raw)
    
    # 2. Clipping de Mora 
    mora_clipped = 720 if dias_mora > 720 else dias_mora
    
    # 3. Construcción del Vector
    input_data = pd.DataFrame(columns=model_cols)
    input_data.loc[0] = 0 # Inicializar en 0
    
    # Llenado de valores numéricos
    input_data['LOG_SALDO'] = log_saldo
    input_data['ANTIGUEDAD_MESES'] = antiguedad
    input_data['EDAD_CLIENTE'] = edad
    input_data['MESES_DESDE_ULTIMO_PAGO'] = recencia
    input_data['DIAS MORA'] = mora_clipped
    
    # Llenado de Dummies
    if f'SEXO_{sexo}' in input_data.columns:
        input_data[f'SEXO_{sexo}'] = 1
    if f'EST_CIVIL_CLEAN_{civil}' in input_data.columns:
        input_data[f'EST_CIVIL_CLEAN_{civil}'] = 1
        
    #  PREDICCIÓN 
    prob = clf.predict_proba(input_data)[0, 1]
    monto = reg.predict(input_data)[0]
    ev = prob * monto
    
    #  RESULTADOS 
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad", f"{prob:.1%}")
    c2.metric("Recuperación Est.", f"${monto:,.2f}")
    c3.metric("Valor Esperado", f"${ev:,.2f}", delta=f"ROI: {(ev/saldo_raw):.1%}")
    
    if ev > saldo_raw * 0.15:
        st.success("OPORTUNIDAD DE COMPRA")
    else:
        st.error("RIESGO ALTO")
