# Valuador de Carteras de Crédito con Random Forest

## Descripción del Problema
En la industria de adquisición de deuda, valorar incorrectamente una cartera de créditos vencidos (NPLs) puede resultar en pérdidas millonarias. Actualmente, la valuación se realiza mediante promedios históricos simples, ignorando la calidad individual de los deudores.

## Solución Propuesta
Hemos desarrollado un **Modelo Híbrido de Dos Etapas (Hurdle Model)** que predice el Valor Esperado de Recuperación (ROI) para cada cuenta individual.

* **Etapa 1:** Un Clasificador (Random Forest) estima la *Probabilidad de Pago*.
* **Etapa 2:** Un Regresor estima el *Monto Recuperable* condicionado al pago.

## Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Ingeniería de Datos:** Pandas, NumPy (Log Transforms, Clipping de Outliers).
* **Modelado:** Scikit-Learn (Random Forest, validación cruzada).
* **Validación Estadística:** SciPy (Mann-Whitney U Test).
* **Despliegue:** Streamlit (WebApp interactiva).

## Cómo Ejecutar el Proyecto

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt