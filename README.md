# Satisfacción de Pasajeros

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-4B8BBE?style=for-the-badge)

Proyecto completo de Machine Learning diseñado para predecir la satisfacción de pasajeros de aerolíneas. Incluye optimización de hiperparámetros, una API activa, Inteligencia Artificial Explicable (SHAP) y dashboards dedicados tanto para monitoreo técnico como para decisiones de negocio.

## Características Principales

*   **Pipeline Predictivo de 2 Fases:** 
    *   *Fase 1 (Pre-Vuelo):* Predice riesgos de insatisfacción anticipadamente usando solo datos demográficos y de la reserva. **Impacto de Negocio:** Permite al personal de tierra intervenir proactivamente antes de que el pasajero aborde (ej: upgrades o pases VIP).
    *   *Fase 2 (Post-Vuelo):* Predicción de alta precisión que incorpora el feedback del servicio a bordo. **Impacto de Negocio:** Permite enviar disculpas automatizadas y personalizadas orientadas a retener al cliente si su experiencia fue deficiente.
*   **IA Explicable (XAI):** Integración con SHAP para proveer los motivos matemáticos e interpretables detrás de cada predicción en tiempo real. Fundamental para estrategias de retención de clientes.
*   **Base MLOps Robusta:** Estricta separación de datos para evitar "data leakage" (fuga de datos). Las variables post-vuelo como "Gate location" fueron auditadas para confirmar que reflejan puntajes de satisfacción del pasajero, no proxys directos del target.
*   **Selección y Tuning de Modelos:** El modelo ganador (Random Forest) fue seleccionado tras comparar su rendimiento empírico contra Regresión Logística y Árboles de Decisión. Su hiperparametrización se realizó mediante `RandomizedSearchCV` seguido de un `GridSearchCV` acotado, balanceando métricas y evitando sobreajuste.
*   **Arquitectura de Microservicios:** Una API REST ultra rápida construida con **FastAPI**.
*   **Frontend Interactivo:** Interfaz en HTML/JS puro y TailwindCSS que ofrece:
    *   *App Predictora:* Inferencia en tiempo real con barras dinámicas de impacto SHAP.
    *   *Dashboard técnico:* Dashboard monitoreando métricas clave como ROC-AUC, F1-Scores y Matrices de Confusión.
    *   *Business Insights (Negocio):* Dashboard ejecutivo que traduce los impactos matemáticos del modelo en estrategias de negocio accionables.

## Rendimiento del Modelo

Los modelos fueron entrenados usando RandomForestClassifier y optimizados con Validación Cruzada (RandomizedSearchCV).

| Fase | Variables Utilizadas | Accuracy | F1-Score | Mejor `max_depth` |
| :--- | :--- | :--- | :--- | :--- |
| **Fase 1** | `Age`, `Gender`, `Customer Type`, `Type of Travel`, `Class`, `Flight Distance` | **78.9%** | 78.5% | 13 (Evita Overfitting) |
| **Fase 2** | Fase 1 + `Seat comfort`, `Departure/Arrival time`, `Food and drink`, `Gate location`, `Inflight wifi`, `Inflight entertainment`, `Online support`, `Ease of Online booking`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Cleanliness`, `Online boarding`, `Arrival Delay` | **95.3%** | 94.5% | 24 (Acotada para optimización) |

## Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/AgustinaHourcade/ML-airline-passenger-satisfaction.git
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Opcional) Entrenar los modelos desde cero:**
   *Nota: Los modelos pre-entrenados `.joblib` ya están incluidos en el repositorio*
   ```bash
   python -m src.data.make_dataset
   python -m src.models.train
   ```

## Uso 

La aplicación ha sido configurada para que la API de FastAPI sirva los archivos estáticos (el frontend) directamente.

1. **Iniciar el Backend de FastAPI:**
   ```bash
   python -m src.api.main
   ```
   *La API iniciará en `http://localhost:8000`.*

2. **Acceder a los Dashboards:**
   Abre un navegador web e ingresa a `http://localhost:8000`. El predictor y los dashboards estarán listos para usarse

## Estructura del Proyecto

```text
├── data/
│   ├── raw/             # Dataset original 
│   └── processed/       # Divisiones procesadas (Train/Test)
├── models/              # Modelos .joblib serializados y metadatos JSON
├── notebooks/           # Jupyter notebooks originales (EDA, prototipos)
├── src/
│   ├── api/             # Aplicación FastAPI y endpoints
│   ├── data/            # Scripts de ingesta y preprocesamiento de datos
│   ├── frontend/        # Dashboards UI en HTML/JS/Tailwind
│   └── models/          # Scripts de entrenamiento y Pipelines de sklearn
├── requirements.txt     # Lista limpia de dependencias
├── Dockerfile           # Receta de contenedor optimizada para la nube
└── README.md
```

## Despliegue (Deployment)
Este repositorio incluye un Dockerfile diseñado específicamente para desplegar la aplicación completa (Backend + Frontend) en **Hugging Face Spaces**.
