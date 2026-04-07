# Superconductor Tc Regression: Advanced Materials Discovery via Ensembles

## Project Overview
Este proyecto aborda el problema de regresión para la predicción de la Temperatura Crítica (Tc) en materiales superconductores. El objetivo es modelar la relación no lineal entre las propiedades físicas elementales derivadas de la fórmula química y su comportamiento termodinámico, optimizando la precisión predictiva en un dataset con alta variabilidad y presencia de valores extremos.

## Dataset and Feature Engineering
Se utilizó el dataset de la NIMS (Japón) con 21,263 muestras. La ingeniería de características se centró en la extracción de 81 descriptores basados en la composición química, incluyendo momentos estadísticos de propiedades atómicas (masa, radio, energía de ionización, etc.).

### Data Pipeline & Pre-processing
* **Outlier Management:** Implementación de Winsorización para mitigar el impacto de colas pesadas en la distribución de Tc sin pérdida sustancial de varianza.
* **Dimensionality Reduction:** Aplicación de Variance Threshold para eliminar features constantes y análisis de colinealidad (Pearson > 0.7) para reducir la redundancia, mejorando la convergencia del modelo.
* **Scaling:** Estandarización robusta para algoritmos basados en distancias (SVR) y regularización lineal.

## Model Architecture and Training Strategy
Se evaluó un espectro de modelos con diferentes capacidades de hipótesis para encontrar el equilibrio óptimo entre sesgo y varianza:

1.  **Baseline:** Ridge Regression con expansión polinomial para capturar interacciones de segundo orden.
2.  **Kernel Methods:** Support Vector Regression (SVR) con kernel RBF para mapeo a espacios de alta dimensionalidad.
3.  **Boosting:** XGBoost optimizado mediante RandomizedSearchCV para manejo eficiente de estructuras de datos tabulares.
4.  **Bagging:** Random Forest Regressor para reducción de varianza mediante promediado de árboles profundos.

### Meta-Learning (Stacking)
Para maximizar el rendimiento, se implementó un **Stacking Regressor**. Se utilizaron XGBoost, SVR y Random Forest como base learners, y una regresión lineal como meta-modelo. Esta arquitectura permite corregir los errores sistemáticos de los modelos individuales, logrando una mejor generalización en el set de prueba.

## Performance Metrics
El modelo final (Stacking) presenta un rendimiento superior, especialmente en la capacidad de capturar picos de Tc en materiales de alta temperatura.

| Modelo | R^2 Score | RMSE |
| :--- | :---: | :---: |
| Stacking Ensemble | 0.93 | 9.21 K |
| Random Forest | 0.93 | 9.30 K |
| XGBoost | 0.93 | 9.31 K |
| Polynomial Ridge | 0.87 | 14.21 K |
| SVR | 0.84 | 13.79 K |

## Tech Stack
* Python 3 (Scikit-Learn, Pandas, NumPy)
* XGBoost Framework
* Google Colab para cómputo distribuido
* Matplotlib/Seaborn para análisis exploratorio y de residuos

## Future Work
* Implementación de Nested Cross-Validation para una estimación del error de generalización más insesgada.
* Exploración de arquitecturas de Deep Learning (MLP) con capas de Dropout para regularización adicional.
* Uso de SHAP (SHapley Additive exPlanations) para interpretabilidad del modelo a nivel de feature contribution.
