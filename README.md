# IA Proyecto: Prediccion de Abandono de Clientes (Churn Target)

Este repositorio contiene un entorno de experimentacion modular en Machine Learning para predecir el abandono de clientes (churn) en telecomunicaciones. El pipeline evalua formalmente el impacto del preprocesamiento y escalado de datos comparando modelos lineales (Regresion Logistica) contra modelos basados en arboles (Random Forest).

---

## Estructura del Directorio

El proyecto sigue una organizacion estandar para proyectos de Ciencia de Datos:

```text
IA_Proyecto/
├── data/
│   ├── raw/          # Contiene el dataset original sin modificaciones
│   └── processed/    # Archivos generados (Telco_clean.csv, Telco_limpio.csv, Telco_validate.csv y Telco_validado.csv) 
└──  notebooks/       # Flujo de experimentacion numerado paso a paso
     └──logs/         # Registros de ejecucion del sistema
```
# COMPARACIÓN GLOBAL DE ALGORITMOS
| Metrica | Reg. Logistica | Reg. Log. (Vieja) | Random Forest | RF (Viejo) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 74.0% | 73.5% | 75.3% | 76.0% |
| **Precision** | 50.7% | 50.0% | 52.3% | 53.2% |
| **Recall** | 78.1% | 78.9% | 78.3% | 79.7% |
| **F1-Score** | 61.5% | 61.2% | 62.7% | 63.8% |

## Tecnologías Utilizadas

* **Python 3.12.2**
* **Pandas** & **NumPy** (Manipulación e ingesta)
* **Scikit-Learn** (Particionado, Escalado, Modelado y Métricas)
* **Matplotlib** (Visualización de matrices de confusión)

## Requerimientos
* pandas>=2.0.0
* numpy>=1.24.0
* matplotlib>=3.7.0
* seaborn>=0.12.0
* scikit-learn>=1.3.0
* jupyterlab

## Cómo Ejecutar el Proyecto

1. Clona este repositorio
   ```bash
   git clone [https://github.com/HailAngel/gestion_datos_3.git](https://github.com/HailAngel/gestion_datos_E.git)

## 2. Modos de Ejecución

Elige el método que mejor se adapte a tu entorno de trabajo:

### Opción A: Usando GitHub Codespaces

1. Abre el repositorio en un Codespace.
2. Al iniciar, Codespaces detectará el archivo, construirá el contenedor e instalará las extensiones automáticamente.
3. **Para correr el pipeline:** Abre una terminal integrada en VS Code y ejecuta directamente:
   python IA_Proyecto/notebooks/pipeline.py

### Opción B: Usando Docker Local (Tradicional)
Si prefieres clonar el proyecto en tu máquina local y manejar el ciclo de vida manualmente desde tu terminal, usa estos comandos desde la raíz principal (gestion_datos_3):

1. Construir la Imagen

docker build -t telco-pipeline .

2. Ejecutar el Pipeline de Limpieza automáticamente
Procesa y valida los datos, exportando los archivos .csv finales directamente a tu carpeta local de forma sincronizada:

docker run --rm -v "${PWD}/IA_Proyecto:/app/IA_Proyecto" telco-pipeline

3. Levantar Jupyter Lab para el Entrenamiento
Levanta el servidor interactivo de Jupyter para que puedas entrenar el modelos desde el navegador:

docker run --rm -it -p 8888:8888 -v "${PWD}/IA_Proyecto:/app/IA_Proyecto" telco-pipeline jupyter lab --ip=0.0.0.0 --allow-root --no-browser