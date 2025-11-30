# Proyecto Final de Inteligencia Artificial  
## Predicción de Churn de Clientes de Telecomunicaciones

Este repositorio contiene el proyecto final del curso **Inteligencia Artificial (ELP 8012)** de la Universidad del Norte.

El objetivo es predecir si un cliente de una compañía de telecomunicaciones **abandonará el servicio (churn)** usando:

- Modelos de **aprendizaje automático** en Python (supervisado y no supervisado).
- Una implementación desde cero del algoritmo **K-Nearest Neighbors (KNN)** en lenguaje C, aplicada a un subconjunto numérico del mismo problema.   

---

## ¿Para qué es este proyecto?

El proyecto sirve para:

- Analizar un dataset real de churn de clientes de telecomunicaciones (Telco Customer Churn).
- Explorar, limpiar y transformar los datos.
- Entrenar y evaluar modelos de clasificación en Python.
- Implementar y evaluar el algoritmo KNN en C, calculando métricas como **accuracy, precision, recall y F1-score**.

En conjunto, demuestra el uso práctico de IA con herramientas de alto nivel (Python) y la comprensión algorítmica a bajo nivel (C).

---

## Contenido del repositorio

- `informe_tecnico.ipynb`  
  Notebook de Jupyter con:
  - Análisis exploratorio del dataset.
  - Preprocesamiento y división train/test.
  - Modelos supervisados y no supervisados.
  - Resultados y conclusiones.

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
  Dataset original de churn de clientes de telecomunicaciones (fuente abierta).

- `main.c`  
  Programa principal en C que:
  - Carga los archivos de entrenamiento y prueba.
  - Ejecuta KNN.
  - Calcula y muestra las métricas en consola.

- `knn.c` y `knn.h`  
  Implementación modular del algoritmo **KNN** y funciones auxiliares: carga de dataset, distancia euclídea, predicción y métricas (accuracy, precision, recall, F1).   

- `train_reducido.csv`, `test_reducido.csv`  
  Archivos CSV numéricos generados a partir del dataset original, usados por el programa en C. Por defecto, `main.c` espera estos nombres de archivo.

---

## Cómo reproducir el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/Almor21/ia-proyecto-final.git
cd ia-proyecto-final
````

### 2. Ejecutar el notebook de Python

**Requisitos:**

* Python 3.x
* Paquetes recomendados:

  * `numpy`
  * `pandas`
  * `scikit-learn`
  * `matplotlib`
  * `seaborn`
  * `jupyter`

Instalar dependencias (ejemplo):

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

Luego:

```bash
jupyter notebook
```

1. Abrir `informe_tecnico.ipynb`.
2. Ejecutar las celdas en orden para:

   * Cargar y preprocesar el dataset.
   * Entrenar los modelos.
   * Visualizar las métricas y conclusiones.

---

### 3. Ejecutar la implementación en C (KNN)

#### 3.1. Preparar los archivos CSV

El código en C espera archivos CSV **numéricos**, donde:

* Cada fila es un ejemplo.
* Las primeras columnas son características (`double`).
* La **última columna** es la etiqueta de clase (`0` = No churn, `1` = Churn).

Por defecto, `main.c` usa: `train_reducido.csv` y `test_reducido.csv` en el mismo directorio del ejecutable. 

#### 3.2. Compilar

Desde la carpeta donde estén `main.c`, `knn.c` y `knn.h`:

```bash
gcc main.c knn.c -o knn -lm
```

#### 3.3. Ejecutar

Con los nombres por defecto:

```bash
./knn
```

También se pueden pasar los nombres de los archivos y el valor de `k` por línea de comandos:

```bash
./knn train_reducido.csv test_reducido.csv 5
```

El programa imprimirá en consola:

* Número de muestras y características.
* Métricas: **Accuracy, Precision, Recall, F1-score**.
* Un ejemplo de predicción individual sobre una muestra de prueba. 

---

## Integrantes del equipo

* Yordi Gonzalez
* Jason Estrada
* Edinson Noriega
