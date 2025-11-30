#ifndef KNN_H
#define KNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Estructura para representar una muestra (fila del dataset)
struct Sample {
    double *features;  // vector de características (tamaño n_features)
    int label;         // etiqueta: 0 = No churn, 1 = Yes churn
};

// Estructura para representar un dataset completo
struct Dataset {
    struct Sample *samples;  // arreglo de muestras
    int n_samples;           // número de filas
    int n_features;          // número de columnas de características
};

// Estructura para almacenar distancias durante la predicción
struct Distance {
    int index;      // índice del sample en el dataset
    double dist;    // distancia al punto de consulta
};

// Función para cargar dataset desde archivo CSV
// Formato esperado: n_features columnas numéricas, última columna es la etiqueta (0 o 1)
int load_dataset(const char *filename, struct Dataset *data);

// Función para liberar memoria del dataset
void free_dataset(struct Dataset *data);

// Calcula la distancia Euclídea entre dos vectores
double euclidean_distance(double *x, double *y, int n_features);

// Función de comparación para qsort (ordenar distancias)
int compare_distances(const void *a, const void *b);

// Predice la clase de un ejemplo usando KNN
int knn_predict(struct Dataset *train, double *x_query, int k, int n_features);

// Calcula la exactitud (accuracy) del modelo sobre un conjunto de prueba
double accuracy(struct Dataset *test, struct Dataset *train, int k);

// Calcula la precisión (precision) para la clase positiva
double precision(struct Dataset *test, struct Dataset *train, int k);

// Calcula el recall (sensibilidad) para la clase positiva
double recall(struct Dataset *test, struct Dataset *train, int k);

// Calcula el F1-score
double f1_score(struct Dataset *test, struct Dataset *train, int k);

#endif // KNN_H

