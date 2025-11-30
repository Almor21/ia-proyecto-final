#include "knn.h"

// Función para cargar dataset desde CSV
int load_dataset(const char *filename, struct Dataset *data) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: No se pudo abrir el archivo %s\n", filename);
        return 0;
    }

    // Primera pasada: contar líneas y determinar número de características
    char line[4096];
    int line_count = 0;
    int n_features = 0;
    
    if (fgets(line, sizeof(line), file) != NULL) {
        // Contar comas en la primera línea (n_features = número de comas)
        char *ptr = line;
        while (*ptr) {
            if (*ptr == ',') n_features++;
            ptr++;
        }
        // La última columna es la etiqueta, así que n_features es correcto
        line_count++;
    }
    
    while (fgets(line, sizeof(line), file) != NULL) {
        line_count++;
    }
    
    rewind(file);
    
    // Asignar memoria
    data->n_samples = line_count;
    data->n_features = n_features;
    data->samples = (struct Sample *)malloc(data->n_samples * sizeof(struct Sample));
    
    if (data->samples == NULL) {
        printf("Error: No se pudo asignar memoria\n");
        fclose(file);
        return 0;
    }
    
    // Segunda pasada: leer datos
    int i = 0;
    while (fgets(line, sizeof(line), file) != NULL && i < data->n_samples) {
        // Asignar memoria para features
        data->samples[i].features = (double *)malloc(data->n_features * sizeof(double));
        
        if (data->samples[i].features == NULL) {
            printf("Error: No se pudo asignar memoria para features\n");
            fclose(file);
            return 0;
        }
        
        // Parsear línea CSV
        char *token = strtok(line, ",");
        int j = 0;
        
        while (token != NULL && j < data->n_features) {
            data->samples[i].features[j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        
        // Último token es la etiqueta
        if (token != NULL) {
            data->samples[i].label = atoi(token);
        }
        
        i++;
    }
    
    fclose(file);
    return 1;
}

// Liberar memoria del dataset
void free_dataset(struct Dataset *data) {
    if (data->samples != NULL) {
        for (int i = 0; i < data->n_samples; i++) {
            if (data->samples[i].features != NULL) {
                free(data->samples[i].features);
            }
        }
        free(data->samples);
    }
    data->n_samples = 0;
    data->n_features = 0;
}

// Calcula distancia Euclídea
double euclidean_distance(double *x, double *y, int n_features) {
    double sum = 0.0;
    for (int i = 0; i < n_features; i++) {
        double diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Función de comparación para qsort
int compare_distances(const void *a, const void *b) {
    struct Distance *da = (struct Distance *)a;
    struct Distance *db = (struct Distance *)b;
    
    if (da->dist < db->dist) return -1;
    if (da->dist > db->dist) return 1;
    return 0;
}

// Predicción KNN
int knn_predict(struct Dataset *train, double *x_query, int k, int n_features) {
    // Crear arreglo de distancias
    struct Distance *distances = (struct Distance *)malloc(train->n_samples * sizeof(struct Distance));
    
    if (distances == NULL) {
        printf("Error: No se pudo asignar memoria para distancias\n");
        return -1;
    }
    
    // Calcular distancias a todos los ejemplos de entrenamiento
    for (int i = 0; i < train->n_samples; i++) {
        distances[i].index = i;
        distances[i].dist = euclidean_distance(x_query, train->samples[i].features, n_features);
    }
    
    // Ordenar distancias
    qsort(distances, train->n_samples, sizeof(struct Distance), compare_distances);
    
    // Contar votos de los k vecinos más cercanos
    int votes[2] = {0, 0}; // votes[0] = No churn, votes[1] = Yes churn
    
    for (int i = 0; i < k && i < train->n_samples; i++) {
        int label = train->samples[distances[i].index].label;
        if (label == 0 || label == 1) {
            votes[label]++;
        }
    }
    
    // Retornar la clase con más votos
    free(distances);
    return (votes[1] > votes[0]) ? 1 : 0;
}

// Calcula accuracy
double accuracy(struct Dataset *test, struct Dataset *train, int k) {
    int correct = 0;
    for (int i = 0; i < test->n_samples; i++) {
        int predicted = knn_predict(train, test->samples[i].features, k, test->n_features);
        if (predicted == test->samples[i].label) {
            correct++;
        }
    }
    return (double)correct / test->n_samples;
}

// Calcula precision (para clase positiva)
double precision(struct Dataset *test, struct Dataset *train, int k) {
    int true_positives = 0;
    int false_positives = 0;
    
    for (int i = 0; i < test->n_samples; i++) {
        int predicted = knn_predict(train, test->samples[i].features, k, test->n_features);
        int actual = test->samples[i].label;
        
        if (predicted == 1 && actual == 1) {
            true_positives++;
        } else if (predicted == 1 && actual == 0) {
            false_positives++;
        }
    }
    
    if (true_positives + false_positives == 0) return 0.0;
    return (double)true_positives / (true_positives + false_positives);
}

// Calcula recall (sensibilidad)
double recall(struct Dataset *test, struct Dataset *train, int k) {
    int true_positives = 0;
    int false_negatives = 0;
    
    for (int i = 0; i < test->n_samples; i++) {
        int predicted = knn_predict(train, test->samples[i].features, k, test->n_features);
        int actual = test->samples[i].label;
        
        if (predicted == 1 && actual == 1) {
            true_positives++;
        } else if (predicted == 0 && actual == 1) {
            false_negatives++;
        }
    }
    
    if (true_positives + false_negatives == 0) return 0.0;
    return (double)true_positives / (true_positives + false_negatives);
}

// Calcula F1-score
double f1_score(struct Dataset *test, struct Dataset *train, int k) {
    double prec = precision(test, train, k);
    double rec = recall(test, train, k);
    
    if (prec + rec == 0.0) return 0.0;
    return 2.0 * (prec * rec) / (prec + rec);
}

