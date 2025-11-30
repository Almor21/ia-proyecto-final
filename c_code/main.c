#include "knn.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    printf("========================================\n");
    printf("Implementacion KNN en C para Churn Prediction\n");
    printf("========================================\n\n");
    
    // Parámetros por defecto
    const char *train_file = "train_reducido.csv";
    const char *test_file = "test_reducido.csv";
    int k = 5;
    
    // Permitir argumentos de línea de comandos
    if (argc >= 2) {
        train_file = argv[1];
    }
    if (argc >= 3) {
        test_file = argv[2];
    }
    if (argc >= 4) {
        k = atoi(argv[3]);
    }
    
    printf("Archivo de entrenamiento: %s\n", train_file);
    printf("Archivo de prueba: %s\n", test_file);
    printf("k (numero de vecinos): %d\n\n", k);
    
    // Cargar datasets
    struct Dataset train_data = {NULL, 0, 0};
    struct Dataset test_data = {NULL, 0, 0};
    
    printf("Cargando datos de entrenamiento...\n");
    if (!load_dataset(train_file, &train_data)) {
        printf("Error al cargar datos de entrenamiento\n");
        return 1;
    }
    printf("  - Muestras: %d\n", train_data.n_samples);
    printf("  - Caracteristicas: %d\n\n", train_data.n_features);
    
    printf("Cargando datos de prueba...\n");
    if (!load_dataset(test_file, &test_data)) {
        printf("Error al cargar datos de prueba\n");
        free_dataset(&train_data);
        return 1;
    }
    printf("  - Muestras: %d\n", test_data.n_samples);
    printf("  - Caracteristicas: %d\n\n", test_data.n_features);
    
    // Verificar que ambos datasets tengan el mismo número de características
    if (train_data.n_features != test_data.n_features) {
        printf("Error: Los datasets tienen diferente numero de caracteristicas\n");
        free_dataset(&train_data);
        free_dataset(&test_data);
        return 1;
    }
    
    // Evaluar modelo
    printf("Evaluando modelo KNN (k=%d)...\n", k);
    printf("Esto puede tomar unos momentos...\n\n");
    
    double acc = accuracy(&test_data, &train_data, k);
    double prec = precision(&test_data, &train_data, k);
    double rec = recall(&test_data, &train_data, k);
    double f1 = f1_score(&test_data, &train_data, k);
    
    printf("========================================\n");
    printf("Resultados del modelo:\n");
    printf("========================================\n");
    printf("Accuracy:  %.4f (%.2f%%)\n", acc, acc * 100.0);
    printf("Precision: %.4f (%.2f%%)\n", prec, prec * 100.0);
    printf("Recall:    %.4f (%.2f%%)\n", rec, rec * 100.0);
    printf("F1-score:  %.4f\n", f1);
    printf("========================================\n\n");
    
    // Ejemplo de predicción individual
    printf("Ejemplo de prediccion individual:\n");
    if (test_data.n_samples > 0) {
        double *sample = test_data.samples[0].features;
        int predicted = knn_predict(&train_data, sample, k, test_data.n_features);
        int actual = test_data.samples[0].label;
        
        printf("  Muestra de prueba #0:\n");
        printf("    Prediccion: %s\n", predicted == 1 ? "Churn (Yes)" : "No Churn (No)");
        printf("    Real:       %s\n", actual == 1 ? "Churn (Yes)" : "No Churn (No)");
        printf("    Correcto:   %s\n", predicted == actual ? "Si" : "No");
    }
    
    // Liberar memoria
    free_dataset(&train_data);
    free_dataset(&test_data);
    
    printf("\nPrograma finalizado correctamente.\n");
    return 0;
}

