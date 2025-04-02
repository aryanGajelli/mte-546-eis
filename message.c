#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <pthread.h>

#define N 10
#define SECONDS 30
#define COUNTS_PER_SEC 10
#define T_INC (1. / COUNTS_PER_SEC)
#define T_SIZE (SECONDS * COUNTS_PER_SEC)
#define ITERATIONS 50000000
#define MAX_M 0.3
#define THREAD_COUNT 24

typedef struct {
    int thread_id;
    double *f;
    double *t;
    double *best_M;
    double *best_phi;
    double *best_amp;
    pthread_mutex_t *mutex;
} ThreadData;

void *compute(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int thread_id = data->thread_id;
    double *f = data->f;
    double *t = data->t;
    double *best_M = data->best_M;
    double *best_phi = data->best_phi;
    double *best_amp = data->best_amp;
    pthread_mutex_t *mutex = data->mutex;

    for (int iteration = thread_id; iteration < ITERATIONS; iteration += THREAD_COUNT) {
        double amp[N];
        double phi[N];
        double s[T_SIZE] = {0};
        double M;

        // Generate random phi values in range [0, 2*pi]
        for (int i = 0; i < N; i++) {
            phi[i] = ((double)rand() / RAND_MAX) * 2 * M_PI;
        }

        // Generate random amp values in range [0.1, 0.2]
        for (int i = 0; i < N; i++) {
            amp[i] = ((double)rand() / RAND_MAX) * 0.05 + 0.1;
        }

        // Compute sum of sine functions
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < T_SIZE; j++) {
                s[j] += amp[i] * sin(2 * M_PI * f[i] * t[j] + phi[i]);
            }
        }

        // Find max absolute value of s
        M = fabs(s[0]);
        for (int j = 1; j < T_SIZE; j++) {
            if (fabs(s[j]) > M) {
                M = fabs(s[j]);
            }
        }

        // Update best_M and best_phi if current M is smaller
        pthread_mutex_lock(mutex);
        if (M < *best_M) {
            *best_M = M;
            for (int i = 0; i < N; i++) {
                best_phi[i] = phi[i];
                best_amp[i] = amp[i];
            }
            printf("Thread %d\tIteration %d\tBest_M: %f\n", thread_id, iteration, *best_M);
            printf("Best_amp: ");
            for (int i = 0; i < N; i++) {
                printf("%f ", best_amp[i]);
            }
            printf("\n");
            printf("Best_phi: ");
            for (int i = 0; i < N; i++) {
                printf("%f ", best_phi[i]);
            }
            printf("\n");
        }
        pthread_mutex_unlock(mutex);

        if (*best_M < MAX_M) {
            break;
        }
    }

    return NULL;
}

int main() {
    double f[N];
    double t[T_SIZE];
    double best_M = DBL_MAX;
    double best_phi[N];
    double best_amp[N];
    pthread_t threads[THREAD_COUNT];
    pthread_mutex_t mutex;

    // Initialize mutex
    pthread_mutex_init(&mutex, NULL);

    // Generate geometric sequence for f
    double start = 0.1, end = 1.0;
    for (int i = 0; i < N; i++) {
        f[i] = start * pow(end / start, (double)i / (N - 1));
    }

    // Generate time array t
    for (int i = 0; i < T_SIZE; i++) {
        t[i] = i * T_INC;
    }

    // Create threads
    ThreadData thread_data[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].f = f;
        thread_data[i].t = t;
        thread_data[i].best_M = &best_M;
        thread_data[i].best_phi = best_phi;
        thread_data[i].best_amp = best_amp;
        thread_data[i].mutex = &mutex;

        pthread_create(&threads[i], NULL, compute, &thread_data[i]);
    }

    // Join threads
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }

    // Destroy mutex
    pthread_mutex_destroy(&mutex);

    return 0;
}