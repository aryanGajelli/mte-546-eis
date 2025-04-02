#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 5
#define SECONDS 30
#define COUNTS_PER_SEC 10
#define T_INC (1./COUNTS_PER_SEC)
#define T_SIZE (SECONDS*COUNTS_PER_SEC)
#define ITERATIONS 100000
#define MAX_M 0.3

int main() {
    double f[N];
    double t[T_SIZE];
    double best_M = DBL_MAX;
    double best_phi[N];
    double best_amp[N];

    // Generate geometric sequence for f
    double start = 0.1, end = 1.0;
    for (int i = 0; i < N; i++) {
        f[i] = start * pow(end / start, (double)i / (N - 1));
    }

    // Generate time array t
    for (int i = 0; i < T_SIZE; i++) {
        t[i] = i * T_INC;
    }

    // Main loop
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
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
            amp[i] = ((double)rand() / RAND_MAX) * 0.05+0.1;
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

        if (M < best_M) {
            best_M = M;
            for (int i = 0; i < N; i++) {
                best_phi[i] = phi[i];
                best_amp[i] = amp[i];
            }
            printf("%d\tBest_M: %f\n", iteration, best_M);
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
        if (M < MAX_M)
            return 0;
    }

    return 0;
}
