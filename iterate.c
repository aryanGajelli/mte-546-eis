#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 10
#define T_SIZE 3000  // Number of time steps (30 / 0.01)

// Function to generate geometric sequence (similar to np.geomspace)
void geomspace(double start, double end, int num, double *result) {
    double factor = pow(end / start, 1.0 / (num - 1));
    result[0] = start;
    for (int i = 1; i < num; i++) {
        result[i] = result[i - 1] * factor;
    }
}

// Function to generate linear space (similar to np.linspace)
void linspace(double start, double end, int num, double *result) {
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
}

int main() {
    double f01_1[N];
    geomspace(0.1, 1.0, N, f01_1);

    double t[T_SIZE];
    for (int i = 0; i < T_SIZE; i++) {
        t[i] = i * 0.01;
    }

    double best_M = DBL_MAX;
    double best_phi[N];
    
    double phase_shifts[N];
    double p_values[N];
    linspace(0, 2 * M_PI, N, p_values);
    int start_idx[] = {0, 0, 1, 3, 4, 5, 6, 0, 1, 2};
    long long unsigned i =0;

    for (int p1_idx = start_idx[0]; p1_idx < N; p1_idx++) {
        for (int p2_idx = start_idx[1]; p2_idx < N; p2_idx++) {
            for (int p3_idx = start_idx[2]; p3_idx < N; p3_idx++) {
                for (int p4_idx = start_idx[3]; p4_idx < N; p4_idx++) {
                    for (int p5_idx = start_idx[4]; p5_idx < N; p5_idx++) {
                        for (int p6_idx = start_idx[5]; p6_idx < N; p6_idx++) {
                            for (int p7_idx = start_idx[6]; p7_idx < N; p7_idx++) {
                                for (int p8_idx = start_idx[7]; p8_idx < N; p8_idx++) {
                                    for (int p9_idx = start_idx[8]; p9_idx < N; p9_idx++) {
                                        for (int p10_idx = start_idx[9]; p10_idx < N; p10_idx++) {
                                            phase_shifts[0] = p_values[p1_idx];
                                            phase_shifts[1] = p_values[p2_idx];
                                            phase_shifts[2] = p_values[p3_idx];
                                            phase_shifts[3] = p_values[p4_idx];
                                            phase_shifts[4] = p_values[p5_idx];
                                            phase_shifts[5] = p_values[p6_idx];
                                            phase_shifts[6] = p_values[p7_idx];
                                            phase_shifts[7] = p_values[p8_idx];
                                            phase_shifts[8] = p_values[p9_idx];
                                            phase_shifts[9] = p_values[p10_idx];

                                            double s[T_SIZE] = {0};
                                            for (int i = 0; i < T_SIZE; i++) {
                                                for (int j = 0; j < N; j++) {
                                                    s[i] += sin(2 * M_PI * f01_1[j] * t[i] + phase_shifts[j]);
                                                }
                                            }

                                            double M = fabs(s[0]);
                                            for (int i = 1; i < T_SIZE; i++) {
                                                if (fabs(s[i]) > M) {
                                                    M = fabs(s[i]);
                                                }
                                            }

                                            if (M < best_M) {
                                                best_M = M;
                                                for (int k = 0; k < N; k++) {
                                                    best_phi[k] = phase_shifts[k];
                                                }
                                                printf("Best M: %.6f\n", best_M);
                                                printf("Best phi: ");
                                                for (int k = 0; k < N; k++) {
                                                    printf("%.6f ", best_phi[k]);
                                                }
                                                printf("\n");
                                                 // print indexes of the
                                                printf("{%d, %d, %d, %d, %d, %d, %d, %d, %d, %d}\n",
                                                    p1_idx, p2_idx, p3_idx, p4_idx,
                                                    p5_idx, p6_idx, p7_idx, p8_idx,
                                                    p9_idx, p10_idx);
                                            }
                                            i++;
                                            if (i%1000==0) {
                                                printf("{%d, %d, %d, %d, %d, %d, %d, %d, %d, %d}\n",
                                                    p1_idx, p2_idx, p3_idx, p4_idx,
                                                    p5_idx, p6_idx, p7_idx, p8_idx,
                                                    p9_idx, p10_idx);
                                            }
                                           

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}
