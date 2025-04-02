#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI   3.14159265358979323846264338327950288
#endif

#define N 5
#define SECONDS 30
#define COUNTS_PER_SEC 1000
#define T_INC (1. / COUNTS_PER_SEC)
#define T_SIZE (SECONDS * COUNTS_PER_SEC)
#define ITERATIONS 10000000
#define MAX_M 0.3
double start = 0.1, end = 1.0;

__global__ void compute(double *f, double *t, double *global_best_M, double *global_best_phi, double *global_best_amp, int iterations) {
    extern __shared__ double shared_best_M[];
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= iterations) return;

    curandState state;
    curand_init(thread_id, 0, 0, &state);

    double amp[N], phi[N], s[T_SIZE] = {0}, local_M;

    // Generate random phi values in range [0, 2*pi]
    for (int i = 0; i < N; i++) {
        phi[i] = curand_uniform(&state) * 2 * M_PI;
    }

    // Generate random amp values in range [0.1, 0.3]
    for (int i = 0; i < N; i++) {
        amp[i] = curand_uniform(&state) * 0.2 + 0.1;
    }

    // Compute sum of sine functions
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < T_SIZE; j++) {
            s[j] += amp[i] * sin(2 * M_PI * f[i] * t[j] + phi[i]);
        }
    }

    // Find max absolute value of s
    local_M = fabs(s[0]);
    for (int j = 1; j < T_SIZE; j++) {
        if (fabs(s[j]) > local_M) {
            local_M = fabs(s[j]);
        }
    }

    // Store local minimum in shared memory
    shared_best_M[threadIdx.x] = local_M;
    __syncthreads();

    // Perform parallel reduction to find the minimum value in the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_best_M[threadIdx.x] = fmin(shared_best_M[threadIdx.x], shared_best_M[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Update global minimum if thread 0 has the smallest value in the block
    if (threadIdx.x == 0) {
        atomicMin((unsigned long long int *)global_best_M, __double_as_longlong(shared_best_M[0]));
        if (*global_best_M == shared_best_M[0]) {
            for (int i = 0; i < N; i++) {
                global_best_phi[i] = phi[i];
                global_best_amp[i] = amp[i];
            }
            printf("Best_M: %f\n", *global_best_M);
            printf("Best_amp: ");
            for (int i = 0; i < N; i++) {
                printf("%f ", global_best_amp[i]);
            }
            printf("\n");
            printf("Best_phi: ");
            for (int i = 0; i < N; i++) {
                printf("%f ", global_best_phi[i]);
            }
            printf("\n");
        }
    }
}

int main() {
    double f[N], t[T_SIZE], best_M = DBL_MAX, best_phi[N], best_amp[N];
    
    // Generate geometric sequence for f
    for (int i = 0; i < N; i++) {
        f[i] = start * pow(end / start, (double)i / (N - 1));
    }

    // Generate time array t
    for (int i = 0; i < T_SIZE; i++) {
        t[i] = i * T_INC;
    }

    // Allocate memory on the GPU
    double *d_f, *d_t, *d_best_M, *d_best_phi, *d_best_amp;
    
    cudaMalloc((void **)&d_f, N * sizeof(double));
    cudaMalloc((void **)&d_t, T_SIZE * sizeof(double));
    cudaMalloc((void **)&d_best_M, sizeof(double));
    cudaMalloc((void **)&d_best_phi, N * sizeof(double));
    cudaMalloc((void **)&d_best_amp, N * sizeof(double));

    cudaMemcpy(d_f, f, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, t, T_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_best_M, &best_M, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (ITERATIONS + threads_per_block - 1) / threads_per_block;

    compute<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(double)>>>(d_f, d_t, d_best_M, d_best_phi, d_best_amp, ITERATIONS);

    // Copy results back to host
    cudaMemcpy(&best_M, d_best_M, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_phi, d_best_phi, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_amp, d_best_amp, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_f);
    cudaFree(d_t);
    cudaFree(d_best_M);
    cudaFree(d_best_phi);
    cudaFree(d_best_amp);

    
    
    return 0;
}
