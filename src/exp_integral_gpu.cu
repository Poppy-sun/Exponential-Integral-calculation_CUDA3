#include "exp_integral.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__constant__ float constEulerFloat = 0.5772156649015329f;
__constant__ double constEulerDouble = 0.5772156649015329;

__device__ float exponentialIntegralFloatKernel(int n, float x) {
    const float epsilon = 1.E-30f;
    const float big = 1.E+30f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n == 0) return expf(-x) / x;
    if (x > 1.0f) {
        b = x + n;
        c = big;
        d = 1.0f / b;
        h = d;
        for (i = 1; i <= 1000; i++) {
            a = -i * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= epsilon)
                return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0) ? 1.0f / nm1 : -logf(x) - constEulerFloat;
        fact = 1.0f;
        for (i = 1; i <= 1000; i++) {
            fact *= -x / i;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                psi = -constEulerFloat;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__device__ double exponentialIntegralDoubleKernel(int n, double x) {
    const double epsilon = 1.E-30;
    const double big = 1.E+300;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n == 0) return exp(-x) / x;
    if (x > 1.0) {
        b = x + n;
        c = big;
        d = 1.0 / b;
        h = d;
        for (i = 1; i <= 1000; i++) {
            a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon)
                return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0) ? 1.0 / nm1 : -log(x) - constEulerDouble;
        fact = 1.0;
        for (i = 1; i <= 1000; i++) {
            fact *= -x / i;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                psi = -constEulerDouble;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__global__ void expIntKernelFloat(int n, int m, float a, float b, float* result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < m) {
        float x = a + ((b - a) * (j + 1) / m);
        result[i * m + j] = exponentialIntegralFloatKernel(i + 1, x);
    }
}

__global__ void expIntKernelDouble(int n, int m, double a, double b, double* result) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < m) {
        double x = a + ((b - a) * (j + 1) / m);
        result[i * m + j] = exponentialIntegralDoubleKernel(i + 1, x);
    }
}

void exponentialIntegralGPUFloat(int n, int m, float a, float b, float* host_result, double* time) {
    float* d_result;
    size_t size = n * m * sizeof(float);
    cudaMalloc(&d_result, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + 15) / 16, (n + 15) / 16);
    expIntKernelFloat<<<numBlocks, threadsPerBlock>>>(n, m, a, b, d_result);

    cudaMemcpyAsync(host_result, d_result, size, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime (&milliseconds, start, stop);
    *time = milliseconds/1000.0;

    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void exponentialIntegralGPUDouble(int n, int m, double a, double b, double* host_result, double* time) {
    double* d_result;
    size_t size = n * m * sizeof(double);
    cudaMalloc(&d_result, size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + 15) / 16, (n + 15) / 16);
    expIntKernelDouble<<<numBlocks, threadsPerBlock>>>(n, m, a, b, d_result);
   
    cudaMemcpyAsync(host_result, d_result, size, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *time = milliseconds / 1000.0;

    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

