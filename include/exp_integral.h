#include <cuda_runtime.h>

#ifndef EXP_INTEGRAL_H
#define EXP_INTEGRAL_H


// CPU versions
double exponentialIntegralDoubleCPU(int n, double x);
float  exponentialIntegralFloatCPU(int n, float x);

// GPU entry 
void exponentialIntegralGPUFloat(int n, int m, float a, float b, float* result,double* time, cudaStream_t stream=0 );
void exponentialIntegralGPUDouble(int n, int m, double a, double b, double* result,double* time, cudaStream_t stream=0);

#endif // EXP_INTEGRAL_H
