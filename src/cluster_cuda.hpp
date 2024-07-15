#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cooperative_groups.h>

double* launchKernelCuda(const double* X, const int row);
void memAllocate();
void memFree();

static cudaStream_t s1;
static cublasHandle_t handle;
static double* d_th; 
static int* d_idmaxth;
static double* d_x;
static double* d_c1;
static int* d_row;
static double* d_c2;
static double* d_max;
static double* d_min;
static double* d_c1max;
static double* d_c1min;
static double* d_c2max;
static double* d_c2min;
static double* d_b;
static double* d_coef;
static double* h_coef;
static double* h_x;



