#include<cmath>
#include <cuda_runtime.h>

// Unary ----------------------------------------------------------------------
__global__ void exp(float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = expf(x[idx]);
    }
}

// Binary ---------------------------------------------------------------------
__global__ void add(float* a, float* b, float* y, int sizea, int sizeb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int scale = sizea/sizeb;
    if (idx < sizea) {
        y[idx] = a[idx] + b[idx/scale];
    }
}

__global__ void sub(float* a, float* b, float* y, int sizea, int sizeb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int scale = sizea/sizeb;
    if (idx < sizea) {
        y[idx] = a[idx] - b[idx/scale];
    }
}

__global__ void mul(float* a, float* b, float* y, int sizea, int sizeb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int scale = sizea/sizeb;
    if (idx < sizea) {
        y[idx] = a[idx] * b[idx/scale];
    }
}

__global__ void div(float* a, float* b, float* y, int sizea, int sizeb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int scale = sizea/sizeb;
    if (idx < sizea) {
        y[idx] = a[idx] / b[idx/scale];
    }
}

// Matrix ---------------------------------------------------------------------
__global__ void linear(float* x, float* w, float* b, float* y, int size) {
    
}

__global__ void mm(float* a, float* b, float* y, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int value = 0;
        for (int i = 0; i < n; ++i) {
            value += a[row * n + i] * b[i * n + col];
        }
        y[row * n + col] = value;
    }
}

// Normalization --------------------------------------------------------------
__global__ void softmax(float* x, float* y, int size) {
    extern __shared__ float minisums[channels];

    // channles must be reduced to calculate the sum; 
    int batch = blockIdx.x;
    int batchsize = size / blockDim.x
    int batchoffset = batch * batchsize;
    
    float sum;
    for (int i = 0; i < batchsize; i++) {
        y[batchoffset + i] = __expf(x[batchoffset + i]);
        sum += y[batchoffset + i];
    }

    for (int i = 0; i < batchsize; i++) {
        y[batchoffset + i] = y[batchoffset + i] / sum;
    }
}

__global__ void layer_norm(float* x, float* w, float* b, float* y, int layersize, int size, float eps) {
    int layer = blockIdx.x * blockDim.x + threadIdx.x;
    if(layer*layersize < size) {
        // calculate expectation
        float mu = 0;
        for (int i = 0; i < layersize; i++) {
            mu += x[layer*layersize + i];
        }
        mu /= layersize;

        // calculate variance
        float sigsquared = 0;;
        for (int i = 0; i < layersize; i++) {
            sigsquared += powf(x[layer*layersize + i] - mu, 2);
        }
        sigsquared /= layersize;
        
        for(int i = 0; i < layersize; i++) {
            y[layer*layersize + i]  = (x[layer*layersize + i] - mu) 
                    / sqrtf(sigsquared + eps) 
                    * w[i]
                    + b[i];
        }
    }
}