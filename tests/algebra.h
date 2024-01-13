#include"../include/tensor.h"
#include"../include/algebra.h"
#include<cuda_runtime.h>
#include<iostream>

int test_embed() {
    int* idx = (int*)malloc(sizeof(int)*4);
    for (int i = 0; i < 4; i++) { idx[i] = i+2; }
    float* embd = (float*)malloc(sizeof(float)*8*3);
    for (int i = 0; i < 8*3; i++) { embd[i] = 1.0*i; }

    int* devidx;
    cce(cudaMalloc(&devidx, sizeof(int)*4), "malloc devidx");
    float* devembd;
    cce(cudaMalloc(&devembd, sizeof(float)*8*3), "malloc embd");
    float* devout;
    cce(cudaMalloc(&devout, sizeof(float)*4*3), "malloc out");

    cce(cudaMemcpy(devidx, idx, sizeof(int)*4, cudaMemcpyHostToDevice), "cpy devidx");
    cce(cudaMemcpy(devembd, embd, sizeof(float)*8*3, cudaMemcpyHostToDevice), "cpy devembd");

    algebra::embedding<<<32, 32>>>(devidx, devembd, devout, 3, 4);

    float* out = (float*)malloc(sizeof(float)*4*3);
    cce(cudaMemcpy(out, devout, sizeof(float)*4*3, cudaMemcpyDeviceToHost), "cpy out");

    std::cout << "Embedding output:\n";
    for(int i = 0; i < 4; i++) {
        std::cout << i << ": [" << out[i*3] << ", " << out[i*3 + 1] << ", " << out[i*3 + 2] << "]\n";
    }

    return 0;
}

int test_layernorm() {
    // four batches, embedding size 3, normalize the embedding
    const int num_layers = 3;
    const int layer_size = 56;
    float* x = (float*)malloc(sizeof(float)*num_layers*layer_size);
    for (int i = 0; i < num_layers*layer_size; i++) { x[i] = 1.0f; }
    float* w = (float*)malloc(sizeof(float)*layer_size);
    for (int i = 0; i < 3; i++) { w[i] = 1.0f; }
    float* b = (float*)malloc(sizeof(float)*3);
    for (int i = 0; i < 3; i++) { b[i] = 0.0f; }

    float* devx;
    cce(cudaMalloc(&devx, sizeof(float)*4*3), "malloc layernorm x");
    cce(cudaMemcpy(devx, x, sizeof(float)*4*3, cudaMemcpyHostToDevice));
    float* devw;
    cce(cudaMalloc(&devw, sizeof(float)*3), "malloc layernorm w");
    cce(cudaMemcpy(devw, w, sizeof(float)*3, cudaMemcpyHostToDevice));
    float* devb;
    cce(cudaMalloc(&devb, sizeof(float)*3), "malloc layernorm b");
    cce(cudaMemcpy(devb, b, sizeof(float)*3, cudaMemcpyHostToDevice));

    algebra::layernorm<<<32, 32>>>(devx, devw, devb, 3, 4*3);

    cce(cudaMemcpy(x, devx, sizeof(float)*4*3, cudaMemcpyDeviceToHost));

    std::cout << "Normalized Layers:\n";
    for(int i = 0; i < 4; i++) {
        std::cout << i << ": [" << x[i*3] << ", " << x[i*3 + 1] << ", " << x[i*3 + 2] << "]\n";
    }

    return 0;
}


/*
int test_linear() {

    float* x = (float*)malloc(sizeof(float)*3*2);
    float* w = (float*)malloc(sizeof(float)*2*4);
    float* b = (float*)malloc(sizeof(float)*4);
    float* y = (float*)malloc(sizeof(float)*3*4);

    for(int i = 0; i < 6; i++) { x[i] = 1; }
    for(int i = 0; i < 8; i++) { w[i] = 1; }
    for(int i = 0; i < 4; i++) { b[i] = 0; }

    float* devx;
    float* devw;
    float* devb;
    float* devy;
    cce(cudaMalloc(&devx, sizeof(float)*3*2));
    cce(cudaMalloc(&devw, sizeof(float)*2*4));
    cce(cudaMalloc(&devb, sizeof(float)*4));
    cce(cudaMalloc(&devy, sizeof(float)*3*4));

    cce(cudaMemcpy(devx, x, sizeof(float)*3*2, cudaMemcpyHostToDevice));
    cce(cudaMemcpy(devw, w, sizeof(float)*2*4, cudaMemcpyHostToDevice));
    cce(cudaMemcpy(devb, b, sizeof(float)*4, cudaMemcpyHostToDevice));

    algebra::linear<<<64, 64>>>(devx, devw, devb, devy, 2, 4, 12);

    cce(cudaMemcpy(y, devy, sizeof(float)*3*4, cudaMemcpyDeviceToHost));

    std::cout << "Linear Transformation:\n";
    for(int i = 0; i < 3; i++) {
        std::cout << "[" << x[i*2] << ", " << x[i*2 + 1] << "]\n";
    }

    std::cout << "x \n";

    for(int i = 0; i < 2; i++) {
        std::cout << "[" << w[i*4] << ", " << w[i*4 + 1] << ", " << w[i*4 + 2] << ", " << w[i*4 + 3] << "]\n";
    }

    std::cout << "-> \n";

    for(int i = 0; i < 3; i++) {
        std::cout << "[" << y[i*4] << ", " << y[i*4 + 1] << ", " << y[i*4 + 2] << ", " << y[i*4 + 3] << "]\n";
    }

    return 0;
}
*/

int test_softmax() {

    const int layers = 10;
    const int layer_size = 10;

    float* x = (float*)malloc(sizeof(float)*layers*layer_size);
    for (int i = 0; i < layers*layer_size; i++) { x[i] = 1.0f; }

    float* devx;
    cce(cudaMalloc(&devx, sizeof(float)*layers*layer_size));
    cce(cudaMemcpy(devx, x, sizeof(float)*layers*layer_size, cudaMemcpyHostToDevice));

    for(int l = 0; l < layers; l++) {
        std::cout << "[ ";
        for(int i = 0; i < layer_size; i++) { std::cout << x[l*layer_size + i] << " "; }
        std::cout << "]\n";
    }

    algebra::softmax<<<(layers+1)*layer_size, layer_size>>>(devx, devx, layer_size, layer_size*layers);

    cce(cudaMemcpy(x, devx, sizeof(float)*layers*layer_size, cudaMemcpyDeviceToHost), "softmax back to host");

    std::cout << "Softmaxxed:\n";
    for(int l = 0; l < layers; l++) {
        std::cout << "[ ";
        for(int i = 0; i < layer_size; i++) { std::cout << x[l*layer_size + i] << " "; }
        std::cout << "]\n";
    }

    return 0;
}

int test_mm() {

    float* a = (float*)malloc(sizeof(float)*3*2);
    float* b = (float*)malloc(sizeof(float)*2*4);
    float* y = (float*)malloc(sizeof(float)*3*4);
    for(int i = 0; i < 6; i++) { a[i] = 1.0f; }
    for(int i = 0; i < 8; i++) { b[i] = 1.0f; }

    float* deva;
    float* devb;
    float* devy;
    cce(cudaMalloc(&deva, sizeof(float)*3*2), "malloc a");
    cce(cudaMalloc(&devb, sizeof(float)*2*4), "malloc b");
    cce(cudaMalloc(&devy, sizeof(float)*3*4), "malloc y");
    cce(cudaMemcpy(deva, a, sizeof(float)*3*2, cudaMemcpyHostToDevice), "a to dev");
    cce(cudaMemcpy(devb, b, sizeof(float)*2*4, cudaMemcpyHostToDevice), "b to dev");

    algebra::mm<<<32, 32>>>(deva, devb, devy, 3, 2, 4, 3*4);

    cce(cudaMemcpy(y, devy, sizeof(float)*3*4, cudaMemcpyDeviceToHost), "y back");

    std::cout << "Matmulled:\n";
    for(int i = 0; i < 3; i++) {
        std::cout << "[" << y[i*4] << ", " << y[i*4 + 1] << ", " << y[i*4 + 2] << ", " << y[i*4 + 3] << "]\n";
    }

    return 0;
}

int test_transpose() {

    float* x = (float*)malloc(sizeof(float)*9);
    float* y = (float*)malloc(sizeof(float)*9);
    for (int i = 0; i < 9; i++) { x[i] = i; }

    float* devx;
    float* devy;
    cce(cudaMalloc(&devx, sizeof(float)*9));
    cce(cudaMalloc(&devy, sizeof(float)*9));
    cce(cudaMemcpy(devx, x, sizeof(float)*9, cudaMemcpyHostToDevice));

    algebra::transpose<<<32, 32>>>(devx, devy, 3, 3, 1, 9);

    cce(cudaMemcpy(y, devy, sizeof(float)*9, cudaMemcpyDeviceToHost));

    std::cout << "2D Transpose:\n";
    for(int i = 0; i < 3; i++) {
        std::cout << "[" << x[i*3] << ", " << x[i*3 + 1] << ", " << x[i*3 + 2] << "]\n";
    }
    std::cout << "->\n";
    for(int i = 0; i < 3; i++) {
        std::cout << "[" << y[i*3] << ", " << y[i*3 + 1] << ", " << y[i*3 + 2] << "]\n";
    }

    float* a = (float*)malloc(sizeof(float)*16);
    float* b = (float*)malloc(sizeof(float)*16);
    for (int i = 0; i < 16; i++) { a[i] = i; }

    float* deva;
    float* devb;
    cce(cudaMalloc(&deva, sizeof(float)*16));
    cce(cudaMalloc(&devb, sizeof(float)*16));
    cce(cudaMemcpy(deva, a, sizeof(float)*16, cudaMemcpyHostToDevice));

    algebra::transpose<<<64, 64>>>(deva, devb, 2, 2, 2, 16);

    cce(cudaMemcpy(b, devb, sizeof(float)*16, cudaMemcpyDeviceToHost));

    std::cout << "4D Transpose:\n";
    for(int i = 0; i < 2; i++) {
        std::cout << "[\n";
        for(int j = 0; j < 2; j ++) {
            std::cout << "  [[" << a[i*8 +j*4] << ", "  << a[i*8 +j*4 + 1] << "] [" << a[i*8 +j*4 +2] << ", "  << a[i*8 +j*4 +3]<< "]]\n";
        }
        std::cout << "]\n";
    }
    std::cout << "->\n";
    for(int i = 0; i < 2; i++) {
        std::cout << "[\n";
        for(int j = 0; j < 2; j ++) {
            std::cout << "  [[" << b[i*8 +j*4] << ", "  << b[i*8 +j*4 + 1] << "] [" << b[i*8 +j*4 +2] << ", "  << b[i*8 +j*4 +3]<< "]]\n";
        }
        std::cout << "]\n";
    }

    return 0;
}

int test_gelu() {
    float* x = (float*)malloc(sizeof(float)*9);
    for (int i = 0; i < 9; i++) { x[i] = i-4; }

    float* devx;
    cce(cudaMalloc(&devx, sizeof(float)*9), "malloc");
    cce(cudaMemcpy(devx, x, sizeof(float)*9, cudaMemcpyHostToDevice), "copy to device");

    float* devy;
    cce(cudaMalloc(&devy, sizeof(float)*9), "malloc");

    algebra::gelu<<<32, 32>>>(devx, devy, 9);

    cce(cudaMemcpy(x, devy, sizeof(float)*9, cudaMemcpyDeviceToHost), "copy from device");
    std::cerr << "[ ";
    for(int i = 0; i < 9; i++) { std::cerr << x[i] << " "; }
    std::cerr << "]\n";

    return 0;
}

int test_split() {
    float* x = (float*) malloc(sizeof(float)*18);
    for (int i = 0; i < 18; i++) { x[i] = i; }

    float* devx;
    cce(cudaMalloc(&devx, sizeof(float)*18), "malloc");
    cce(cudaMemcpy(devx, x, sizeof(float)*18, cudaMemcpyHostToDevice));
    float* devy;
    cce(cudaMalloc(&devy, sizeof(float)*18), "malloc");

    algebra::split<<<32, 32>>>(devx, devy, 3, 3, 18);

    float* a = (float*)malloc(sizeof(float)*6);
    float* b = (float*)malloc(sizeof(float)*6);
    float* c = (float*)malloc(sizeof(float)*6);
    cce(cudaMemcpy(a, devy, sizeof(float)*6, cudaMemcpyDeviceToHost), "copy to a");
    cce(cudaMemcpy(b, devy+6, sizeof(float)*6, cudaMemcpyDeviceToHost), "copy to b");
    cce(cudaMemcpy(c, devy+12, sizeof(float)*6, cudaMemcpyDeviceToHost), "copy to c");
    std::cerr << "[\n";
    for(int i = 0; i < 3; i++) { std::cerr << "[" << a[i*2] << ", " << a[i*2+1] << "]\n"; }
    std::cerr << "]\n";
    std::cerr << "[\n";
    for(int i = 0; i < 3; i++) { std::cerr << "[" << b[i*2] << ", " << b[i*2+1] << "]\n"; }
    std::cerr << "]\n";
    std::cerr << "[\n";
    for(int i = 0; i < 3; i++) { std::cerr << "[" << c[i*2] << ", " << c[i*2+1] << "]\n"; }
    std::cerr << "]\n";

    return 0;
}


int test_loss() {

    float* x = (float*)malloc(sizeof(float*)*9);
    for (int i = 0; i < 9; i++) { x[i] = 1.0f+i; }
    float* devx;
    cce(cudaMalloc(&devx, sizeof(float)*9));
    cce(cudaMemcpy(devx, x, sizeof(float)*9, cudaMemcpyHostToDevice));

    float* y = (float*)malloc(sizeof(float)*9);
    float* devy;
    cce(cudaMalloc(&devy, sizeof(float)*9));

    algebra::loss<<<9, 9>>>(devx, devy, 9);
    cce();

    cce(cudaMemcpy(y, devy, sizeof(float)*9, cudaMemcpyDeviceToHost));

    std::cerr << "Test Negative Normalized Log-Likelihood: " << y[0] << "\n";

    return 0;
}

int test_cat_simple() {
    int* a_shape = (int*)malloc(sizeof(int)*1);
    a_shape[0] = 2;
    int* b_shape = (int*)malloc(sizeof(int)*1);
    b_shape[0] = 2;
    
    Tensor* a = new Tensor(1, a_shape);
    Tensor* b = new Tensor(1, b_shape);
    a->cpu_ptr()[0] = 0;
    a->cpu_ptr()[1] = 1;
    b->cpu_ptr()[0] = 2;
    b->cpu_ptr()[1] = 3;

    a->toGPU();
    b->toGPU();

    Tensor* y = Tensor::cat(a, b, 0);
    y->toCPU();

    std::cerr << "cat simple: " << y->pretty() << "\n";

    return 0;
}

int test_cat_dims() {
    int* a_shape = (int*)malloc(sizeof(int)*3);
    a_shape[0] = 2;
    a_shape[1] = 3;
    a_shape[2] = 2;
    int* b_shape = (int*)malloc(sizeof(int)*3);
    b_shape[0] = 2;
    b_shape[1] = 3;
    b_shape[2] = 4;
    
    Tensor* a = new Tensor(3, a_shape);
    Tensor* b = new Tensor(3, b_shape);
    for(int i = 0; i < a->size(); i++) {
        a->cpu_ptr()[i] = 1.0f*i;
    }
    for(int i = 0; i < b->size(); i++) {
        b->cpu_ptr()[i] = 1.0f*i;
    }

    a->toGPU();
    b->toGPU();

    Tensor* y = Tensor::cat(a, b, 2);
    y->toCPU();
    a->toCPU();
    b->toCPU();
    std::cerr << "a:\n" << a->pretty() << "\n";
    std::cerr << "b:\n" << b->pretty() << "\n";
    std::cerr << "cat dims:\n" << y->pretty() << "\n";

    return 0;
}