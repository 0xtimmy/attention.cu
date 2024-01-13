#include<string>
#include<iostream>
#include"../include/tensor.h"

int test_tensor() {

    float* data0 = (float*)malloc(sizeof(float) * 4);
    int* shape0 = (int*)malloc(sizeof(int));
    shape0[0] = 4;
    for (int i = 0; i < 4; i++) { data0[i] = (float)i; }
    Tensor* t0 = new Tensor(1, shape0, data0);
    std::cerr << "1 dimentional tensor:\n" << (t0->pretty()) << "\n";

    float* data1 = (float*)malloc(sizeof(float) * 4);
    int shape1[] = { 2, 2 };
    for (int i = 0; i < 4; i++) { data1[i] = (float)i; }
    Tensor* t1 = new Tensor(2, shape1, data1);
    std::cerr << "2 dimentional tensor:\n" << (t1->pretty()) << "\n";

    float* data2 = (float*)malloc(sizeof(float) * 12);
    int shape2[] = {3, 2, 2};
    for (int i = 0; i < 12; i++) { data2[i] = (float)i; }
    Tensor* t2 = new Tensor(3, shape2, data2);
    std::cerr << "3 dimentional tensor:\n" << (t2->pretty()) << "\n";

    float* data3 = (float*)malloc(sizeof(float) * 16);
    int shape3[] = {2, 2, 2, 2};
    for (int i = 0; i < 16; i++) { data3[i] = (float)i; }
    Tensor* t3 = new Tensor(4, shape3, data3);
    std::cerr << "4 dimentional tensor:\n" << (t3->pretty()) << "\n";

    return 0;
}

int test_zeros() {
    int shape[] = {2, 2, 2, 2};
    Tensor* t = new Tensor(4, shape);
    std::cerr <<  "size: " << t->size() << "\n";
    t->zeros();
    std::cerr << "tensor of zeros on cpu:\n" << (t->pretty()) << "\n";

    t->toGPU();
    t->zeros();
    t->toCPU();

    std::cerr << "tensor of zeros on gpu:\n" << (t->pretty()) << "\n";

    return 0;
}

int test_set() {
    int shape[] = {2, 2, 2, 2};
    Tensor* t = new Tensor(4, shape);
    std::cerr <<  "size: " << t->size() << "\n";
    t->toGPU();
    t->set(1.0);
    t->toCPU();
    std::cerr << "tensor of ones:\n" << (t->pretty()) << "\n";
    return 0;
}

/*
int test_uniform() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    int shape[] = {10};
    Tensor* t = new Tensor(10, shape);
    t->toGPU();
    t->uniform(gen);
    std::cerr << "tensor w/ uniform dist:\n" << (t->pretty()) << "\n";

    curandDestroyGenerator(gen);

    return 0;
}

int test_normal() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    int shape[] = {10};
    Tensor* t = new Tensor(10, shape);
    t->toGPU();
    t->normal(gen);
    std::cerr << "tensor w/ normal dist -- mean=0.0, std=1.0:\n" << (t->pretty()) << "\n";

    t->normal(gen, 1.0, 0.25);
    std::cerr << "tensor w/ normal dist -- mean=1.0, std=0.25:\n" << (t->pretty()) << "\n";

    curandDestroyGenerator(gen);

    return 0;
}
*/