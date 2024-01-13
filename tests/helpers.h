#ifndef TEST_HELPERS
#define TEST_HELPERS

#include<string>
#include<iostream>
#include"../../include/tensor.h"

void check(float error, std::string msg="") {
    if(error < 0.0000000001) std::cout << "[\033[1;32mPASSED\033[0m] " << msg << "\n";
    else std::cout << "[\033[1;31mFAILED\033[0m] " << msg << ": error = " << std::to_string(error) << "\n";
}

float tensor_diff(Tensor* expected, Tensor* actual, bool grad = false) {
    expected->toCPU();
    actual->toCPU();
    if(expected->size() != actual->size()) {
        std::cerr << "mismatched sizes: expected " << expected->pretty_shape() << " got " << actual->pretty_shape() << "\n";
        return INFINITY;
    }
    float error = 0;
    if(grad) {
        for(int i = 0; i < expected->size(); i++) {
            error += fabs(expected->cpu_ptr()[i] - actual->cpu_grad_ptr()[i]);
        }
    } else {
        for(int i = 0; i < expected->size(); i++) {
            error += fabs(expected->cpu_ptr()[i] - actual->cpu_ptr()[i]);
        }
    }
    return error;
}

bool check_verbose(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            return true;
        }
    }
    return false;
}

#endif