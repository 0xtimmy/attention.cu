#include"tests/tensor.h"

void check(int status, std::string msg);

int main() {
    check(test_tensor(), "Tensor Initialization");
}

void check(int status, std::string msg) {
    if(status == 0) std::cout << "\033[1;32m[PASSED]\033[0m " << msg << "\n";
    else std::cout << "\033[1;31m[FAILED]\033[0m " << msg << "\n";
}

