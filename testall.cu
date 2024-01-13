//#include"tests/tensor.h"
//#include"tests/algebra.h"
#include"tests/grad.h"

void check(int status, std::string msg);

int main() {
    //check(test_tensor(), "Tensor Initialization");
    //check(test_zeros(), "Tensor zeros");
    //check(test_set(), "Tensor set");
    //check(test_uniform(), "Tensor Uniform");
    //check(test_normal(), "Tensor Normal");
    //check(test_embed(), "Embed");
    //check(test_layernorm(), "Layernorm");
    //check(test_linear(), "Linear Transformation");
    //check(test_softmax(), "Softmax");
    //check(test_mm(), "matmul");
    //check(test_transpose(), "transpose");
    //check(test_gelu(), "GELU");
    //check(test_split(), "Deep Split");
    //check(test_loss(), "Negative Loglikelihood");
    //check(test_cat_simple(), "cat simple");
    //check(test_cat_dims(), "cat dims");

    //check(grad::test_arithmetic(), "Gradient: Arithmetic");
    //check(grad::test_matmul(), "Gradient: Matmul");
    //check(grad::test_index(), "Gradient: Index");
    //check(grad::test_loss(), "Gradient: loss");
    //check(grad::test_softmax(), "Gradient: softmax");
    check(grad::test_split(), "split");
}

void check(int status, std::string msg) {
    if(status == 0) std::cout << "\033[1;32m[PASSED]\033[0m " << msg << "\n";
    else std::cout << "\033[1;31m[FAILED]\033[0m " << msg << "\n";
}

