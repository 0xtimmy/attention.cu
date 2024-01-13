#include<string>
#include<fstream>
#include<ostream>
#include<math.h>
#include"../../include/tensor.h"
#include"../helpers.h"

int main(int argc, char *argv[]) {
    
    bool verbose = check_verbose(argc, argv);

    Tensor* x = Tensor::fromFile("x.bin");
    Tensor* x_grad = Tensor::fromFile("x_grad.bin");
    Tensor* y_exp = Tensor::fromFile("y.bin");
    x->toGPU();
    y_exp->toGPU();

    Tensor* y_act = x->softmax(2); 
    y_act->backward();

    check(tensor_diff(y_exp, y_act), "softmax");
    if(verbose) std::cout << "y_act:\n" << y_act->pretty() << "\n";
    if(verbose) std::cout << "y_grad:\n" << y_act->pretty_grad(true) << "\n";
    check(tensor_diff(x_grad, x, true), "softmax grad");
    if(verbose) std::cout << "x_grad:\n" << x->pretty_grad(true) << "\n";
    return 0;
}
