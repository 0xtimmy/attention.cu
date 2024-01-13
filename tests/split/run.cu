#include<string>
#include<fstream>
#include<ostream>
#include<math.h>
#include"../../include/tensor.h"
#include"../helpers.h"

int main(int argc, char *argv[]) {
    
    bool verbose = check_verbose(argc, argv);

    Tensor* x = Tensor::fromFile("x.bin");
    Tensor* weight = Tensor::fromFile("weight.bin");
    Tensor* x_grad = Tensor::fromFile("x_grad.bin");
    //Tensor* x0_grad = Tensor::fromFile("x0_grad.bin");
    //Tensor* x1_grad = Tensor::fromFile("x1_grad.bin");
    Tensor* y_exp = Tensor::fromFile("y.bin");
    x->toGPU();
    weight->toGPU();
    y_exp->toGPU();

    Tensor** xs = x->split(2, 2);
    Tensor* y0 = xs[0]->mul(weight);
    Tensor* y_act = y0->add(xs[1]);

    y_act->backward();

    check(tensor_diff(y_exp, y_act), "split");
    if(verbose) std::cout << "y_act:\n" << y_act->pretty() << "\n";
    check(tensor_diff(x_grad, x, true), "x grad");
    if(verbose) std::cout << "x_grad:\n" << x->pretty_grad(true) << "\n";
    /*
    check(tensor_diff(x0_grad, xs[0], true), "x0 grad");
    if(verbose) std::cout << "x0_grad:\n" << xs[0]->pretty_grad(true) << "\n";
    check(tensor_diff(x1_grad, xs[1], true), "x1 grad");
    if(verbose) std::cout << "x1_grad:\n" << xs[1]->pretty_grad(true) << "\n";
    */
    return 0;
}
