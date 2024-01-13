#include<string>
#include<fstream>
#include<ostream>
#include<math.h>
#include"../../include/tensor.h"
#include"../../projects\shakespeare\model.h"
#include"../helpers.h"

int main(int argc, char *argv[]) {
    
    bool verbose = check_verbose(argc, argv);

    Tensor* x = Tensor::fromFile("x.bin");
    Tensor* y_exp = Tensor::fromFile("y.bin");
    Tensor* ln_weight = Tensor::fromFile("ln_weight.bin");
    Tensor* ln_bias = Tensor::fromFile("ln_bias.bin");
    Tensor* ln_weight_grad = Tensor::fromFile("ln_weight_grad.bin");
    Tensor* ln_bias_grad = Tensor::fromFile("ln_bias_grad.bin");
    x->toGPU();
    y_exp->toGPU();

    Linear* ln = new Linear(x->shape()[1], y_exp->shape()[1]);
    ln_weight->toGPU();
    ln->parameters["weights"] = ln_weight;
    ln->parameters["bias"] = ln_bias;

    Tensor* y_act = ln->forward(x);

    std::cerr << "y_act shape: " << y_act->pretty_shape() << "\n";

    y_act->backward();

    check(tensor_diff(y_exp, y_act), "linear");
    if(verbose) std::cout << "y_act:\n" << y_act->pretty() << "\n";
    check(tensor_diff(ln_weight_grad, ln->parameters["weights"], true), "linear weight grad");
    if(verbose) std::cout << "ln_weight_grad:\n" << ln->parameters["weights"]->pretty_grad(true) << "\n";
    check(tensor_diff(ln_bias_grad, ln->parameters["bias"], true), "linear bias grad");
    if(verbose) std::cout << "ln_bias_grad:\n" << x->pretty_grad(true) << "\n";
    return 0;
}
