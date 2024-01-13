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
    Tensor* c_attn_weight = Tensor::fromFile("c_attn_weight.bin");
    Tensor* c_attn_bias = Tensor::fromFile("c_attn_bias.bin");
    Tensor* c_proj_weight = Tensor::fromFile("c_proj_weight.bin");
    Tensor* c_proj_bias = Tensor::fromFile("c_proj_bias.bin");
    Tensor* c_attn_weight_grad = Tensor::fromFile("c_attn_weight_grad.bin");
    Tensor* c_attn_bias_grad = Tensor::fromFile("c_attn_bias_grad.bin");
    Tensor* c_proj_weight_grad = Tensor::fromFile("c_proj_weight_grad.bin");
    Tensor* c_proj_bias_grad = Tensor::fromFile("c_proj_bias_grad.bin");
    //Tensor* x_grad = Tensor::fromFile("x_grad.bin");
    x->toGPU();
    x->name = "x";
    y_exp->toGPU();

    SelfAttention* sa = new SelfAttention(32, 4, 16);
    sa->toGPU();
    sa->c_attn->loadparameter("weights", c_attn_weight);
    sa->c_attn->loadparameter("bias", c_attn_bias);
    sa->c_proj->loadparameter("weights", c_proj_weight);
    sa->c_proj->loadparameter("bias", c_proj_bias);
    
    sa->c_attn->parameters["weights"]->name = "c_attn weights";
    sa->c_attn->parameters["bias"]->name = "c_attn bias";
    sa->c_proj->parameters["weights"]->name = "c_proj weights";
    sa->c_proj->parameters["bias"]->name = "c_proj bias";

    sa->c_attn->parameters["weights"]->toGPU();
    sa->c_attn->parameters["bias"]->toGPU();
    sa->c_proj->parameters["weights"]->toGPU();
    sa->c_proj->parameters["bias"]->toGPU();

    Tensor* y_act = sa->forward(x);
    std::cerr << "y_act\n";
    y_act->toGPU();

    y_act->backward();
    std::cerr << "y_act backward\n";

    x->toCPU();
    if(verbose) std::cout << "x:\n" << x->pretty() << "\n";

    check(tensor_diff(y_exp, y_act), "attention");
    if(verbose) std::cout << "y_act:\n" << y_act->pretty() << "\n";
    //check(tensor_diff(x_grad, x, true), "x_grad");
    //if(verbose) std::cout << "x_grad:\n" << x->pretty_grad(true) << "\n";
    check(tensor_diff(c_attn_weight_grad, sa->c_attn->parameters["weights"], true), "c_attn_weight_grad grad");
    if(verbose) std::cout << "c_attn_weight_grad:\n" << sa->c_attn->parameters["weights"]->pretty_grad(true) << "\n";
    check(tensor_diff(c_attn_bias_grad, sa->c_attn->parameters["bias"], true), "c_attn_bias_grad grad");
    if(verbose) std::cout << "c_attn_bias_grad:\n" << sa->c_attn->parameters["bias"]->pretty_grad(true) << "\n";
    check(tensor_diff(c_proj_weight_grad, sa->c_proj->parameters["weights"], true), "c_proj_weight_grad grad");
    if(verbose) std::cout << "c_proj_weight_grad:\n" << sa->c_proj->parameters["weights"]->pretty_grad(true) << "\n";
    check(tensor_diff(c_proj_bias_grad, sa->c_proj->parameters["bias"], true), "c_proj_bias_grad grad");
    if(verbose) std::cout << "c_proj_bias_grad:\n" << sa->c_proj->parameters["bias"]->pretty_grad(true) << "\n";
    return 0;
}
