#include<string>
#include<fstream>
#include<ostream>
#include<math.h>
#include"../../include/tensor.h"
#include"../helpers.h"

int main(int argc, char *argv[]) {
    
    bool verbose = check_verbose(argc, argv);

    Tensor* a = Tensor::fromFile("a.bin");
    //Tensor* a_grad = Tensor::fromFile("a_grad.bin");
    Tensor* b = Tensor::fromFile("b.bin");
    //Tensor* b_grad = Tensor::fromFile("b_grad.bin");
    Tensor* c_exp = Tensor::fromFile("c.bin");
    //Tensor* c_grad = Tensor::fromFile("c_grad.bin");
    a->toGPU();
    b->toGPU();
    c_exp->toGPU();

    Tensor* c_act = a->mm(b); 
    c_act->backward();

    check(tensor_diff(c_exp, c_act), "matmmul");
    if(verbose) std::cout << "c_act:\n" << c_act->pretty() << "\n";
    /*
    if(verbose) std::cout << "c_grad:\n" << c_act->pretty_grad(true) << "\n";
    check(tensor_diff(a_grad, a, true), "matmul grad in");
    if(verbose) std::cout << "a_grad:\n" << a->pretty_grad(true) << "\n";
    check(tensor_diff(b_grad, b, true), "matmul grad out");
    if(verbose) std::cout << "b_grad:\n" << b->pretty_grad(true) << "\n";
    */
    return 0;
}
