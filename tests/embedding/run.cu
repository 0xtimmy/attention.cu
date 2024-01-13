#include<string>
#include<fstream>
#include<ostream>
#include<math.h>
#include"../../include/tensor.h"
#include"../../projects\shakespeare\model.h"
#include"../helpers.h"

int main(int argc, char *argv[]) {
    
    bool verbose = check_verbose(argc, argv);

    std::string filename = "x.bin";
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file while reading tensor: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    int dim;
    if (!(file.read(reinterpret_cast<char*>(&dim), sizeof(dim)))) {
        std::cerr << "Error reading dim while reading tesnor from file: " << filename << std::endl;
        return false;
    }
    int* shape = (int*)malloc(sizeof(int)*dim);
    if(!(file.read(reinterpret_cast<char*>(shape), sizeof(int)*dim))) {
        std::cerr << "Error reading shape while reading tensor from file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    int size = 1;
    for(int i = 0; i < dim; i++) { size *= shape[i]; }
    int* x = (int*)malloc(sizeof(int) * size);
    if(!(file.read(reinterpret_cast<char*>(x), sizeof(int)*size))) {
        std::cerr << "Error reading data while reading tensor from file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cerr << "x: [ ";
    for(int i = 0; i < size; i++) { std::cerr << x[i] << " "; }
    std::cerr << "]\n";

    int* dev_x;
    cce(cudaMalloc(&dev_x, sizeof(int)*size));
    cce(cudaMemcpy(dev_x, x, sizeof(int)*size, cudaMemcpyHostToDevice));

    Tensor* y_exp = Tensor::fromFile("y.bin");
    Tensor* emb_weight = Tensor::fromFile("emb_weight.bin");
    Tensor* emb_weight_grad = Tensor::fromFile("emb_weight_grad.bin");
    y_exp->toGPU();

    Embedding* emb = new Embedding(64, 32);
    emb->toGPU();
    emb->loadparameter("weights", emb_weight);
    
    emb->parameters["weights"]->name = "emb weights";

    emb->parameters["weights"]->toGPU();

    Tensor* y_act = emb->forward(dev_x, shape[1], shape[0]);
    std::cerr << "y_act\n";
    y_act->toGPU();

    y_act->backward();

    //if(verbose) std::cout << "x:\n" << x->pretty() << "\n";


    check(tensor_diff(y_exp, y_act), "embedding");
    if(verbose) std::cout << "y_act:\n" << y_act->pretty() << "\n";
    //check(tensor_diff(x_grad, x, true), "x_grad");
    //if(verbose) std::cout << "x_grad:\n" << x->pretty_grad(true) << "\n";
    check(tensor_diff(emb_weight_grad, emb->parameters["weights"], true), "emb_weight_grad grad");
    if(verbose) std::cout << "emb_weight_grad:\n" << emb->parameters["weights"]->pretty_grad(true) << "\n";
    return 0;
}
