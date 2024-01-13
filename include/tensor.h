#ifndef TENSOR_H
#define TENSOR_H
#include<string>
#include<stdexcept>
#include<cuda_runtime.h>
#include<random>
#include<set>
#include<list>
#include<iostream>
#include<fstream>
#include <functional>
#include"./algebra.h"

//#include <curand.h>

const int NUM_THREADS = 128;

void cce(cudaError_t error, std::string msg="") {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error -- " << msg << ":" << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cce(std::string msg="") {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error -- " << msg << ":" << cudaGetErrorString(error);
        exit(EXIT_FAILURE);
    }
}

enum op_t { init_op, add_op, sub_op, mul_op, div_op, mm_op, index_op, loss_op, softmax_op, transpose_op, split_op, cat_op, gelu_op, layernorm_op, mask_op };
std::string op_to_string(op_t op) {
    switch (op) {
        case init_op: return "";
        case add_op: return "add";
        case sub_op: return "sub";
        case mul_op: return "mul";
        case div_op: return "div";
        case mm_op: return "mm";
        case index_op: return "index";
        case loss_op: return "loss";
        case softmax_op: return "softmax";
        case transpose_op: return "transpose";
        case split_op: return "split";
        case cat_op: return "cat";
        case gelu_op: return "gelu";
        case layernorm_op: return "layernorm";
        case mask_op: return "mask_op";
        default: return "unknown_op";
    }
}

class Tensor {
    int _dim;
    int _size;
    int* _shape;

    float* _cpu;
    float* _gpu;
    bool _onCPU;

    op_t _op;
    std::set<Tensor*> _children;

    bool _learnable;
    bool _persist;
    float* _gpu_grad;
    float* _cpu_grad;
    std::function<void()> _backward;


    public:
        std::string name;
        
        // Constructors
        Tensor(int dim, int* shape) {
            name = "unnamed";
            _op = init_op;

            _dim = dim;
            _shape = shape;

            int size = 1;
            for(int i = 0; i < _dim; i++) { size *= _shape[i]; }
            _size = size;
            _cpu = (float*)malloc(sizeof(float) * size);
            cce(cudaMalloc(&_gpu, sizeof(float) * size), "allocating tensor");
            cce(cudaMemset(_gpu, 0, sizeof(float)*size));
            _cpu_grad = (float*)malloc(sizeof(float) * size);
            memset(_cpu_grad, 0, sizeof(float)*size);
            cce(cudaMalloc(&_gpu_grad, sizeof(float) * size), "allocating tensor grad");
            cce(cudaMemset(_gpu_grad, 0, sizeof(float)*size));
            cudaDeviceSynchronize();
            _onCPU = true;
            _backward = []() {};
            _learnable = false;
            _persist = false;
        }

        Tensor(int dim, int* shape, std::set<Tensor*> children, op_t op) {
            name = "unnamed";
            _children = children;
            _op = op;

            _dim = dim;
            _shape = (int*)malloc(sizeof(int)*dim);
            for(int i = 0; i < dim; i++) { _shape[i] = shape[i]; }

            int size = 1;
            for(int i = 0; i < _dim; i++) { size *= _shape[i]; }
            _size = size;
            _cpu = (float*)malloc(sizeof(float)*size);
            cce(cudaMalloc(&_gpu, sizeof(float)*size), "allocating tensor");
            cce(cudaMemset(_gpu, 0, sizeof(float)*size));
            _cpu_grad = (float*)malloc(sizeof(float)*size);
            memset(_cpu_grad, 0, sizeof(float)*size);
            cce(cudaMalloc(&_gpu_grad, sizeof(float)*size), "allocating tensor grad");
            cce(cudaMemset(_gpu_grad, 0, sizeof(float)*size));
            _onCPU = true;
            _backward = []() {};
            _learnable = false;
            _persist = false;
        }

        Tensor(int dim, int* shape, float* data) {
            name = "unnamed";
            _op = init_op;

            _dim = dim;
            _shape = shape;
            int size = 1;
            for(int i = 0; i < _dim; i++) { size *= _shape[i]; }
            _size = size;
            _cpu = data;
            cce(cudaMalloc(&_gpu, sizeof(float) * size), "allocating tensor\'");
            cce(cudaMemset(_gpu, 0, sizeof(float)*size));
            _cpu_grad = (float*)malloc(sizeof(float) * size);
            memset(_cpu_grad, 0, sizeof(float)*size);
            cce(cudaMalloc(&_gpu_grad, sizeof(float) * size), "allocating tensor grad");
            cce(cudaMemset(_gpu_grad, 0, sizeof(float)*size));
            cce(cudaMemcpy(_gpu, _cpu, sizeof(float)*size, cudaMemcpyHostToDevice));
            _onCPU = true;
            _backward = []() {};
            _learnable = false;
            _persist = false;
        }

        Tensor(int dim, int* shape, float* cpu, float* gpu, bool onCPU) {
            name = "unnamed";
            _op = init_op;

            _dim = dim;
            _shape = shape;
            int size = 1;
            for(int i = 0; i < _dim; i++) { size *= _shape[i]; }
            _size = size;
            _cpu = cpu;
            _gpu = gpu;
            _cpu_grad = (float*)malloc(sizeof(float) * size);
            memset(_cpu_grad, 0, sizeof(float)*size);
            cce(cudaMalloc(&_gpu_grad, sizeof(float) * size), "allocating tensor grad");
            cce(cudaMemset(_gpu_grad, 0, sizeof(float)*size));
            _onCPU = onCPU;
            _backward = []() {};
            _learnable = false;
            _persist = false;
        }

        ~Tensor() {
            free(_shape);
            free(_cpu);
            cce(cudaFree(_gpu));
            free(_cpu_grad);
            cce(cudaFree(_gpu_grad));
        }

        static Tensor* fromFile(std::string filename) {
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

            Tensor* t = new Tensor(dim, shape);
            if(!(file.read(reinterpret_cast<char*>(t->cpu_ptr()), sizeof(int)*t->size()))) {
                std::cerr << "Error reading data while reading tensor from file: " << filename << std::endl;
                exit(EXIT_FAILURE);
            }

            return t;
        }

        static Tensor* tri(int blocksize) {
            int* shape = (int*)malloc(sizeof(int)*2);
            shape[0] = blocksize;
            shape[1] = blocksize;
            Tensor* t = new Tensor(2, shape);
            t->toGPU();
            t->name = "triangle";
            algebra::tri<<<(blocksize*blocksize/32+1)*32, 32>>>(t->gpu_ptr(), blocksize);
            return t;
        }

        // Getters
        int dim() const { return _dim; }
        int* shape() const { return _shape; }
        int size() const { return _size; }
        std::set<Tensor*> children() { return _children; }
        float* gpu_ptr() { 
            if(_onCPU) {
                this->toGPU();
                std::cerr << "Tensor: " << name << " is not on the GPU!! moving..." << "\n";
                //exit(EXIT_FAILURE);
            }
            return _gpu; 
        }
        float* cpu_ptr() { 
            if(!_onCPU) {
                this->toCPU();
                std::cerr << "Tensor: " << name << " is not on the CPU!! moving..." << "\n";
                //exit(EXIT_FAILURE);
            }
            return _cpu; 
        }
        float* gpu_grad_ptr() { 
            if(_onCPU) {
                this->toGPU();
                std::cerr << "Tensor: " << name << " is not on the GPU!! moving..." << "\n";
                //exit(EXIT_FAILURE);
            }
            return _gpu_grad; 
        }
        float* cpu_grad_ptr() { 
            if(!_onCPU) {
                this->toCPU();
                std::cerr <<  "Tensor: " << name << " is not on the CPU!! moving..." << "\n";
                //exit(EXIT_FAILURE);
            }
            return _cpu_grad; 
        }
        void add_child(Tensor* t) {
            this->_children.insert(t);
        }
        void set_op(op_t op) {
            this->_op = op;
        }

        void data_hasnan() {
            this->toCPU();
            for(int i = 0; i < this->size(); i++) { 
                if(isnan(this->cpu_ptr()[i])) {
                    std::cerr << "\n\n\n" << this->name << " has nan in data!!\n\n" << this->pretty();
                    exit(EXIT_FAILURE);
                }
            }
            this->toGPU();
        }

        bool grad_hasnan() {
            this->toCPU();
            for(int i = 0; i < this->size(); i++) { 
                if(isnan(this->cpu_grad_ptr()[i])) {
                    std::cerr << "\n\n\n--" << this->name << " has nan in grad!!\n\n" << this->pretty_grad(true);
                    std::cerr << "\n\n(data):\n\n" << this->pretty() << "\n\n";
                    return false;
                }
            }
            this->toGPU();
            return true;
        }

        // Set Backward Pass
        void learnable() {
            this->_learnable = true;
        }

        void persist() {
            this->_persist = true;
        }

        void learn(float lp) {
            if(_learnable) {
                float* delta;
                cce(cudaMalloc(&delta, sizeof(float)*this->size()));
                algebra::mul<<<(this->size()/128+1)*128, 128>>>(this->gpu_grad_ptr(), lp, delta, this->size());
                algebra::sub<<<(this->size()/128+1)*128, 128>>>(this->gpu_ptr(), delta, this->size(), this->size());
                cudaFree(delta);
            }
        }

        void clear_grad() {
            algebra::set<<<(this->size()/128+1)*128, 128>>>(this->gpu_grad_ptr(), 0.0f, this->size());
        }

        void gradient(const std::function<void()> gengrad) {
            this->_backward = gengrad;
        }

        bool generate_gradient() {
            this->_backward();
            return this->grad_hasnan();
        }

        void backward() {
            algebra::set<<<(this->size()/128+1)*128, 128>>>(this->gpu_grad_ptr(), 1.0f, this->size());

            std::list<Tensor*> topo;
            std::set<Tensor*> visited;

            std::function<void(Tensor*)> build_topo = [&build_topo, &topo, &visited](Tensor* t) {
                if(visited.find(t) == visited.end()) {
                    visited.insert(t);
                    for (Tensor* c : t->children()) {
                        build_topo(c);
                    }
                    topo.push_front(t);
                }
            };

            int i = 0;
            build_topo(this);
            for (Tensor* t : topo) {
                if(!t->generate_gradient()) {
                    std::cerr << "generating gradient...\n";
                    std::cerr << "\nComplete faulty gradient:\n\n";
                    std::cerr << this->pretty_grad(true);
                    std::cerr << "\nComplete faulty graph:\n\n";
                    std::cerr << this->pretty_graph(true);
                    exit(EXIT_FAILURE);
                }
            }
        }

        void optimize(float lp=0.0001) {
            std::set<Tensor*> visited;

            std::function<void(Tensor*)> build_set = [&build_set, &visited](Tensor* t) {
                if(visited.find(t) == visited.end()) {
                    visited.insert(t);
                    for (Tensor* c : t->children()) {
                        build_set(c);
                    }
                }
            };

            build_set(this);
            for (Tensor* t : visited) {
                t->learn(lp);
                t->clear_grad();
            }
        }

        void free_graph() {
            std::set<Tensor*> visited;

            std::function<void(Tensor*)> build_set = [&build_set, &visited](Tensor* t) {
                if(visited.find(t) == visited.end()) {
                    visited.insert(t);
                    for (Tensor* c : t->children()) {
                        build_set(c);
                    }
                }
            };

            build_set(this);
            for (Tensor* t : visited) {
                t->free_tensor();
            }
        }

        void free_tensor() {
            if(!_persist && !_learnable) { 
                delete this; 
            }
        }



        // Printing
        std::string pretty() {
            if(!_onCPU) {
                std::cerr << "Tensor must be on the cpu to print!\n";
                exit(EXIT_FAILURE);
            }
            int cursor = 0;
            return _pretty(&cursor, _dim, _shape, _cpu);
        }

        std::string pretty_shape() {
            std::string out = "[ ";
            //out += "shape ";
            for(int i = 0; i < this->dim(); i++) { out += std::to_string(this->shape()[i]) + " "; }
            out += "]";
            return out;
        }

        std::string pretty_graph(bool showgrad=false) {
            this->toCPU();

            std::list<Tensor*> topo;
            std::set<Tensor*> visited;

            std::function<void(Tensor*)> build_topo = [&build_topo, &topo, &visited](Tensor* t) {
                if(visited.find(t) == visited.end()) {
                    visited.insert(t);
                    for (Tensor* c : t->children()) {
                        build_topo(c);
                    }
                    topo.push_back(t);
                }
            };

            build_topo(this);

            std::string out = "";
            for (Tensor* t : topo) {
                out += op_to_string(t->_op) + "(" + t->pretty_shape() + ") : " + t->name + "\n";
                t->toCPU();
                out += t->pretty() + "\n\n";
                t->toGPU();
            }
            toGPU();
            return out;
        }

        std::string pretty_grad(bool showgrad=false) {
            toCPU();
            if(!_onCPU && showgrad) {
                std::cerr << "Gradient must be on the cpu to print!\n";
                exit(EXIT_FAILURE);
            }
            std::string step = "\n<" + name + "> = " + op_to_string(_op) + "( ";
            for(Tensor* c : _children) { step += c->name + " "; }
            step += ")\n";
            if(showgrad) {
                int cursor = 0;
                step += _pretty(&cursor, _dim, _shape, _cpu_grad);
            }
            for(Tensor* c : _children) { step += c->pretty_grad(showgrad); }
            return step;
        }

        // Device Management
        bool onCPU() {
            return _onCPU;
        }

        void toGPU() {
            if(_onCPU) {
                cce(cudaMemcpy(_gpu, _cpu, sizeof(float)*(this->size()), cudaMemcpyHostToDevice), "copy data to gpu");
                cce(cudaMemcpy(_gpu_grad, _cpu_grad, sizeof(float)*(this->size()), cudaMemcpyHostToDevice), "copy grad to gpu");
                _onCPU = false;
            }
        }

        void toCPU() {
            if(!_onCPU) {
                cce(cudaMemcpy(_cpu, _gpu, sizeof(float)*size(), cudaMemcpyDeviceToHost), "copy data to cpu");
                cce(cudaMemcpy(_cpu_grad, _gpu_grad, sizeof(float)*size(), cudaMemcpyDeviceToHost), "copy grad to cpu");
                _onCPU = true;
            }
        }

        // returns a "deep" copy of the tensor
        Tensor* copy() {
            float *data = (float*)malloc(sizeof(float)*size());
            int *shape = (int*)malloc(sizeof(int)*_dim);
            memccpy(data, _cpu, 0, sizeof(float)*size());
            memccpy(shape, _shape, 0, sizeof(int)*_dim);
            return new Tensor(_dim, shape, data);
        }

        Tensor** split(int splits, int dim=0) {
            if(dim > _dim) { 
                std::cerr << "Split dimension, " << dim << ", is greater than tensor dimension, " << _dim << "\n";
                exit(EXIT_FAILURE);
            }
            if(_shape[dim] < splits) {
                std::cerr << "There are more splits, " << splits << ", than there is space in dimension " << dim << ", " << _shape[dim] << "\n";
                exit(EXIT_FAILURE);
            }
            int batches = 1;
            for (int i = 0; i < dim; i++) { batches *= _shape[i]; }

            float* devy;
            cudaMalloc(&devy, sizeof(float)*this->size());

            algebra::split<<<(size()/32+1)*32, 32>>>(gpu_ptr(), devy, batches, splits, size());

            Tensor** ts = (Tensor**)malloc(sizeof(Tensor*)*splits);
            int splitsize = size() / splits;

            for(int i = 0; i < splits; i++) { 
                float* y = (float*)malloc(sizeof(float)*splitsize);
                int* output_shape = (int*)malloc(sizeof(int)*_dim);
                for(int j = 0; j < _dim; j++) { 
                    if(j == dim) output_shape[j] = _shape[j] / splits;
                    else output_shape[j] = _shape[j];
                }
                float* dev_i;
                cce(cudaMalloc(&dev_i, sizeof(float)*splitsize), "cuda alloc split");
                cce(cudaMemcpy(dev_i, devy+(i*splitsize), sizeof(float)*splitsize, cudaMemcpyDeviceToDevice));
                ts[i] = new Tensor(_dim, output_shape, y, dev_i, false);
                ts[i]->name = this->name + "[" + std::to_string(i) + "]";
                ts[i]->add_child(this);
                ts[i]->set_op(split_op);

                ts[i]->gradient([ts, this, batches, splits, i]() {
                    algebra::split_grad<<<(ts[i]->size()/128+1)*128, 128>>>(ts[i]->gpu_grad_ptr(), this->gpu_grad_ptr(), batches, splits, i, ts[i]->size());
                });
            }
            cce(cudaFree(devy));

            return ts;
        }

        static Tensor* cat(Tensor* a, Tensor* b, int dim=0) {
            if(a->dim() != b->dim()) {
                std::cerr << "Tensors have the same dimensions to be concatenated\n";
                exit(EXIT_FAILURE);
            }

            int batches = 1;
            int depth = 1;
            int* output_shape = (int*)malloc(sizeof(int)*a->dim());
            for(int i = 0; i < a->dim(); i++) {
                if(i == dim) {
                    output_shape[i] = a->shape()[i] + b->shape()[i];
                } else {
                    if(a->shape()[i] == b->shape()[i]) {
                        output_shape[i] = a->shape()[i];

                        if(i < dim) {
                            batches *= a->shape()[i];
                        } else {
                            depth *= a->shape()[i];
                        }
                    } else {
                        std::cerr << "Tensors have the same shape in every dimentsion other than the cat dimension to be concatenated\n";
                        exit(EXIT_FAILURE);
                    }
                }
            }

            Tensor* t = new Tensor(a->dim(), output_shape);
            t->add_child(a);
            t->add_child(b);
            t->set_op(cat_op);
            t->name = "cat(" + a->name + ", " + b->name + ")";

            t->toGPU();
            algebra::cat<<<(t->size()/64+1)*64, 64>>>(a->gpu_ptr(), b->gpu_ptr(), t->gpu_ptr(), batches, a->shape()[dim]*depth, b->shape()[dim]*depth);
            cce("cat");

            t->gradient([t, a, b, batches, depth, dim]() {
                algebra::cat_grad<<<(t->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), b->gpu_grad_ptr(), batches, a->shape()[dim]*depth, b->shape()[dim]*depth);
            });
            return t;
        }

        Tensor* view(int dim, int* shape) {
            int newsize = 1;
            for(int i = 0; i < dim; i++) { newsize *= shape[i]; }
            if(newsize == size()) {
                _dim = dim;
                free(_shape);
                _shape = shape;
                return this;
            }
            std::cerr << "Tensor view error: new shape must be the same size as the old shape, instead got new shape: [ ";
            for(int i = 0; i < dim; i++) { std::cerr << shape[i] << " "; }
            std::cerr << "] with old shape [ ";
            for(int i = 0; i < _dim; i++) { std::cerr << _shape[i] << " "; }
            std::cerr << "]\n";
            exit(EXIT_FAILURE); 
        }

        Tensor* transpose(int dim1, int dim2) {
            int height = 0;
            int width = 0;
            int depth = 1;

            int* output_shape = (int*)malloc(sizeof(int)*_dim);
            for(int i = 0; i < _dim; i++) { output_shape[i] = _shape[i]; }


            if(dim1 - dim2 == 1) {
                height = _shape[dim2];
                width = _shape[dim1];
                depth = 1;
                for(int i = dim1+1; i < this->dim(); i++) { depth *= this->shape()[i]; }
                output_shape[dim2] = width;
                output_shape[dim1] = height;
            } else if (dim1 - dim2 == -1) {
                height = _shape[dim1];
                width = _shape[dim2];
                depth = 1;
                for(int i = dim2+1; i < this->dim(); i++) { depth *= this->shape()[i]; }
                output_shape[dim1] = width;
                output_shape[dim2] = height;
            } else {
                std::cerr << "cry";
                exit(EXIT_FAILURE);
            }
            
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            Tensor* t = new Tensor(_dim, output_shape, children, transpose_op);
            free(output_shape);
            t->name = this->name + ".transpose()";
            t->toGPU();

            algebra::transpose<<<(size()/256+1)*256, 256>>>(_gpu, t->gpu_ptr(), width, height, depth, this->size());
            cudaDeviceSynchronize();

            t->gradient([t, this, width, height, depth]() {
                algebra::transpose<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), height, width, depth, this->size());
                cudaDeviceSynchronize();
            });

            return t;
        }

        // Distributions

        void zeros() {
            if(_onCPU) memset(_cpu, 0, sizeof(float)*size());
            else cce(cudaMemset(_gpu, 0, sizeof(float)*size()));
        }

        void set(float val) {
            if(_onCPU) throw std::runtime_error("Tensor needs to be on the GPU");
            algebra::set<<<(size() + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS>>>(_gpu, val, size());
            cudaDeviceSynchronize();
        }
        void uniform(float min=0.0, float max=1.0) {
            if(_onCPU) {
                std::default_random_engine generator;
                std::uniform_real_distribution<double> distribution(min, max);

                for(int i = 0; i < size(); i++) {
                    _cpu[i] = distribution(generator);
                }
            }
            else throw std::runtime_error("Tensor needs to be on the CPU");
            //else curandGenerateUniform(generator, _gpu, size());
        }

        void normal(/*curandGenerator_t  generator,*/float mu=0.0, float sigsquared=1.0) {
            if(_onCPU) {
                std::default_random_engine generator;
                std::normal_distribution<float> distribution(mu,sigsquared);
                for(int i = 0; i < size(); i++) {
                    _cpu[i] = distribution(generator);
                }
            }
            else throw std::runtime_error("Tensor needs to be on the CPU");
            //else curandGenerateNormal(generator, _gpu, size(), mu, sigsquared);
        }

        // Operations ---------------------------------------------------------
        // add
        Tensor* add(Tensor* a) {
            if(size() >= a->size()) {
                // a is being broadcast
                // check if sizes can be broadcast
                if(size() % a->size() != 0) {
                    std::cerr << "cannot broadcast shape: [ ";
                    for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
                    std::cerr << "] onto shape: [ ";
                    for(int i = 0; i < dim(); i++) { std::cerr << shape()[i] << " "; }
                    std::cerr << "]\n";
                    exit(EXIT_FAILURE);
                }

                std::set<Tensor*> children = std::set<Tensor*>();
                children.insert(this);
                children.insert(a);

                int output_dim = dim();
                int* output_shape = this->shape();
                Tensor* t = new Tensor(output_dim, output_shape, children, add_op);
                t->name = this->name + ".add(" + a->name + ")";

                t->toGPU();
                algebra::add<<<(this->size()/128+1)*128, 128>>>(this->gpu_ptr(), a->gpu_ptr(), t->gpu_ptr(), size(), a->size());
                cudaDeviceSynchronize();

                t->gradient([t, this, a]() {
                    algebra::add_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_ptr(), t->size(), this->size());
                    algebra::add_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), t->size(), a->size());
                    cce("addition grad");
                });
                
                return t;
            } else {
                // this is being broadcast
                // check if sizes can be broadcast
                if(a->size() % size() != 0) {
                    std::cerr << "cannot broadcast shape: [ ";
                    for(int i = 0; i < dim(); i++) { std::cerr << shape()[i] << " "; }
                    std::cerr << "] onto shape: [ ";
                    for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
                    std::cerr << "]\n";
                    exit(EXIT_FAILURE);
                }

                std::set<Tensor*> children = std::set<Tensor*>();
                children.insert(this);
                children.insert(a);

                int output_dim = a->dim();
                int* output_shape = a->shape();
                Tensor* t = new Tensor(output_dim, output_shape, children, add_op);
                t->name = this->name + ".add(" + a->name + ")";

                t->toGPU();
                algebra::add<<<(a->size()/128+1)*128, 128>>>(a->gpu_ptr(), this->gpu_ptr(), t->gpu_ptr(), a->size(), this->size());
                cudaDeviceSynchronize();

                t->gradient([t, this, a]() {
                    algebra::add_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), t->size(), this->size());
                    algebra::add_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), t->size(), a->size());
                    cudaDeviceSynchronize();
                });
                return t;
            }
        }

        Tensor* add(float a) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, add_op);
            t->name = this->name + ".add(" + std::to_string(a) + ")";
            t->toGPU();
            algebra::add<<<(this->size()/128+1)*128, 128>>>(this->gpu_ptr(), a, t->gpu_ptr(), this->size());
            cudaDeviceSynchronize();

            t->gradient([t, this, a]() {
                algebra::add_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), this->size());
                cudaDeviceSynchronize();
            });
            return t;
        }

        // sub
        Tensor* sub(Tensor* a) {
            if(this->size() >= a->size()) {
                // a is being broadcast
                // check if sizes can be broadcast
                if(size() % a->size() != 0) {
                    std::cerr << "cannot broadcast shape: [ ";
                    for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
                    std::cerr << "] onto shape: [ ";
                    for(int i = 0; i < dim(); i++) { std::cerr << shape()[i] << " "; }
                    std::cerr << "]\n";
                    exit(EXIT_FAILURE);
                }

                std::set<Tensor*> children = std::set<Tensor*>();
                children.insert(this);
                children.insert(a);

                int output_dim = dim();
                int* output_shape = shape();
                Tensor* t = new Tensor(output_dim, output_shape, children, sub_op);
                t->name = this->name + ".sub(" + a->name + ")";

                t->toGPU();
                algebra::sub<<<(size()/128+1)*128, 128>>>(gpu_ptr(), a->gpu_ptr(), t->gpu_ptr(), size(), a->size());
                cudaDeviceSynchronize();

                t->gradient([t, this, a]() {
                    algebra::add_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), t->size(), this->size());
                    algebra::sub_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), t->size(), a->size());
                    cudaDeviceSynchronize();
                });
                return t;
            }
            std::cerr << "cannot broadcast shape: [ ";
            for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
            std::cerr << "] onto shape: [ ";
            for(int i = 0; i < dim(); i++) { std::cerr << shape()[i] << " "; }
            std::cerr << "]\n";
            exit(EXIT_FAILURE);
        }

        Tensor* sub(float a) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, sub_op);
            t->name = this->name + ".sub(" + std::to_string(a) + ")";

            t->toGPU();
            algebra::sub<<<(size()/128+1)*128, 128>>>(gpu_ptr(), a, t->gpu_ptr(), this->size());
            cudaDeviceSynchronize();

            t->gradient([t, this, a]() {
                algebra::add_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), this->size());
                cudaDeviceSynchronize();
            });
        }

        // mul
        Tensor* mul(Tensor* a) {
            if(size() >= a->size()) {
                // a is being broadcast
                // check if sizes can be broadcast
                if(size() % a->size() != 0) {
                    std::cerr << "cannot broadcast shape: [ ";
                    for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
                    std::cerr << "] onto shape: [ ";
                    for(int i = 0; i < this->dim(); i++) { std::cerr << this->shape()[i] << " "; }
                    std::cerr << "]\n";
                    exit(EXIT_FAILURE);
                }

                std::set<Tensor*> children = std::set<Tensor*>();
                children.insert(this);
                children.insert(a);

                int output_dim = this->dim();
                int* output_shape = this->shape();
                Tensor* t = new Tensor(output_dim, output_shape, children, mul_op);
                t->name = this->name + ".mul(" + a->name + ")";

                t->toGPU();
                algebra::mul<<<(this->size()/128+1)*128, 128>>>(this->gpu_ptr(), a->gpu_ptr(), t->gpu_ptr(), this->size(), a->size());
                cudaDeviceSynchronize();

                t->gradient([t, this, a]() {
                    algebra::mul_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a->gpu_ptr(), this->size(), a->size());
                    algebra::mul_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), this->gpu_ptr(), a->size(), this->size());
                    cudaDeviceSynchronize();
                });
                return t;

            } else {
                // this is being broadcast
                // check if sizes can be broadcast
                if(a->size() % this->size() != 0) {
                    std::cerr << "cannot broadcast shape: [ ";
                    for(int i = 0; i < this->dim(); i++) { std::cerr << this->shape()[i] << " "; }
                    std::cerr << "] onto shape: [ ";
                    for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
                    std::cerr << "]\n";
                    exit(EXIT_FAILURE);
                }

                std::set<Tensor*> children = std::set<Tensor*>();
                children.insert(this);
                children.insert(a);

                int output_dim = a->dim();
                int* output_shape = a->shape();
                Tensor* t = new Tensor(output_dim, output_shape, children, mul_op);
                t->name = this->name + ".mul(" + a->name + ")";

                t->toGPU();
                algebra::mul<<<(a->size()/128+1)*128, 128>>>(a->gpu_ptr(), this->gpu_ptr(), t->gpu_ptr(), a->size(), this->size());
                cudaDeviceSynchronize();

                t->gradient([t, this, a]() {
                    algebra::mul_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a->gpu_ptr(), this->size(), a->size());
                    algebra::mul_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), this->gpu_ptr(), a->size(), this->size());
                    cudaDeviceSynchronize();
                });
                return t;
            }
        }

        Tensor* mul(float a) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, mul_op);

            t->toGPU();
            algebra::mul<<<(this->size()/128+1)*128, 128>>>(this->gpu_ptr(), a, t->gpu_ptr(), this->size());
            cudaDeviceSynchronize();
            t->name = this->name + ".mul(" + std::to_string(a) + ")";

            t->gradient([t, this, a]() {
                algebra::mul_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a, this->size());
                cudaDeviceSynchronize();
            });
            return t;
        }

        // div
        Tensor* div(Tensor* a) {
            if(this->size() >= a->size()) {
                // a is being broadcast
                // check if sizes can be broadcast
                if(this->size() % a->size() != 0) {
                    std::cerr << "cannot broadcast shape: [ ";
                    for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
                    std::cerr << "] onto shape: [ ";
                    for(int i = 0; i < this->dim(); i++) { std::cerr << this->shape()[i] << " "; }
                    std::cerr << "]\n";
                    exit(EXIT_FAILURE);
                }

                std::set<Tensor*> children = std::set<Tensor*>();
                children.insert(this);
                children.insert(a);

                int output_dim = this->dim();
                int* output_shape = this->shape();
                Tensor* t = new Tensor(output_dim, output_shape, children, div_op);
                t->name = this->name + ".div(" + a->name + ")";

                t->toGPU();
                algebra::div<<<(size()/128+1)*128, 128>>>(this->gpu_ptr(), a->gpu_ptr(), t->gpu_ptr(), this->size(), a->size());
                cudaDeviceSynchronize();

                t->gradient([t, this, a]() {
                    algebra::dividend_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a->gpu_ptr(), this->size(), a->size());
                    algebra::divisor_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), a->gpu_ptr(), this->gpu_ptr(), t->size(), a->size());
                    cudaDeviceSynchronize();
                });
                return t;
            }
            std::cerr << "cannot broadcast shape: [ ";
            for(int i = 0; i < a->dim(); i++) { std::cerr << a->shape()[i] << " "; }
            std::cerr << "] onto shape: [ ";
            for(int i = 0; i < this->dim(); i++) { std::cerr << this->shape()[i] << " "; }
            std::cerr << "]\n";
            exit(EXIT_FAILURE);
        }

        Tensor* div(float a) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, div_op);
            t->name = this->name + ".div(" + std::to_string(a) + ")";

            t->toGPU();
            algebra::div<<<(size()/128+1)*128, 128>>>(this->gpu_ptr(), a, t->gpu_ptr(), this->size());
            cce("div");

            t->gradient([t, this, a]() {
                algebra::dividend_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a, this->size());
                cce("div grad");
            });
            return t;
        }

        Tensor* gelu() {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = (int*)malloc(sizeof(int)*this->dim());
            for(int i = 0; i < this->dim(); i++) { output_shape[i] = this->shape()[i]; }
            Tensor* t = new Tensor(output_dim, output_shape, children, gelu_op);
            t->name = this->name + ".gelu()";
            free(output_shape);
            t->toGPU();

            algebra::gelu<<<(t->size()/32+1)*32, 32>>>(this->gpu_ptr(), t->gpu_ptr(), t->size());
            cce("gelu");

            t->gradient([t, this]() {
                algebra::gelu_grad<<<(this->size()/32+1)*32, 32>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), this->gpu_ptr(), this->size());
                cce("gelu grad");
            });

            return t;
        }

        Tensor* mm(Tensor* a) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);
            children.insert(a);

            int output_dim = this->dim();
            int* output_shape = (int*)malloc(sizeof(int)*this->dim());
            for(int i = 0; i < this->dim(); i++) { output_shape[i] = this->shape()[i]; }
            output_shape[output_dim-1] = a->shape()[a->dim()-1];
            Tensor* t = new Tensor(output_dim, output_shape, children, mm_op);
            t->name = this->name + ".mm(" + a->name + ")";
            free(output_shape);
            
            t->toGPU();
            int m = this->shape()[this->dim()-2];
            int n = a->shape()[a->dim()-1];
            int inner = this->shape()[this->dim()-1];
            if(inner != a->shape()[a->dim()-2]) {
                std::cerr << "matmul must have matching inner dimentions: " << inner << " and " << a->shape()[a->dim()-2] << " and shapes: ";
                std::cerr << "this: " << this->pretty_shape() << ", and a: " << a->pretty_shape() << "\n";
            }
            int this_batches = this->size()/inner/m;
            int a_batches = a->size()/inner/n;
            bool broadcasting;
            if (this_batches == a_batches) broadcasting = false;
            else if (a_batches == 1) broadcasting = true;
            else {
                std::cerr << "bad mm shapes, can't batch or broadcast: " << this->shape() << " @ " << a->shape() << "\n";
                exit(EXIT_FAILURE);
            }
            //std::cerr << this->pretty_shape() << " mm " << a->pretty_shape() << "\n";
            //std::cerr << "m: " << m << ", inner: " << inner << ", n: " << n << "\n";

            algebra::mm<<<(t->size()/32+1)*32, 32>>>(this->gpu_ptr(), a->gpu_ptr(), t->gpu_ptr(), m, inner, n, this_batches, t->size(), broadcasting);
            cce("mm");

            t->gradient([t, this, a, m, n, inner, this_batches, broadcasting]() {
                algebra::mm_in_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a->gpu_ptr(), m, inner, n, this_batches, this->size(), broadcasting);
                algebra::mm_out_grad<<<(a->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), a->gpu_grad_ptr(), this->gpu_ptr(), m, inner, n, this_batches, a->size(), broadcasting);
                cce("mm grad");
            });

            return t;
        }

        Tensor* mask(Tensor* m) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);
            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, mask_op);
            t->name = this->name + ".mask(" + m->name + ")";
            t->toGPU();

            if(this->size() % m->size() != 0) {
                std::cerr << "mask shape must go evenly into tensor shape, instead got tensor of " << this->pretty_shape() << " and mask of " << m->pretty_shape() << "\n";
                exit(EXIT_FAILURE);
            }
            int batches = this->size()/m->size();

            algebra::mask<<<(this->size()/32+1)*32, 32>>>(this->gpu_ptr(), t->gpu_ptr(), m->gpu_ptr(), -INFINITY, batches, this->size());
            cce("mask");

            t->gradient([t, this, m, batches]() {
                algebra::mask_grad<<<(this->size()/32+1)*32, 32>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), m->gpu_ptr(), batches, this->size());
                cce("maks grad");
            });

            return t;
        }

        Tensor* index(int* a) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim;
            int* output_shape;
            if(this->dim() == 1) {
                output_dim = 1;
                output_shape = (int*)malloc(sizeof(int));
                output_shape[0] = 1;
            } else {
                output_dim = this->dim()-1;
                output_shape = (int*)malloc(sizeof(int)*output_dim);
                for(int i = 0; i < this->dim()-1; i++) { output_shape[i] = this->shape()[i]; }
            }
            Tensor* t = new Tensor(output_dim, output_shape, children, index_op);
            t->name = this->name + ".index()";
            free(output_shape);
            
            t->toGPU();

            algebra::index<<<(t->size()/128+1)*128, 128>>>(a, this->gpu_ptr(), t->gpu_ptr(), this->shape()[this->dim()-1], t->size());
            cce("index");

            t->gradient([t, this, a]() {
                algebra::index_grad<<<(this->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), a, this->shape()[this->dim()-1], t->size());
                cce("index grad");
            });

            return t;
        }

        Tensor* loss() {
            int batches;
            if(this->dim() == 1) {
                batches = 1;
            } else if(this->dim() == 2) {
                batches = this->shape()[0];
            } else {
                std::cerr << "loss expects a tensor of at most two dimenstions\n";
            }

            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);
            
            int output_dim = 1;
            int* output_shape = (int*)malloc(sizeof(int));
            output_shape[0] = batches;

            Tensor* t = new Tensor(output_dim, output_shape, children, loss_op);
            t->name = this->name + ".loss()";
            free(output_shape);
            t->toGPU();

            algebra::loss<<<(t->size()/32+1)*32, 32>>>(this->gpu_ptr(), t->gpu_ptr(), this->size()/batches, batches);
            cce("loss");

            t->gradient([t, this, batches]() {
                algebra::loss_grad<<<(this->size()/32+1)*32, 32>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), this->gpu_ptr(), this->size()/batches, t->size());
                cce("loss grad");
            });

            return t;
        }

        Tensor* softmax(int dim=0) {
            int batches = 1;
            int batchsize = 1;
            for(int i = 0; i < this->dim(); i++) {
                if(i <= dim) batches *= this->shape()[i];
                else batchsize *= this->shape()[i];
            }
            return this->softmax(batchsize, batches);
        }

        Tensor* softmax(int batchsize, int batches) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, softmax_op);
            t->name = this->name + ".softmax()";
            t->toGPU();

            algebra::softmax<<<(t->size()/32+1)*32, 32>>>(this->gpu_ptr(), t->gpu_ptr(), batchsize, batchsize*batches);
            cce("softmax");

            t->gradient([t, this, batchsize, batches]() {
                algebra::softmax_grad<<<(this->size()/32+1)*32, 32>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), this->gpu_ptr(), batchsize, batchsize*batches);
                cce("softmax grad");
            });

            return t;
        }

        Tensor* layernorm(int layersize) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, layernorm_op);
            t->name = this->name + ".layernorm()";
            t->toGPU();
            int batches = t->size() / layersize;
            algebra::layernorm<<<(t->size()/128+1)*128, 128>>>(this->gpu_ptr(), t->gpu_ptr(), layersize, t->size());
            cce("layernorm");

            t->gradient([t, this, layersize]() {
                algebra::layernorm_grad<<<(t->size()/128+1)*128, 128>>>(t->gpu_grad_ptr(), this->gpu_grad_ptr(), this->gpu_ptr(), layersize, t->size());
                cce("layernorm grad");
            });

            return t;
        }

        Tensor* sum(int batchsize, int batches) {
            std::set<Tensor*> children = std::set<Tensor*>();
            children.insert(this);

            int output_dim = this->dim();
            int* output_shape = this->shape();
            Tensor* t = new Tensor(output_dim, output_shape, children, softmax_op);
            t->name = this->name + ".softmax()";
            t->toGPU();

            algebra::softmax<<<batchsize*batches, batchsize>>>(this->gpu_ptr(), t->gpu_ptr(), batchsize, batchsize*batches);

            t->gradient([t, this, batchsize, batches]() {
                // no gradient yet
            });

            return t;
        }


    private:
        std::string _pretty(int *cursor, int dim, int* shape, float* data) {
            std::string out = ""; 
            if (dim == 1) {
                out += "[";
                for (int i = 0; i < shape[0] - 1; i++) {
                   out += std::to_string(data[*cursor + i]) + ", ";
                }
                out += std::to_string(data[*cursor + shape[0] - 1]) + "]\n";
                *cursor += shape[0];
                return out;
            }
            if (dim > 1) {
                out += "[";
                for(int i = 0; i < shape[0] - 1; i++) {
                    out += _pretty(cursor, dim-1, shape+1, data) + ", ";
                }
                out += _pretty(cursor, dim-1, shape+1, data) + "]";
                return out;
            }
            return "";
        }

};

#endif