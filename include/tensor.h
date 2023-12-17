#ifndef TENSOR_H
#define TENSOR_H
#include<string>

class Tensor {
    int _dim;
    int* _shape;
    float* _data;

    public:
        Tensor(int dim, int* shape, float* data) {
            _dim = dim;
            _shape = shape;
            _data = data;
        }

        ~Tensor() {
            free(_shape);
            free(_data);
        }

        int dim() const { return _dim; }
        int* shape() const { return _shape; }
        int size(int d=0) const {
            if(d >= _dim) return 0;
            int out = 1;
            for(int i = d; i < _dim; i++) {
                out *= _shape[i];
            }
            return out;
        }

        std::string pretty() {
            int cursor = 0;
            return _pretty(&cursor, _dim, _shape, _data);
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