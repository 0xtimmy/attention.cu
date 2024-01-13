#ifndef ALGEBRA_H
#define ALGEBRA_H

#include<cmath>
#include<cuda_runtime.h>

namespace algebra {
    // Unary ----------------------------------------------------------------------
    __global__ void label_index(float* x, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size) {
            x[idx] = idx;
        }
    }

    __global__ void max_idx(float* x, int* y, int batchsize, int batches) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < batches) {
            float max = x[idx*batchsize];
            int max_idx = 0;
            for (int i = 1; i < batchsize; i++) { 
                if(x[idx*batchsize + i] > max) {
                    max = x[idx*batchsize + i];
                    max_idx = i;
                }
            }
            y[idx] = max_idx;
        }
    }

    __global__ void set(float* x, float val, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size) {
            x[idx] = val;
        }
    }


    __global__ void exp(float* x, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            y[idx] = expf(x[idx]);
        }
    }

    __global__ void _log(float*x, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            x[idx] = log(x[idx]);
        }
    }

    __global__ void embedding(int* x, float* embeddings, float* y, int embedding_size, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < size) {
            for(int i = 0; i < embedding_size; i++) {
                y[idx * embedding_size + i] = embeddings[x[idx] * embedding_size + i];
            }
        }
    }

    __global__ void index(int* x, float* embeddings, float* y, int batchsize, int batches) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < batches) {
            y[idx] = embeddings[x[idx] + idx*batchsize];
        }
    }

    __global__ void index_grad(float* t, float* x_grad, int* i, int batchsize, int batches) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(idx < batchsize * batches) {
            x_grad[idx] += (batch_idx == i[batch]) ? t[batch] : 0.0f;
        }
    }

    __global__ void onehot(int* x, float* y, int num_classes, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 

        if(idx < size) {
            for(int i = 0; i < num_classes; i++) {
                y[idx*num_classes + i] = (i == x[idx]) ? 1.0f : 0.0f;
            }
        }
    }

    __global__ void transpose(float* x, float* y, int width, int height, int depth, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int local_size = width * height * depth;
        int local = idx / local_size;
        int local_idx = idx % local_size;
        int row = (local_idx / depth / width) % height;
        int col = (local_idx / depth) % width;
        int deep = idx % depth;

        if(idx < size) y[local * local_size + col * height * depth + row * depth + deep] = x[idx];
    }

    //gelu = 0.5*x*(1 +tanh(sqrt(2/pi)*(x + 0.0044715x^3)))
    __global__ void gelu(float* x, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float pi = 3.1415;
        if(idx < size) y[idx] = 0.5 * x[idx] * (1 + tanhf(sqrtf(2 / pi) * (x[idx] + 0.0044715 * x[idx] * x[idx] * x[idx])));
    }
    __global__ void gelu_grad(float* t, float* x_grad, float* x, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float pi = 3.1415;
        if(idx < size) {
            float grad = 0.5 * x[idx] / coshf(powf(sqrtf(2 / pi)*(x[idx] + 0.0044715 * x[idx] * x[idx] * x[idx]), 2));
            grad *= sqrtf(2 / pi)*(1 + 3*0.0044715);
            grad += 0.5 * (1 + tanhf(sqrtf(2 / pi) * (x[idx] + 0.0044715 * x[idx] * x[idx] * x[idx])));
            x_grad[idx] += t[idx] * grad;
        }
    }

    // Binary ---------------------------------------------------------------------
    // add
    __global__ void add(float* a, float* b, float* y, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] + b[idx % sizeb];
        }
    }

    __global__ void add_grad(float* t, float* a, int sizet, int sizea) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < sizea) {
            for(int i = 0; i < sizet/sizea; i++) { a[idx] = t[i*sizea + (idx % sizea)]; }
        }
    }

    __global__ void add_grad(float* t, float* a, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size) {
            a[idx] += t[idx];
        }
    }

    __global__ void add(float* x, float a, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            y[idx] = x[idx] + a;
        }
    }

    __global__ void add(float* x, float a, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            x[idx] = x[idx] + a;
        }
    }
    
    // sub
    __global__ void sub(float* a, float* b, float* y, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] - b[idx % sizeb];
        }
    }

    __global__ void sub_grad(float* t, float* a, int sizet, int sizea) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < sizea) {
            for(int i = 0; i < sizet/sizea; i++) { a[idx] -= t[i*sizea + (idx % sizea)]; }
        }
    }

    __global__ void sub(float* x, float a, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            y[idx] = x[idx] - a;
        }
    }

    // for applying learning only
    __global__ void sub(float* a, float* b, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            a[idx] = a[idx] - b[idx % sizeb];
        }
    }

    // mul
    __global__ void mul(float* a, float* b, float* y, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] * b[idx % sizeb];
        }
    }

    __global__ void mul_grad(float* t, float* a, float* b, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(idx < sizea || idx < sizeb) {
            if(sizea >= sizeb) a[idx] += b[idx % sizeb]*t[idx];
            else for(int i = 0; i < sizeb/sizea; i++) { a[idx] += b[i*sizea + (idx % sizea)] * t[i*sizea + (idx % sizea)]; }
        }
    }

    __global__ void mul(float* x, float a, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            y[idx] = x[idx] * a;
        }
    }

    __global__ void mul_grad(float* t, float* a, float b, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(idx < size) {
            a[idx] += b*t[idx];
        }
    }


    // div
    __global__ void div(float* a, float* b, float* y, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < sizea) {
            y[idx] = a[idx] / b[idx % sizeb];
        }
    }

    __global__ void dividend_grad(float* t, float* a, float* b, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int scale = sizea/sizeb;
        
        if(idx < sizea) {
            a[idx] += (1.0f/b[idx % scale])*t[idx];
        }
    }

    __global__ void dividend_grad(float* t, float* a, float b, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if(idx < size) {
            a[idx] += (1.0f/b)*t[idx];
        }
    }

    __global__ void divisor_grad(float* t, float* grad_a, float* a, float* b, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int scale = sizea/sizeb;
        
        if(idx < sizea) {
            for(int i = 0; i < sizeb/sizea; i++) { grad_a[idx] += b[i*sizeb + (idx % sizeb)] * t[i*sizeb + (idx % sizeb)]; }
        }
    }

    __global__ void div(float* x, float* a, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int scale = sizea/sizeb;
        if (idx < sizea) {
            x[idx] = x[idx] / a[idx/scale];
        }
    }

    __global__ void div(float* x, float a, float* y, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            y[idx] = x[idx] / a;
        }
    }

    __global__ void div(float* x, float a, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            x[idx] = x[idx] / a;
        }
    }

    // Matrix ---------------------------------------------------------------------
    /*
    __global__ void linear(float* x, float* w, float* b, float* y, int in_features, int out_features, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int row = idx / out_features;
        int col = idx % out_features;

        if (idx < size) {
            float value = 0;
            for (int i = 0; i < in_features; ++i) {
                value += x[row * in_features + i] * w[i * in_features + col];
            }
            y[idx] = value + b[col];
        }
    }
    */

    __global__ void mm(float* a, float* b, float* y, int m, int inner, int n, int batches, int output_size, bool broadcasting) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchsize = output_size / batches;
        int batch = idx / batchsize;
        int a_batch_offset = batch * (m*inner);
        int b_batch_offset = broadcasting ? 0 : batch * (n*inner);

        int row = (idx % batchsize) / n;
        int col = idx % n;

        if (idx < output_size) {
            float dot = 0;
            for (int i = 0; i < inner; i++) {
                dot += a[row * inner + i + a_batch_offset] *  b[i * n + col + b_batch_offset];
            }
            y[idx] = dot;
        }
    }

    __global__ void tri(float* y, int blocksize) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int row = idx / blocksize;
        int col = idx % blocksize;
        if(idx < blocksize*blocksize) {
            if(col > row) y[idx] = 0.0;
            else y[idx] = 1.0;
        }
    }

    __global__ void mask(float* x, float* y, float* mask, float fill, int batches, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchsize = size / batches;
        int batch_idx = idx % batchsize;

        if(idx < size) {
            if(mask[batch_idx] == 0.0)  y[idx] = fill;
            else y[idx] = x[idx];
        }
    }

    __global__ void mask_grad(float* t, float* x_grad, float* mask, int batches, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchsize = size / batches;
        int batch_idx = idx % batchsize;

        if(idx < size) {
            if(mask[batch_idx] == 0.0)  x_grad[idx] = 0.0;
            else x_grad[idx] = t[idx];
        }
    }

    /* ERROR -- there's something up with the first/second half of each gradient */

    // M_first x M_second
    __global__ void mm_in_grad(float* t_grad, float* a_grad, float* b, int m, int inner, int n, int batches, int size, bool broadcasting) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchsize = size / batches;
        int batch = idx / batchsize;
        int a_batch_offset = batch * (m*n);
        int b_batch_offset = broadcasting ? 0 : batch * (n*inner);

        int row = (idx % batchsize) / inner;
        int col = idx % inner;

        if (idx < size) {
            if(broadcasting) {
                float sum = 0;
                for(int i = 0; i < n; i++) {
                    //sum += b[col][i] * t_grad[row][i];
                    sum += b[col*n + i] * t_grad[row*n + i + a_batch_offset];
                }
                a_grad[idx] += sum;
            } else {
                float sum = 0;
                for(int i = 0; i < n; i++) {
                    //sum += b[col][i] * t_grad[row][i];
                    sum += b[col*n + i + b_batch_offset] * t_grad[row*n + i + a_batch_offset];
                }
                a_grad[idx] += sum;
            }
        }
    }

    __global__ void mm_out_grad(float* t_grad, float* b_grad, float* a, int m, int inner, int n, int batches, int size, bool broadcasting) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size) {
            if(broadcasting) {
                int batchsize = size / batches;
                int row = idx / n;
                int col = idx % n;

                float sum = 0;
                for(int batch = 0; batch < batches; batch++) {
                    int t_batch_offset = batch * (m*n);
                    int a_batch_offset = batch * (m*inner);
                    for(int i = 0; i < m; i++) {
                        //sum += a[i][row] * t_grad[i][col];
                        sum += a[i*inner + row + a_batch_offset] * t_grad[i*n + col + t_batch_offset];
                    }
                }
                b_grad[idx] += sum; 
            } else {
                int batchsize = size / batches;
                int row = (idx % batchsize) / n;
                int col = idx % n;
                int batch = idx / batchsize;
                int b_batch_offset = batch * (m*n);
                int a_batch_offset = batch * (m*inner);

                float sum = 0;
                for(int i = 0; i < m; i++) {
                    //sum += a[i][row] * t_grad[i][col];
                    sum += a[i*inner + row + a_batch_offset] * t_grad[i*n + col + b_batch_offset];
                }
                b_grad[idx] += sum; 
            }
        }
    }

    __global__ void product(float* x, float* y, int batchsize, int batches) {
        extern __shared__ float prod[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        
        if(tid < batchsize) prod[tid] = x[idx];
        __syncthreads();

        for(int stride = 1; stride <= batchsize; stride *= 2) {
            if(tid % stride*2 == 0 && (tid/stride + stride) < batchsize) prod[tid/stride] *= prod[tid/stride + stride];
            __syncthreads();
        }

        y[0] = prod[0];
    }

    __global__ void loss(float* x, float* y, int batchsize, int batches) {
        //extern __shared__ float prod[];

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < batches) {
            float ll = 0.0;
            for(int i = 0; i < batchsize; i++) {
                ll += logf(x[i]);
            }
            y[idx] = -ll/batchsize;
        }
    }

    __global__ void loss_grad(float* t, float* x_grad, float* x, int batchsize, int batches) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(idx < batchsize*batches) {
            x_grad[idx] += t[batch] * (1 - x[batch*batchsize + batch_idx]) / batchsize;
        }
    }

    __global__ void sum(float* x, float* y, int batchsize, int size) {
        extern __shared__ float _sum[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int lid = idx % batchsize;
        int layer = idx / batchsize;

        if(idx < size) {
            _sum[lid] = x[idx];
        }

        for(int stride = 1; stride <= batchsize; stride *= 2) {
            if(lid % stride*2 == 0 && (lid/stride + stride) < batchsize) _sum[lid/stride] += _sum[lid/stride + stride];
            __syncthreads();
        }

        if(lid == 0 && idx < size) y[layer] = _sum[0];
    }

    // Normalization --------------------------------------------------------------
    __global__ void softmax(float* x, float* y, int batchsize, int size) {
        //extern __shared__ float _sum[];

        // channles must be reduced to calculate the sum; 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;

       if(idx < size) {
        float sum = 0;
        for(int i = 0; i < batchsize; i++) {
                sum += expf(x[batch*batchsize + i]);
        }

        y[idx] = expf(x[idx]) / (sum + 0.000001);
       }
       
    }

    __global__ void softmax_grad(float* t, float* x_grad, float* x, int batchsize, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(idx < size) {
            float exp_x = expf(x[idx]);
            float _sum = 0;
            for(int i = 0; i < batchsize; i++) {
                _sum += expf(x[batch*batchsize+i]);
            }

            for(int i = 0; i < batchsize; i++) {
                if(i == batch_idx) {
                    x_grad[idx] += t[batch*batchsize+i] * ((_sum-exp_x)*exp_x)/(_sum*_sum + 0.000001);
                } else {
                    x_grad[idx] -= t[batch*batchsize+i] * (expf(x[batch*batchsize+i])*exp_x)/(_sum*_sum + 0.000001);
                }
            }
        }
    }

    /*
    __global__ void softmax_grad(float* t, float* x_grad, float* x, int batchsize, int size) {
        extern __shared__ float _sum[];

        // channles must be reduced to calculate the sum; 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int lid = idx % batchsize;
        int layer = idx / batchsize;

        float x_exp = 0;
        if(idx < size) {
            x_exp = __expf(x[idx]);;
            _sum[lid] = x_exp;
        }

        __syncthreads();

        for(int stride = 1; stride <= batchsize; stride *= 2) {
            if(lid % stride*2 == 0 && (lid/stride + stride) < batchsize) _sum[lid/stride] += _sum[lid/stride + stride];
            __syncthreads();
        }

        if(idx < size) x_grad[idx] += t[idx] * 1/(sum[0] + 0.000001);
    }
    */

    __global__ void layernorm(float* x, float* y, int layersize, int size) {
        //extern __shared__ float _sum[];

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //int tid = threadIdx.x;
        int layer = idx / layersize;

        //if(idx < size) _sum[tid] = x[idx];

        /*
        __syncthreads();

        for(int stride = 1; stride <= layersize; stride *= 2) {
            if(tid % stride*2 == 0 && (tid/stride + stride) < layersize) _sum[tid/stride] += _sum[tid/stride + stride];
            __syncthreads();
        }
        */
       if(idx < size) {
            float avg = 0;
            for(int i = 0; i < layersize; i++) {
                avg += x[layer*layersize + i];
            }

            avg = avg / layersize;
            //(idx < size) _sum[tid] = powf(x[idx] - avg, 2.0);

            float _var = 0;
            for(int i = 0; i < layersize; i++) {
                _var += powf(x[layer*layersize + i] - avg, 2.0);
            }
            _var = _var / layersize;

            /*
            __syncthreads();


            for(int stride = 1; stride <= layersize; stride *= 2) {
                if(tid % stride*2 == 0 && (tid/stride + stride) < layersize) _sum[tid/stride] += _sum[tid/stride + stride];
                __syncthreads();
            }
            */

            //if(idx < size && tid == 0) var[batch] = _sum[0] / layersize;

            y[idx] = (x[idx] - avg) / sqrtf(fabsf(_var) + 0.000001);
       }
    }

    __global__ void layernorm_grad(float* t, float* x_grad, float* x, int layersize, float size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int layer = idx / layersize;
        int layer_idx = idx % layersize;
        if(idx < size) {
            float avg = 0;

            for(int i = 0; i < layersize; i++) {
                avg += x[layer*layersize + i];
            }

            avg = avg / layersize;
            //(idx < size) _sum[tid] = powf(x[idx] - avg, 2.0);

            float var = 0;
            for(int i = 0; i < layersize; i++) {
                var += powf(x[layer*layersize + i] - avg, 2.0);
            }
            var = var / layersize;
            float std = sqrtf(fabsf(var));

            x_grad[idx] = t[idx] * 1.0f/std;

            /*
            for(int i = 0; i < layersize; i++) {
                if(i == layer_idx) x_grad[idx] += t[layer*layersize+i] * (1/(std + 0.000001) * ((i == layer_idx ? 1 : 0) - (1/layersize)) - ((x[layer*layersize+i]-avg)*(x[idx]-avg))/(powf(std+0.000001, 2)*(layersize*std)));
            }
            */
        }
    }

    // Type Conversion
    __global__ void inttofloat(int* x, float* y, float size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        y[idx] = __int2float_rd(x[idx]);
    }

    __global__ void split(float* x, float* y, int batches, int splits, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int splitsize = size / splits;
        int split = idx / splitsize;

        int rowsize = size / batches;
        int split_rowsize = splitsize / batches;
        int split_row_idx = idx % split_rowsize;
        int split_row = (idx / split_rowsize) % batches; 

        if(idx < size) y[idx] = x[split_row*rowsize + split*split_rowsize + split_row_idx];
    }

    __global__ void split_grad(float* t, float* x_grad, int batches, int splits, int split, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int splitsize = size;
        int totalsize = size * splits;
        int rowsize = totalsize / batches;

        int split_rowsize = size / batches;
        int split_row_idx = idx % split_rowsize;
        int split_row = (idx / split_rowsize) % batches; 

        //if(idx < size) x_grad[split_row*rowsize + split*split_rowsize + split_row_idx] += t[idx];
        if(idx < size) x_grad[split_row*rowsize + split*split_rowsize + split_row_idx] = split_row;
    }

    __global__ void cat(float* a, float* b, float* y, int batches, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int batchsize = sizea+sizeb;

        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(batch_idx < sizea) {
            y[idx] = a[batch*sizea + batch_idx];
        } else {
            y[idx] = b[batch*sizeb + (batch_idx - sizea)];
        }
    }

    __global__ void cat_grad(float* t, float* a_grad, float* b_grad, int batches, int sizea, int sizeb) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int batchsize = sizea+sizeb;

        int batch = idx / batchsize;
        int batch_idx = idx % batchsize;

        if(batch_idx < sizea) {
            a_grad[batch*sizea + batch_idx] = t[idx];
        } else {
            b_grad[batch*sizeb + (batch_idx - sizea)] = t[idx];
        }
    }

}

#endif