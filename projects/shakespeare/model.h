#include<set>
#include<map>
#include<string>
#include<iostream>
#include<memory>
#include"../../include/tensor.h"
#include"../../include/algebra.h"

class Module {
    public:
        std::set<std::string> names;
        std::map<std::string, Tensor*> parameters;

        void loadparameter(std::string name, Tensor* param) {
            names.insert(name);
            parameters[name] = param;
        }

        Tensor* getparameter(std::string name) {
            return parameters[name];
        }

        void toGPU() {
            for (std::string name : names) {
                parameters[name]->toGPU();
            }
        }
};

class Embedding : public Module {

    int _num_embeddings;
    int _embedding_dim;

    public:
        std::string name;
        Embedding(int num_embeddings, int embedding_dim) {
            name = "embedding";
            _num_embeddings = num_embeddings;
            _embedding_dim = embedding_dim;
            int* shape = (int*)malloc(sizeof(int)*2);
            shape[0] = num_embeddings;
            shape[1] = embedding_dim;

            loadparameter("weights", new Tensor(2, shape));
            parameters["weights"]->normal();
            parameters["weights"]->learnable();
            parameters["weights"]->persist();
            parameters["weights"]->name = name + ": weights";

            toGPU();
            
        }

        Tensor* forward(int* x, int size, int batches) {
            //int output_shape[3] = {batches, size, _embedding_dim}; //batches size embeddings
            int* output_shape = (int*)malloc(sizeof(int)*3);
            output_shape[0] = batches;
            output_shape[1] = size;
            output_shape[2] = _num_embeddings;
            Tensor* y0 = new Tensor(3, output_shape);
            y0->name = name + " tokens";
            y0->toGPU();
            algebra::onehot<<<((size*batches)/256+1)*256, 256>>>(x, y0->gpu_ptr(), _num_embeddings, size*batches);
            cce("Embedding one hot");
            Tensor* y1 = y0->mm(parameters["weights"]);
            cce("Embedding weights");
            return y1;
        }
};

class Linear : public Module {
    std::string _name;

    int _in_features;
    int _out_features;

    public:
        Linear(int in_features, int out_features, std::string name="linear") {
            _name = name;
            _in_features = in_features;
            _out_features = out_features;

            int* weight_shape = (int*)malloc(sizeof(int)*2);
            weight_shape[0] = out_features;
            weight_shape[1] = in_features;
            int* bias_shape = (int*)malloc(sizeof(int));
            bias_shape[0] = out_features;

            loadparameter("weights", new Tensor(2, weight_shape));
            loadparameter("bias", new Tensor(1, bias_shape));

            float k = sqrtf(1.0f/in_features);

            parameters["weights"]->name = name + ": weights";
            parameters["bias"]->name = name + ": bias ";
            parameters["weights"]->uniform(-k, k);
            parameters["bias"]->uniform(-k, k);
            //parameters["bias"]->zeros();
            parameters["weights"]->learnable();
            parameters["weights"]->persist();
            parameters["bias"]->learnable();
            parameters["bias"]->persist();
            toGPU();
        }

        Tensor* forward(Tensor* x) {
            Tensor* transposed = parameters["weights"]->transpose(0, 1);
            Tensor* y0 = x->mm(transposed);
            cce("Linear: weights");
            Tensor* y1 = y0->add(parameters["bias"]);
            cce("Linear: bias");
            return y1;
        }
};

class LayerNorm : protected Module {

    int* _normshape;
    int _eps;
    int _layersize;

    public:
        std::string name;

        LayerNorm(int dim, int* normshape, float eps=0.00001, std::string _name="__layernorm") {
            name = _name;
            _normshape = normshape;
            _eps = eps;
            _layersize = 1;
            for (int i = 0; i < dim; i++) { _layersize *= normshape[i]; }

            int* weight_shape = (int*)malloc(sizeof(int)*dim); 
            for(int i = 0; i < dim; i++) { weight_shape[i] = normshape[i]; }
            int* bias_shape = (int*)malloc(sizeof(int)*dim); 
            for(int i = 0; i < dim; i++) { bias_shape[i] = normshape[i]; }
            loadparameter("weights", new Tensor(dim, weight_shape));
            loadparameter("bias", new Tensor(dim, bias_shape));
            toGPU();
            parameters["weights"]->set(1.0f);
            parameters["bias"]->set(0.0f);
            parameters["weights"]->name = this->name + ": weight";
            parameters["bias"]->name = this->name + ": bias";
            parameters["weights"]->learnable();
            parameters["weights"]->persist();
            parameters["bias"]->learnable();
            parameters["bias"]->persist();
        }

        ~LayerNorm() {
            free(_normshape);
        }

        Tensor* forward(Tensor* x) {
            Tensor* y0 = x->layernorm(_layersize);
            cce("Layernorm");
            Tensor* y1 = y0->mul(parameters["weights"]);
            Tensor* y2 = y1->add(parameters["bias"]);
            return y2;
        }
};

class GELU : protected Module {

    public:
        GELU() {}

        Tensor* forward(Tensor* x) {
            Tensor* y = x->gelu();
            cce("GELU");
            return y;
        }
};

class SelfAttention : public Module {

    int _n_embed;
    int _n_head;
    int _block_size;

    public:
        Linear* c_attn;
        Linear* c_proj;
    
        SelfAttention(int n_embed, int n_head, int block_size) {
            _n_embed = n_embed;
            _n_head = n_head;
            _block_size = block_size;

            loadparameter("mask", Tensor::tri(block_size));
            parameters["mask"]->name = "triangle mask";
            parameters["mask"]->persist();

            c_attn = new Linear(n_embed, 3 * n_embed);
            c_proj = new Linear(n_embed, n_embed);
        }

        Tensor* forward(Tensor* x) {
            // B, T, C batch, sequence length, n_embed

            int B = x->shape()[0];
            int T = x->shape()[1];
            int C = x->shape()[2];

            // calculate q k v
            Tensor* proj_in = c_attn->forward(x);

            Tensor** qkv = proj_in->split(3, 2);

            Tensor* q = qkv[0];
            Tensor* k = qkv[1];
            Tensor* v = qkv[2];

            int* q_attn_shape = (int*)malloc(sizeof(int)*4);
            int* k_attn_shape = (int*)malloc(sizeof(int)*4);
            int* v_attn_shape = (int*)malloc(sizeof(int)*4);
            q_attn_shape[0] = B;
            k_attn_shape[0] = B;
            v_attn_shape[0] = B;
            q_attn_shape[1] = T;
            k_attn_shape[1] = T;
            v_attn_shape[1] = T;
            q_attn_shape[2] = _n_head;
            k_attn_shape[2] = _n_head;
            v_attn_shape[2] = _n_head;
            q_attn_shape[3] = _n_embed / _n_head;
            k_attn_shape[3] = _n_embed / _n_head;
            v_attn_shape[3] = _n_embed / _n_head;

            q = q->view(4, q_attn_shape)->transpose(1, 2); // transpose head and seq length
            k = k->view(4, k_attn_shape)->transpose(1, 2);
            v = v->view(4, v_attn_shape)->transpose(1, 2);

            // calculate attention
            // attn = q mm k / sqrt(k.size)
            Tensor* k_transpose = k->transpose(2, 3);

            Tensor* y0 = q->mm(k_transpose);
            Tensor* y1 = y0->div(sqrtf((float)k->shape()[k->dim()-1]));
            Tensor* masked_attn = y1->mask(parameters["mask"]);
            Tensor* attn = masked_attn->softmax(T, B*T*_n_head);
            // attn = dropout(attn)
            // y = attn mm v
            Tensor* y2 = attn->mm(v);

            int* output_shape = (int*)malloc(sizeof(int)*3);
            output_shape[0] = B;
            output_shape[1] = T;
            output_shape[2] = _n_embed;
            Tensor* y3 = y2->transpose(1, 2)->view(3, output_shape);
            Tensor* y4 = c_proj->forward(y3);
            y4->name = "self-attention(" + x->name + ")";
            return y4;
        }
};

class MLP : protected Module {

    int _n_embed;

    Linear* ln_in;
    GELU* ge;
    Linear* ln_out;

    public:
        MLP(int n_embed) {
            _n_embed = n_embed;

            ln_in = new Linear(_n_embed, 4*_n_embed);
            ge = new GELU();
            ln_out = new Linear(4*_n_embed, _n_embed);
        }

        Tensor* forward(Tensor* x) {
            // x = linear(n_embed, 4*n_embed)
            x = ln_in->forward(x);
            // x = gelu()
            x = ge->forward(x);
            // x = linear(4*n_embed, n_embed)
            x = ln_out->forward(x);
            // x = dropout()
            return x;
        }  
};

class Block : protected Module {
    LayerNorm* ln_in;
    SelfAttention* sa;
    LayerNorm* ln_out;
    MLP* mlp;

    public:
        Block(int n_embed, int n_head, int block_size) {
            int* in_normshape = (int*)malloc(sizeof(int));
            in_normshape[0] = block_size;
            in_normshape[1] = n_embed;
            ln_in = new LayerNorm(2, in_normshape);
            sa = new SelfAttention(n_embed, n_head, block_size);
            int* out_normshape = (int*)malloc(sizeof(int));
            out_normshape[0] = block_size;
            out_normshape[1] = n_embed;
            ln_out = new LayerNorm(2, out_normshape);
            mlp = new MLP(n_embed);
        }

        Tensor* forward(Tensor* x) {
            // layer norm over n_embed
            Tensor* x1 = ln_in->forward(x);

            // self attention
            Tensor* x2 = sa->forward(x1);
            Tensor* x3 = x2->add(x);
            
            // layer norm over n_embed
            Tensor* x4 = ln_out->forward(x3);
            Tensor* x5 = mlp->forward(x4);
            Tensor* y = x5->add(x3);
            return y;
        }
};

class Transformer : protected Module {
    
    int _vocab_size;
    int _n_embed;
    int _n_head;
    int _block_size;
    int _n_layers;
    
    Embedding* wte;
    Embedding* wpe;
    Block** blocks;
    LayerNorm* lnf;
    Linear* lm_head;
    
    public:
        Transformer(int vocab_size, int n_embed, int n_head, int block_size, int n_layers) {
            _vocab_size = vocab_size;
            _n_embed = n_embed;
            _n_head = n_head;
            _block_size = block_size;
            _n_layers = n_layers;

            wte = new Embedding(vocab_size, n_embed);
            wte->name = "wte";
            wpe = new Embedding(block_size, n_embed);
            wpe->name = "wpe";
            // dropout
            blocks = (Block**)malloc((sizeof(Block*)*n_layers));
            for(int i = 0; i < n_layers; i++) { blocks[i] = new Block(n_embed, n_head, block_size); }
            int* norm_shape = (int*)malloc(sizeof(int)*1);
            norm_shape[0] = n_embed;
            lnf = new LayerNorm(1, norm_shape, 0.000001, "lnf"); 
            lm_head = new Linear(n_embed, vocab_size, "linear out"); 
        }

        Tensor* forward(int* tokens, int length, int batches) {
            // figure out timestep

            if(length > _block_size) {
                std::cerr << "Stream length of " << length << " is greater than the model block size (" << _block_size << ")\n";
                exit(EXIT_FAILURE);
            }

            // generate positions
            int* pos = (int*)malloc(sizeof(int)*length*batches);
            int* dev_pos;
            cce(cudaMalloc(&dev_pos, sizeof(float)*length*batches));
            for (int i = 0; i < length; i++) { 
                for(int j = 0; j < batches; j++) pos[j*length + i] = i; 
            }
            cce(cudaMemcpy(dev_pos, pos, sizeof(int)*length*batches, cudaMemcpyHostToDevice));            

            // create token embeddings (B, T, n_embed)
            Tensor* tok_embed = wte->forward(tokens, length, batches);
            // create position embeddings (B, T, n_embed)
            Tensor* pos_embed = wpe->forward(dev_pos, length, batches);

            // dropout

            // run through the blocks
            Tensor* x = tok_embed->add(pos_embed);

            for (int i = 0; i < _n_layers; i++) {
                x->name = "layer=" + std::to_string(i);
                x = blocks[i]->forward(x);
            }
            x->name = "transformer output";

            // layer normalize
            Tensor* x1 = lnf->forward(x);
            Tensor* x2 = lm_head->forward(x1);
            Tensor* x3 = x2->softmax(_vocab_size, batches*length);
            return x3;

            // calculate loss
            // logits = linear transform to tokens
            // cross entropy loss
        }
};

class Bigram : protected Module {
    
    int _vocab_size;
    int _n_embed;

    public:
        Embedding* embed_table;
        Linear* lm_head;

        Bigram(int vocab_size, int n_embed) {
            _vocab_size = vocab_size;
            _n_embed = n_embed;
            embed_table = new Embedding(vocab_size, n_embed);
            embed_table->name = "embedding table";
            lm_head = new Linear(n_embed, vocab_size, "linear head");
        }

        Tensor* forward(int* x, int stream_length, int batches) {
            Tensor* y0 = embed_table->forward(x, stream_length, batches);
            Tensor* y1 = lm_head->forward(y0);
            Tensor* y2 = y1->softmax(_vocab_size, stream_length*batches);
            return y2;
        }


};