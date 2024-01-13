#include<set>
#include<list>
#include <numeric>
#include"tokenizer.h"
#include"train.h"
#include"../../include/tensor.h"
#include"./model.h"

// as true to C as possible

// Constants
const int BATCH_SIZE = 4;

const int N_LAYERS = 1;
const int N_HEAD = 2;
const int N_EMBED = 32;
const int BLOCK_SIZE = 8;
const int CONTEXT_SIZE = 1024;
const int EPOCHS = 10000;


int main() {
    // Read input file
    std::string dataset = readFile("tiny_shakespeare.txt"); // ----------------

    
    std::string snapshot = "";
    for(int i = 64; i < 64+BLOCK_SIZE; i++) {
        snapshot += dataset[i];
    }
    std::cout << "Snapshot:\n...\n" << snapshot << "\n...\n";

    // Tokenize
    std::set<char> chars(dataset.begin(), dataset.end());
    Tokenizer* tokenizer = new Tokenizer(chars); // ---------------------------

    Tokens* snapshot_tokens = new Tokens(tokenizer, snapshot);
    std::cout << "Tokenized Snpashot:\n...\n" << snapshot_tokens->pretty() << "\n...\n";

    // Creating train and validation splits
    int slice_pos = dataset.size() / 10;
    Tokens* val_data = new Tokens(tokenizer, dataset.substr(0, slice_pos));
    Tokens* train_data = new Tokens(tokenizer, dataset.substr(slice_pos+1));

    // Bigram Model
    Transformer* model = new Transformer(tokenizer->size(), N_EMBED, N_HEAD, BLOCK_SIZE, N_LAYERS);

    int total_iterations = EPOCHS * train_data->size() / BLOCK_SIZE;
    int iteration = 0;

    std::list<float> running_losses = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


    for(int i = 0; i < EPOCHS; i++) {
        //Tensor* last_loss = new Tensor(1, init_shape);
        for(int j = 0; j < train_data->size()-BLOCK_SIZE*BATCH_SIZE; j += BLOCK_SIZE) {
            int* x = train_data->sample(BATCH_SIZE, j, BLOCK_SIZE);
            std::cerr << "\n\n----------------------------------------\n\n";
            std::cerr << "\n\nTraining on sample: " << tokenizer->decode(x, BLOCK_SIZE) << "\n";
            int* devx;
            cudaMalloc(&devx, sizeof(int)*BLOCK_SIZE*BATCH_SIZE);
            cudaMemcpy(devx, x, sizeof(int)*BLOCK_SIZE*BATCH_SIZE, cudaMemcpyHostToDevice);

            Tensor* logits = model->forward(devx, BLOCK_SIZE, BATCH_SIZE);


            /*
            logits->toCPU();
            std::cout << "Logits: [ ";
            for(int i = 0; i < logits->dim(); i++) { std::cout << logits->shape()[i] << " "; }
            std::cout << "]\n" << logits->pretty() << "\n";
            logits->toGPU();
            */

            Tensor* probabilities = logits->index(devx);
            Tensor* L = probabilities->loss();
            L->name = "loss";
            L->toCPU();
            if(isnan(L->cpu_ptr()[0])) {
                std::cerr << "\nnan loss caught... printing graph and stopping\n";

                std::cerr << "finished epoch: " << i << " sample:\n";
                std::cerr << "target: " << tokenizer->decode(x, BLOCK_SIZE) << "\n";

                int* target_tokens = tokenizer->encode(tokenizer->decode(x, BLOCK_SIZE));
                std::cerr << "target (tokens): [ ";
                for(int i = 0; i < BLOCK_SIZE; i++) { std::cerr << target_tokens[i] << " "; }
                std::cerr << "]\n";

                std::cerr << "actual: " << tokenizer->decode_logits(logits, BLOCK_SIZE) << "\n";

                int* actual_tokens = tokenizer->encode(tokenizer->decode_logits(logits, BLOCK_SIZE));
                std::cerr << "actual (tokens): [ ";
                for(int i = 0; i < BLOCK_SIZE; i++) { std::cerr << actual_tokens[i] << " "; }
                std::cerr << "]\n";
                
                probabilities->toCPU();
                logits->toCPU();
                std::cerr << "probabilities: " << probabilities->pretty() << "\n";
                std::cerr << "logits: " << logits->pretty() << "\n\n";
                probabilities->toGPU();
                logits->toGPU();
                std::cerr << "Graph:\n" << L->pretty_graph() << "\n\n";
                std::cerr << "Grad:\n" << L->pretty_grad(true) << "\n\n";

                exit(EXIT_FAILURE);
            }
            L->toGPU();
            L->backward();
            
            L->toCPU();

            std::cerr << "Loss dim: " << L->dim() << ", shape: [ ";
            for(int i = 0; i < L->dim(); i++) { std::cerr << L->shape()[i] + " "; }
            std::cerr << "]\n";

            iteration++;
            std::cerr << "[" << (100*iteration/total_iterations) << "%] ";
            std::cerr << "Loss: " << L->cpu_ptr()[0];

            //std::cerr << "Graph:\n" << L->pretty_graph() << "\n\n";
            //std::cerr << "Grad:\n" << L->pretty_grad(true) << "\n\n";         

            running_losses.push_front(L->cpu_ptr()[0]);
            running_losses.pop_back();
            float running_average = 0;
            for(float i : running_losses) { running_average += i; }
            running_average /= 16.0f;
            std::cerr << " -- average of " << running_average << "\n";
            std::cerr << "Sample: ";
            std::cerr << tokenizer->decode_logits(logits, BLOCK_SIZE) << "\n";  
            L->toGPU();
            L->optimize(-0.005f);

            L->free_graph();

            //std::cerr << "--- exiting after first iteration ---\n\n\n";
            /*
            if(j > 10*BLOCK_SIZE) {
                std::cerr << "finished epoch: " << i << " sample:\n";
                std::cerr << "target: " << tokenizer->decode(x, BLOCK_SIZE) << "\n";

                int* target_tokens = tokenizer->encode(tokenizer->decode(x, BLOCK_SIZE));
                std::cerr << "target (tokens): [ ";
                for(int i = 0; i < BLOCK_SIZE; i++) { std::cerr << target_tokens[i] << " "; }
                std::cerr << "]\n";

                std::cerr << "actual: " << tokenizer->decode_logits(logits, BLOCK_SIZE) << "\n";

                int* actual_tokens = tokenizer->encode(tokenizer->decode_logits(logits, BLOCK_SIZE));
                std::cerr << "actual (tokens): [ ";
                for(int i = 0; i < BLOCK_SIZE; i++) { std::cerr << actual_tokens[i] << " "; }
                std::cerr << "]\n";

                probabilities->toCPU();
                logits->toCPU();
                std::cerr << "probabilities: " << probabilities->pretty() << "\n";
                std::cerr << "logits: " << logits->pretty() << "\n\n";
                probabilities->toGPU();
                logits->toGPU();
                std::cerr << "Graph:\n" << L->pretty_graph() << "\n\n";
                std::cerr << "Grad:\n" << last_loss->pretty_grad(true) << "\n\n";
                exit(EXIT_FAILURE);
            }
            */
        } 
    }
    
    std::cerr << "\n\n\ncompleted.\n";
    return 0;
 
  }

