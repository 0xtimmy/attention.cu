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

const int N_LAYERS = 12;
const int N_HEAD = 12;
const int N_EMBED = 32;
const int BLOCK_SIZE = 64;
const int CONTEXT_SIZE = 1024;
const int EPOCHS = 200;


int main() {
    // Read input file
    std::string dataset = readFile("tiny_names.txt"); // ----------------

    
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
    int slice_pos = dataset.size() / 2;
    Tokens* train_data = new Tokens(tokenizer, dataset.substr(0, slice_pos));
    Tokens* val_data = new Tokens(tokenizer, dataset.substr(slice_pos+1));

    // Bigram Model
    Bigram* model = new Bigram(tokenizer->size(), N_EMBED);

    int total_iterations = EPOCHS * train_data->size() / BLOCK_SIZE;
    int iteration = 0;

    std::list<float> running_losses = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for(int i = 0; i < EPOCHS; i++) {
        Tensor* logits;
        for(int j = 0; j < train_data->size()-BLOCK_SIZE; j += BLOCK_SIZE) {
            int* x = train_data->sample(1, j, BLOCK_SIZE);
            int* devx;
            cudaMalloc(&devx, sizeof(int)*BLOCK_SIZE);
            cudaMemcpy(devx, x, sizeof(int)*BLOCK_SIZE, cudaMemcpyHostToDevice);

            logits = model->forward(devx, BLOCK_SIZE, 1);

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
            L->backward();

            L->toCPU();

            if(isnan(L->cpu_ptr()[0])) {
                std::cerr << "!!NAN!!\n";

                logits->toCPU();
                std::cerr << "Logits:\n" << logits->pretty() << "\n";

                exit(EXIT_FAILURE);
            }


            running_losses.push_front(L->cpu_ptr()[0]);
            running_losses.pop_back();
            float running_average = 0;
            for(float i : running_losses) { running_average += i; }
            running_average /= 16.0f;
            iteration++;

            std::cout << "[" << (100*iteration/total_iterations) << "%] ";
            std::cout << "Loss: " << L->cpu_ptr()[0];
            std::cout << " -- average of " << running_average << "\n";
            std::cout << "Sample: ";
            std::cout << tokenizer->decode_logits(logits, BLOCK_SIZE) << "\n";  
            L->toGPU();

            L->optimize(0.0001f);
        } 
    }

    return 0;
 
  }

