#include<set>
#include"tokenizer.h"
#include"train.h"
#include"../../include/tensor.h"

// Constants
const int BLOCK_SIZE = 8;
const int BATCH_SIZE = 4;

int main() {
    // Read input file
    std::string dataset = readFile("tiny_shakespeare.txt"); // ----------------

    
    std::string snapshot = "";
    for(int i = 100; i < 200; i++) {
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


    return 0;
  }

