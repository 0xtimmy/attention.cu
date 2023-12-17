#ifndef TOKENIZER_H
#define TOKENIZER_H

#include<string>
#include<iostream>
#include<map>
#include<set>
#include"../../include/tensor.h"

// Declarations
class Tokens;
class Tokenizer;

// Definitions

class Tokenizer {

    std::set<char> _vocab; 
    std::map<char, int> stoi;
    std::map<int, char> itos;

    public:
        Tokenizer(std::set<char> vocab) {
            _vocab = vocab;
            stoi = {};
            itos = {};
            int i =0;
            for (const char& c : vocab) {
                stoi[c] = i;
                itos[i] = c;
                i++;
            }
        }

        ~Tokenizer() { }

        int size() {
            return _vocab.size();
        }

        int* encode(std::string x) {
            int* y = (int*)malloc(sizeof(int) * x.size());
            for (int i = 0; i < x.size(); i++) {
                y[i] = stoi[x[i]];
            }
            return y;
        }

        std::string decode(int* x, int size) {
            std::string y(size, ' ');
            for (int i = 0; i < size; i++) {
                y[i] = stoi[x[i]];
            }
            return y;
        }

        void print_vocab() {
            std::cout << "There are " << _vocab.size() << " unique characters: \n";

            for (const char& c : _vocab) {
                std::cout << c;
            }
            std::cout << "\n";
        }
};

class Tokens {

    int _size;
    int* _tokens;

    public:
        Tokens(Tokenizer* tokenizer, std::string str) {
            _size = str.size();
            _tokens = tokenizer->encode(str);
        }

        ~Tokens() {
            free(_tokens);
        }

        int size() {
            return _size;
        }
        const int* tokens() const {
            return _tokens;
        }

        std::string pretty() {
            std::string y = "[";
            for(int i = 0; i < _size-1; i++) {
                y += std::to_string(_tokens[i]) + ", ";
            }
            if(_size > 0) y += std::to_string(_tokens[_size-1]);
            y += "]";
            return y;
        }
};

#endif