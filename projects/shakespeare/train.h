#ifndef TRAIN_H
#define TRAIN_H
#include<vector>
#include<iostream>
#include<fstream>
#include<stdexcept>
#include<string>
#include"../../include/tensor.h"
#include"../../include/algebra.h"

std::string readFile(char* filename) {
    std::ifstream file(filename);

    if(!file.is_open()) {
        std::cerr << "Could not open file: \"" << filename << "\"\n";
        throw std::runtime_error("");
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    return content;
}

#endif