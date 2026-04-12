#!/bin/bash

function setup_repo() {
    mkdir -p sitter-libs;
    git clone https://github.com/tree-sitter/tree-sitter-go sitter-libs/go;
    git clone https://github.com/tree-sitter/tree-sitter-javascript sitter-libs/js;
    git clone https://github.com/tree-sitter/tree-sitter-c sitter-libs/c;
    git clone https://github.com/tree-sitter/tree-sitter-cpp sitter-libs/cpp;
    git clone https://github.com/tree-sitter/tree-sitter-c-sharp sitter-libs/cs;
    git clone https://github.com/tree-sitter/tree-sitter-python sitter-libs/py;
    git clone https://github.com/tree-sitter/tree-sitter-java sitter-libs/java;
    git clone https://github.com/tree-sitter/tree-sitter-ruby sitter-libs/ruby;
    mkdir -p "parser";
    python setup_repo.py sitter-libs;
}

setup_repo;