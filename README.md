# tensorgrid

<img src="images/logo.svg" alt="tensorgrid" width="128" align="left" hspace="20">

tensorgrid is a lightweight and easy-to-use tensor library for building and training neural networks from scratch in C. It is optimized for both speed and simplicity.

New here? The best place to start is the [`examples/`](examples) directory. There you can see a minimalist `MLP-simple` example all the way to a full `GPT2` implementation.
<br clear="left">

## Features
- No external dependencies
- N-dimensional tensors
- Basic tensor operations
- Automatic differentiation
- Multithreading/SIMD instructions for CPU performance
- Load and save tensors as `.npy` files

## Prerequisites
- Linux or mac
- GCC or Clang
- make

## Installation
```sh
git clone https://github.com/alexjasson/tensorgrid
cd tensorgrid
sudo make install
```

## Usage
Include the header in your code:
```c
#include <tensorgrid.h>
```

Then compile and run with:
```sh
gcc main.c -o main -ltensorgrid
./main
```
or
```sh
clang main.c -o main -ltensorgrid
./main
```

## Uninstall
To uninstall, run the following from the cloned repo:
```sh
sudo make uninstall
```

Or manually:
```sh
sudo rm /usr/local/lib/libtensorgrid.a /usr/local/include/tensorgrid.h
```
