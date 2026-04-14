# tensorgrid

tensorgrid is a minimalist and easy-to-use tensor library written in C.

## Features
- N-dimensional tensors
- Basic tensor operations
- Automatic differentiation
- Multithreading
- Load and save tensors as `.npy` files
- No external dependencies

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

Then compile with:
```sh
gcc main.c -ltensorgrid
```
or
```sh
clang main.c -ltensorgrid
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
