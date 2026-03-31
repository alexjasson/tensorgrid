# tensorgrid

tensorgrid is a minimalist and easy-to-use tensor library written in C.

## Features
- N-dimensional tensors
- Automatic differentiation
- Multithreading
- Load and save tensors as `.npy` files
- Minimal dependencies, only needs POSIX threads library

## Prerequisites
- A UNIX-like OS (Linux or macOS)
- GCC
- make

## Installation
```sh
git clone https://github.com/alexjasson/tensorgrid
cd tensorgrid
sudo make install
```

Include the header in your code:
```c
#include <tensorgrid.h>
```

Then compile with:
```sh
gcc main.c -ltensorgrid -lm -lpthread
```

To uninstall, run the following from the cloned repo:
```sh
sudo make uninstall
```

Or manually:
```sh
sudo rm /usr/local/lib/libtensorgrid.a /usr/local/include/tensorgrid.h
```