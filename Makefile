# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    ifneq ($(shell which gcc 2>/dev/null),)
        CC := gcc
        AR := gcc-ar
    else ifneq ($(shell which clang 2>/dev/null),)
        CC := clang
        AR := llvm-ar
    else
        $(error Compiler not found! Please install gcc or clang)
    endif
else ifeq ($(UNAME_S),Darwin)
    ifneq ($(shell which clang 2>/dev/null),)
        CC := clang
    else ifneq ($(shell which gcc 2>/dev/null),)
        CC := gcc
    else
        $(error Compiler not found! Please install gcc or clang)
    endif
    AR := ar
else
    $(error Unsupported OS! Only linux and mac are supported)
endif

CFLAGS = -Wall -Wextra -Werror -pedantic -Ofast -march=native -flto

LIBDIR     = /usr/local/lib
INCLUDEDIR = /usr/local/include

SRC = src/tensorgrid.c src/Operation.c src/utility.c                                                     \
      src/Operation/Map.c src/Operation/Zip.c src/Operation/Affine.c src/Operation/Multiply.c            \
      src/Operation/Convolution.c src/Operation/Pool.c src/Operation/Gather.c src/Operation/Scatter.c    \
      src/Operation/Normalize.c src/Operation/Softmax.c src/Operation/CrossEntropy.c src/Operation/MSE.c \
      src/adt/ThreadPool.c src/adt/DynamicArray.c src/adt/MemoryPool.c src/adt/HashMap.c src/adt/Stack.c \
      src/math/Scalar.c src/math/Vector.c src/math/Matrix.c
OBJ = $(SRC:.c=.o)
LIB = libtensorgrid.a

.PHONY: install uninstall

$(LIB): $(OBJ)
	$(AR) rcs $@ $^
	rm -f $(OBJ)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(LIB)
	install -d $(LIBDIR) $(INCLUDEDIR)
	install -m 644 $(LIB) $(LIBDIR)
	install -m 644 src/tensorgrid.h $(INCLUDEDIR)
	rm -f $(LIB)

uninstall:
	rm -f $(LIBDIR)/$(LIB)
	rm -f $(INCLUDEDIR)/tensorgrid.h
