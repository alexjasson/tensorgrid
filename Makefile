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

SRC = src/tensorgrid.c src/Operation.c src/ThreadPool.c src/DynamicArray.c src/utility.c
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
