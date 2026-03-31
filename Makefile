# Detect compiler
ifneq ($(shell which gcc 2>/dev/null),)
    CC := gcc
else
    $(error Compiler not found! Please install gcc)
endif

CFLAGS  = -Wall -Wextra -Werror -pedantic -Ofast -march=native -flto

LIBDIR     = /usr/local/lib
INCLUDEDIR = /usr/local/include

SRC = src/tensorgrid.c src/ThreadPool.c src/DynamicArray.c src/utility.c
OBJ = $(SRC:.c=.o)
LIB = libtensorgrid.a

.PHONY: install uninstall

$(LIB): $(OBJ)
	gcc-ar rcs $@ $^
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
