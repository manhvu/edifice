# Makefile for Edifice CUDA NIF bridge
#
# Compiles c_src/edifice_cuda_nif.c into a shared library that can be
# loaded as an Erlang NIF. Links against libdl (for dlopen of the kernel
# library) and libcudart (for cudaMalloc/cudaFree/cudaDeviceSynchronize).
#
# Usage:
#   make              # Build NIF shared library
#   make clean        # Remove build artifacts
#
# The output goes to priv/libedifice_cuda_nif.so

# Paths
EDIFICE_ROOT := $(realpath ..)
PRIV_DIR     := $(EDIFICE_ROOT)/priv

# Erlang NIF headers — auto-detect from erl
ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().')

# Compiler
CC       := gcc
CFLAGS   := -shared -fPIC -O2 -Wall -Wextra -Wno-unused-parameter \
            -I$(ERTS_INCLUDE_DIR)
LDFLAGS  := -ldl

# Source and target
SRC    := edifice_cuda_nif.c
TARGET := $(PRIV_DIR)/libedifice_cuda_nif.so

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(PRIV_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(TARGET)

.PHONY: all clean
