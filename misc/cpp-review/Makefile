.PHONY: 01 01_obj 01_obj_exe_run 02 03 clean all $(EXE)

GCC = gcc
CC = g++
NVCC_COMPILER = nvcc

CUDA_ROOT_DIR=/usr/local/cuda

# CC compiler options:
CC_FLAGS=
CC_LIBS=

# NVCC compiler options:
CUDA_ARCH_FLAGS = -arch=sm_86
NVCC_FLAGS= $(CUDA_ARCH_FLAGS) -Wno-deprecated-gpu-targets
NVCC_LIBS=


CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart


SRC_DIR = src

OBJ_DIR = bin

INC_DIR = include


EXE = run_test

# Object files for the generic build:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o

01: 01.c
	@$(GCC) -o 01 01.c

# just compiles to object file
01_obj: 01.c
	@$(GCC) -c 01.c -o 01.o

# compiles and runs the object file
01_obj_exe_run: 01_obj
	@$(GCC) 01.o -o 01
	@./01

02: 02.c
	@$(GCC) -o 02 02.c

03: 03.cu
	@$(NVCC_COMPILER) $(NVCC_FLAGS) -o 03_cu 03.cu

all: $(EXE)

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files
$(OBJ_DIR)/main.o : main.cpp
	$(CC) $(CC_FLAGS) -c main.cpp -o $@

# Compile C++ source files from SRC_DIR to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CUDA_INC_DIR)

# Compile CUDA source files from SRC_DIR to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC_COMPILER) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS) $(CUDA_INC_DIR)


# Clean objects in object directory and specific targets.
clean:
	$(RM) -f 01 02 03_cu *.o
	$(RM) -f $(OBJ_DIR)/* $(EXE)



