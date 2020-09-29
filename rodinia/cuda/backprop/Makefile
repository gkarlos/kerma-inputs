include ../../common/make.config

# original Rodinia CC flags
CC_FLAGS = -g -O2

SM=30
CC=30

.PHONY: check-env clang clean nvcc

check-env:
ifndef LLVM_HOME
	$(error LLVM_HOME is undefined)
endif
ifndef CUDA_HOME
	$(error CUDA_HOME is undefined)
endif

all: nvcc clang

clean:
	rm -f backprop.nvcc backprop.clang

nvcc:
	nvcc backprop.cu -o backprop.nvcc

clang: check-env
	${LLVM_HOME}/bin/clang++ backprop.cu --cuda-gpu-arch=sm_$(SM) -o backprop.clang \
		-L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread \


