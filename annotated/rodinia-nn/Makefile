CC_FLAGS = -g -O2

NAME=nn

SM=30
CC=30

.PHONY: check-env clang clean nvcc

all: nvcc clang gen

check-env:
ifndef LLVM_HOME
	$(error LLVM_HOME is undefined)
endif
ifndef CUDA_HOME
	$(error CUDA_HOME is undefined)
endif

clean:
	rm -f ${NAME}.nvcc ${NAME}.clang gen

gen:
	${LLVM_HOME}/bin/clang++ hurricane_gen.cpp -o gen -lm

nvcc:
	nvcc ${NAME}.cu -o ${NAME}.nvcc

clang: check-env
	${LLVM_HOME}/bin/clang++ -O2 ${NAME}.cu --cuda-gpu-arch=sm_$(SM) -o ${NAME}.clang \
		-L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread

