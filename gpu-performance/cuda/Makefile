CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O2 -std=c++17
NVCCFLAGS ?= -O2 -std=c++17

.PHONY: all cpu cuda clean

all: cpu cuda

cpu: cpu_baseline

cpu_baseline: cpu_baseline.cpp vector_add.h
	$(CXX) $(CXXFLAGS) cpu_baseline.cpp -o cpu_baseline

cuda: vector_add

vector_add: vector_add.cu cpu_baseline.cpp vector_add.h
	$(NVCC) $(NVCCFLAGS) -DVECTOR_ADD_NO_MAIN vector_add.cu cpu_baseline.cpp -o vector_add

clean:
	rm -f cpu_baseline vector_add