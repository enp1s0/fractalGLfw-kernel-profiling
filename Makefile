NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_80 -lcurand
TARGET=kernel-perf.out

$(TARGET):main.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS)

clean:
	rm -f $(TARGET)
