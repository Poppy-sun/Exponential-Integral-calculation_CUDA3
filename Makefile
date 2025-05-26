CXX := nvcc
CXXFLAGS := -O2 -std=c++11
INCLUDES := -Iinclude
SRC := src/main.cpp src/exp_integral_cpu.cpp src/exp_integral_gpu.cu
TARGET := bin/exponentialIntegral.out

$(TARGET): $(SRC)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

