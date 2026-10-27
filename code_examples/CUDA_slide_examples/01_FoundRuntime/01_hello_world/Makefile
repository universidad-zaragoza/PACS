NVCC      = nvcc
NVCCFLAGS = -O2

TARGET = hello_world
SRC    = hello_world.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

