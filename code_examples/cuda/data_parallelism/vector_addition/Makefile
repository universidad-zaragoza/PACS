NVCC      = nvcc
NVCCFLAGS = -O2
# To target your specific GPU, uncomment and adjust (e.g. sm_75, sm_86, sm_89):
# NVCCFLAGS += -arch=sm_75

TARGETS = vector_add

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: all
	@for t in $(TARGETS); do echo "== $$t =="; ./$$t; done

clean:
	rm -f $(TARGETS)

.PHONY: all run clean

