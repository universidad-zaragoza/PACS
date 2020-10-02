#include <iostream>
#include <random>
#include <thread>

namespace { // please check why the anonymous namespace
  int shared_variable = 0;
}

void increase_var(const size_t n)
{
    for(size_t i=0; i<n;i++) {
        shared_variable++;
    }
}

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(512*1024*1024, 1024*1024*1024);
    size_t n = dis(gen);
    
	std::thread t(increase_var, n);

    for(size_t i = 0; i < n; ++i) {
        shared_variable--;
    }
    t.join();

    std::cout << "shared_variable=" << shared_variable << std::endl;
}
