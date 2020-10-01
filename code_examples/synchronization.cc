#include <iostream>
#include <thread>

using namespace std;

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

    const size_t n = 1024*1024;
    thread t(increase_var, n);

    for(size_t i = 0; i < n; ++i) {
        shared_variable--;
    }
    t.join();

    std::cout << "shared_variable=" << shared_variable << std::endl;
}
