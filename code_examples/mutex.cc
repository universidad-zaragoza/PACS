#include <iostream>
#include <mutex>
#include <thread>

namespace {

    std::mutex m;
    int shared_var = 0;
}

void increment() {
    m.lock();
    ++shared_var;
    m.unlock();
}

int main() {
    std::thread t0{increment};
    std::thread t1{increment};

    t0.join();
    t1.join();

    std::cout << shared_var << std::endl;
}
