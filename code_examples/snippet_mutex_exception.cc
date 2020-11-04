#include <mutex>

std::mutex m;
int shared_var = 0;

void increment() {
    m.lock();
    int* seven = new int(7);
    shared_var+=*seven;
    m.unlock();
}
