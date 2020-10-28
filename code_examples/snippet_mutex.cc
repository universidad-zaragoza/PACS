#include <mutex>

std::mutex m;
int shared_var = 0;

void increment() {
    m.lock();
    ++shared_var;
    m.unlock();
}
