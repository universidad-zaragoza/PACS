#include <mutex>

std::mutex m;
int shared_var = 0;

void increment() {
    std::lock_guard<std::mutex> lk(m);
    ++shared_var;
    heavy_long_work();
}
