#include <mutex>

std::mutex m;
int shared_var = 0;

void increment() {

    int* siete = new int(7);

    m.lock();
    ++shared_var;
    m.unlock();
}
