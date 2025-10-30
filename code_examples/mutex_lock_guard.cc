#include <iostream>
#include <mutex>
#include <thread>

namespace {
  std::mutex g_m;
  int g_shared_var = 0;
}

void increment() {
  std::lock_guard<std::mutex> lock(g_m);
  ++g_shared_var;
}

int main() {
    std::cout << "shared_var before: " << g_shared_var << std::endl;
    
		std::thread t0(increment);
    std::thread t1(increment);

    t0.join();
    t1.join();

    std::cout << "shared_var after: " << g_shared_var << std::endl;
}
