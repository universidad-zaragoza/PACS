#include <chrono>
#include <iostream>
#include <thread>

using namespace std::chrono;

void hello() {
    while(1) {
        std::cout << "Hello" << std::endl;
        std::this_thread::sleep_for(milliseconds(300));
    }
}

int main() {
    std::thread t(hello); // thread creation
    while(1) {
        std::cout << "PACS" << std::endl;
        std::this_thread::sleep_for(milliseconds(300));
    }
}

