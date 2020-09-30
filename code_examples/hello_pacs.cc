#include <iostream>
#include <thread>

void hello() {
    std::cout << "Hello PACS" << std::endl;
}

int main() {
     std::thread t(hello);
     t.join(); // wait for completion
}

