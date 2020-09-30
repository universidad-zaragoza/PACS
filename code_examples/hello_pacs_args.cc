#include <iostream>
#include <thread>

void hello(const std::string &str) {
    std::cout << str << std::endl;
}

int main() {
     std::string str{"hello PACS"};
     std::thread t(hello,str);
     t.join(); // wait for completion
}

