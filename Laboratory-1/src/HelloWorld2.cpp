#include <iostream>

using namespace std;

extern int func();

int main() {
    std::cout << "Hello World!";
    func();
    return 0;
}
