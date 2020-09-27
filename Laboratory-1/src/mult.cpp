#include <iostream>

using namespace std;

const size_t max_i = 10;
const size_t max_j = 10;

int main() {
    int v = 0;
    for (size_t i=2; i<max_i; ++i) {
        for (size_t j=0; j<max_j; ++j) {
	       v += i * j;
        }
    }

    std::cout << v << std::endl;
    return 0;
}
