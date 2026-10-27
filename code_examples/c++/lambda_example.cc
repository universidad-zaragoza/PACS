#include <iostream>

int main() {

    const size_t N = 3;
    int A[N] = {1, 2, 3};

    auto lambda = [](const size_t N, const int* A) -> int {
        int sum = 0;
        for(size_t i=0; i < N; ++i) { sum+=A[i]; }
        return sum;
    };

    auto sum = lambda(N, A);
    std::cout << sum << std::endl;
}
