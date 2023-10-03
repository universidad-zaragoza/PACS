#include <iostream>
#include <random>
#include <thread>
#include <vector>

// assume a container
template <typename T>
void saxpy(T& z, const typename T::value_type A, const T& x, const T& y,
        size_t begin, size_t end)
{
    for(size_t i = begin; i < end; ++i) {
         z[i] = A * x[i] + y[i];
    }
}

int main() {

    const size_t N {8};
    const float A {3.14f};
    using vf = std::vector<float>;
    vf z (N, 0.0f);
    vf x, y;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};

    for(size_t i = 0; i < N; ++i) {
        x.push_back(dis(gen));
        y.push_back(dis(gen));
    }

    std::thread first_half(saxpy<vf>, std::ref(z), A, std::ref(x), std::ref(y), 0, N/2);
    std::thread second_half(saxpy<vf>, std::ref(z), A, std::ref(x), std::ref(y), N/2, N);
    first_half.join();
    second_half.join();

    std::cout << "x:";
    for(const auto i: x) {
        std::cout << " " << i;
    }
    std::cout << std::endl;

    std::cout << "y:";
    for(const auto i: y) {
        std::cout << " " << i;
    }
    std::cout << std::endl;

    std::cout << "z:";
    for(const auto i: z) {
        std::cout << " " << i;
    }
    std::cout << std::endl;
}

