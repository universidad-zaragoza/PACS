#include <future>
#include <iostream>
#include <random>
#include <vector>

// assume a container
template <typename T>
T saxpy(const typename T::value_type A, const T& x, const T& y,
        size_t begin, size_t end)
{
    T z(x.size());

    for(size_t i = begin; i < end; ++i) {
         z[i] = A * x[i] + y[i];
    }
    return z;
}

int main() {

    const size_t N = 8;
    const float A = 3.14f;
    using vf = std::vector<float>;
    vf z(N, 0.0f);
    vf x, y;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(size_t i = 0; i < N; ++i) {
        x.push_back(dis(gen));
        y.push_back(dis(gen));
    }

    auto first_half =  std::async(std::launch::async, saxpy<vf>, A, std::ref(x), std::ref(y), 0, N/2);
    auto second_half = std::async(std::launch::deferred, saxpy<vf>, A, std::ref(x), std::ref(y), N/2, N);

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
    for(const auto i: first_half.get()) {
        std::cout << " " << i;
    }
    for(const auto i: second_half.get()) {
        std::cout << " " << i;
    }
    std::cout << std::endl;

}

