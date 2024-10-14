#include <algorithm>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

int main() {

    const size_t N {8};
    const float A {3.14f};
    using vf = std::vector<float>;
    vf x(N), y(N);
		vf z;

    std::random_device rd;
    std::mt19937 gen {rd()};
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};

		z.reserve(N);

		std::ranges::generate(x.begin(), x.end(), [&]() { return dis(gen); });
		std::ranges::generate(y, [&]() { return dis(gen); });

		auto both = std::views::zip(x, y);

		for(auto [xx, yy]: both) {
			z.push_back(A*xx + yy);
		}

    std::cout << "x:";
		std::ranges::for_each(x, [](const auto& e) { std::cout << " " << e; });
    std::cout << std::endl;

    std::cout << "y:";
		std::ranges::for_each(y, [](const auto& e) { std::cout << " " << e; });
    std::cout << std::endl;

    std::cout << "z:";
		std::ranges::for_each(z, [](const auto& e) { std::cout << " " << e; });
    std::cout << std::endl;
}

