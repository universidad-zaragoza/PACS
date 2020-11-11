#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

template<typename RandomIt>
typename std::iterator_traits<RandomIt>::value_type // deduce the type of the iterator
parallel_sum(RandomIt beg, RandomIt end)
{
    auto len = end - beg;

    using ret_type = typename std::iterator_traits<RandomIt>::value_type;

    if (len < 16)
	return std::accumulate(beg, end, static_cast<ret_type>(0));

    RandomIt mid = beg + len/2;
    auto handle = std::async(std::launch::async,
	    parallel_sum<RandomIt>, mid, end);
    auto sum = parallel_sum(beg, mid);
    return sum + handle.get();
}


int main() {

    const size_t N = 128;
    using vf = std::vector<float>;
    vf v(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    for(size_t i = 0; i < N; ++i) {
        v.push_back(dis(gen));
    }

    float sum = parallel_sum(v.begin(), v.end());
    std::cout << "Total sum: " << sum << std::endl;
}
