#include <chrono>
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

typedef struct {
    size_t large_chunk;
    size_t small_chunk;
    size_t split_item;
} chunk_info;

// For a given number of iterations N and threads
// the iterations are divided:
// N % threads receive N / threads + 1 iterations
// the rest receive N / threads
constexpr chunk_info
split_evenly(size_t N, size_t threads)
{
    return {N / threads + 1, N / threads, N % threads};
}

std::pair<size_t, size_t>
get_chunk_begin_end(const chunk_info& ci, size_t index)
{
    size_t begin = 0, end = 0;
    if (index < ci.split_item ) {
        begin = index*ci.large_chunk;
        end = begin + ci.large_chunk; // (index + 1) * ci.large_chunk
    } else {
        begin = ci.split_item*ci.large_chunk + (index - ci.split_item) * ci.small_chunk;
        end = begin + ci.small_chunk;
    }
    return std::make_pair(begin, end);
}

int main() {

    const size_t N = 1024*1024*1024;
    const float A = 3.14f;
    using vf = std::vector<float>;
    vf z(N, 0.0f);
    vf x, y;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(size_t i = 0; i < N; ++i) {
        float value = dis(gen);
        x.push_back(value);
        y.push_back(value); // reduce execution time
    }

    std::vector<std::thread> thread_vector;
    std::vector<std::chrono::milliseconds> extime_thread;

    size_t max_threads = std::min(48u, std::thread::hardware_concurrency());
    for(size_t current_threads = 1; current_threads <= max_threads; ++current_threads) {

        auto chunks = split_evenly(N, current_threads);
        // ToDo : run several times and check median and deviation
        // launch the work
        auto start = std::chrono::steady_clock::now();
        for(size_t i = 0; i < current_threads; ++i) {
            auto begin_end = get_chunk_begin_end(chunks, i);
            thread_vector.push_back(std::thread(saxpy<vf>, std::ref(z), A,
                    std::ref(x), std::ref(y), begin_end.first, begin_end.second));
            std::cout << i << ", " << begin_end.first << ", " << begin_end.second << std::endl;
        }

        // wait for completion
        for(size_t i = 0; i < current_threads; ++i) {
            thread_vector[i].join();
        }
        auto stop = std::chrono::steady_clock::now();
        extime_thread.push_back(
                std::chrono::duration_cast<std::chrono::milliseconds>
                (stop-start));

        // clean the vector array
        thread_vector.clear();
    }

    std::cout << "threads, time (ms)" << std::endl;
    for(size_t i = 0; i < extime_thread.size(); ++i) {
        std::cout << i+1 << ", " << extime_thread[i].count() << std::endl;
    }
}

