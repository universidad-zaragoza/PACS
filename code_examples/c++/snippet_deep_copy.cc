#include <vector>

std::vector<float> get_initial_array()
{
    return std::vector<float>{0.0f, 2.71f, 1.61f};
}

int main()
{
    std::vector<float> a;
    a = get_initial_array();

    std::vector<float> b{0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> c = b;

    std::vector<float> d{0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> e = std::move(d);
}
