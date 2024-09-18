#include <iomanip>
#include <iostream>
#include <limits>

// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {

    (void)steps; // ensure no warnings with unused variable
    return 0.0f;
}

int main(int argc, const char *argv[]) {

    // read the number of steps from the command line
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        exit(1);

    }

    size_t steps = std::stoll(argv[1]);
    auto pi = pi_taylor(steps);

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
        << pi << std::endl;
}
