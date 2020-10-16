#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[])
{
    int N = std::atoi(argv[1]);

    int i, j, k;

    MatrixXd a = MatrixXd::Random(N,N);
    MatrixXd b = MatrixXd::Random(N,N);
    MatrixXd c(N,N);

    c = a * b;

    return 0;
}
