#include <iostream>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
    int N = std::atoi(argv[1]);

    int i, j, k;

    double **a = new double*[N];
    double **b = new double*[N];
    double **c = new double*[N];

    for(i = 0; i < N; ++i)
    {
	a[i] = new double[N];
	b[i] = new double[N];
	c[i] = new double[N];
    }

    // Storing elements of matrices
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
        {
            a[i][j] = ((double)rand()) / RAND_MAX;
	    b[i][j] = ((double)rand()) / RAND_MAX;
        }

    // Initializing elements of matrix c to 0
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            c[i][j] = 0;

    // Multiplying matrix a and b and storing in matrix c
    for(i = 0; i < N; ++i)
        for(j = 0; j < N; ++j)
            for(k = 0; k < N; ++k)
                c[i][j] += a[i][k] * b[k][j];

    // Displaying the multiplication of two matrix
    //cout << endl << "Output Matrix: " << endl;
    //for(i = 0; i < N; ++i)
    //   for(j = 0; j < N; ++j)
    //   {
    //       cout << " " << c[i][j];
    //       if(j == N-1)
    //           cout << endl;
    //   }

    return 0;
}
