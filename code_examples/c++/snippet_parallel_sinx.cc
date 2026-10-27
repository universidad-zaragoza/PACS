#include <thread>

void sinx(int N, int terms, float* x, float* result) {
	for (int i=0; i<N; i++) {
		float value = x[i];
		float numer = x[i] * x[i] * x[i];
		int denom = 6;	// 3!
		int sign = -1;

		for (int j=1; j<=terms; j++) {
			value += sign * numer / denom;
			numer *= x[i] * x[i];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}
		result[i] = value;
	}
}

void parallel_sinx(int N, int terms, float* x, float* result) {
	// create and launch a second thread
	std::thread t0(sinx, N - N/2, terms, x, result);
	// compute half work in main thread
	sinx(N - N/2, terms, x + N/2, result + N/2);
	t0.join(); // wait the completion of the thread
}
