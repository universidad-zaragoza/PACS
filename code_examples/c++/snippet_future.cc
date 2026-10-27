#include <future>
#include <iostream>
// function to check if a number is prime
bool is_prime (int x) { ... }

int main () {
    std::future<bool> fut = std::async(
	    std::launch::async, is_prime, 117);

    // ... do other work ...

    bool ret = fut.get(); // waits for is_prime to return
    return 0;
}
