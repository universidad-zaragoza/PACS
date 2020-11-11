#include <iostream>
#include <thread>
#include <future>
#include <functional>

void task_lambda()
{
    std::packaged_task<int(int,int)> task([](int a, int b) {
	    return a + b;
	    });
    std::future<int> result = task.get_future();
    task(2, 9);
    std::cout << "task_lambda:\t" << result.get() << '\n';
}

int add(int a, int b) { return a + b; }

void task_thread()
{
    std::packaged_task<int(int,int)> task(add);
    std::future<int> result = task.get_future();
    std::thread task_td(std::move(task), 2, 10);
    task_td.join();
    std::cout << "task_thread:\t" << result.get() << '\n';
}

int main()
{
    task_lambda();
    task_thread();
}
