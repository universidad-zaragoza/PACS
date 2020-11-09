#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include<join_threads.hpp>
#include<threadsafe_queue.hpp>

class thread_pool
{
  join_threads _joiner;

  using task_type = void();

  void worker_thread()
  {

  }

  public:
  thread_pool(size_t num_threads = std::thread::hardware_concurrency())
    : // please complete
  {
      // complete

  }

  ~thread_pool()
  {
    wait();
  }

  void wait()
  {
      // wait for completion

      // active waiting
  }

  template<typename F>
    void submit(F f)
    {
	// please complete
    }
};
