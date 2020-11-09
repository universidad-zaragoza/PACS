#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template<typename T>
class threadsafe_queue
{
  private:
      // please complete

  public:
    threadsafe_queue() {}

    threadsafe_queue(const threadsafe_queue& other)
    {
	// please complete
    }

    threadsafe_queue& operator=(const threadsafe_queue&) = delete;

    void push(T new_value)
    {
	// please complete
    }

    bool try_pop(T& value)
    {
	// please complete
    }

    void wait_and_pop(T& value)
    {
	// please complete
    }

    std::shared_ptr<T> wait_and_pop()
    {
	// please complete
    }

    bool empty() const
    {
	// please complete
    }
};
