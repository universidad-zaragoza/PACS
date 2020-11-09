#pragma once

#include <thread>
#include <vector>

class join_threads
{
  std::vector<std::thread>& _threads;
  public:
  explicit join_threads(std::vector<std::thread>& threads) :
    _threads(threads)
  {}

  ~join_threads()
  {
    for(auto& t: _threads) {
      if(t.joinable()) {
        t.join();
      }
    }
  }
};
