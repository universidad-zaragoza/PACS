
std::mutex mutex1;
std::mutex mutex2;

void func2() {
    std::lock(mutex1, mutex2);
    // std::adopt_lock assumes that the lock has already been adquired
    // why the lock_guard is required ?
    std::lock_guard<mutex> lck1(mutex1, std::adopt_lock);
    std::lock_guard<mutex> lck2(mutex2, std::adopt_lock);
    do_something2();
}
void func1(){
    std::lock(mutex2, mutex1);
    std::lock_guard<mutex> lck1(mutex2, std::adopt_lock);
    std::lock_guard<mutex> lck2(mutex1, std::adopt_lock);
    do_something1();
}

int main() {
    std::thread t1(func1);
    std::thread t2(func2);
}
