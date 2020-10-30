
std::mutex mutex1;
std::mutex mutex2;

void func2() {
    std::lock_guard<mutex> lck1(mutex1);
    std::lock_guard<mutex> lck2(mutex2);
    do_something2();
}
void func1(){
    std::lock_guard<mutex> lck1(mutex2);
    std::lock_guard<mutex> lck2(mutex1);
    do_something1();
}

int main() {
    std::thread t1(func1);
    std::thread t2(func2);
}
