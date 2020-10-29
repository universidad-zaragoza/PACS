std::mutex myMutex;
int sharedVariable;

void func2()
{
    std::lock_guard<mutex> lck(myMutex);
    doSomething2();
}
void func1()
{
    std::lock_guard<mutex> lck(myMutex);
    doSomething1();
    func2();
}
