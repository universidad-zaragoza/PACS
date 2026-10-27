bool frame_ready;
std::mutex m;

void read_frame() {
    // ...
    {
    std::lock_guard<std::mutex> lk(m);
    frame_ready=true;
    }
    // ...
}

void process_frame() {
    // The class unique_lock is a general-purpose mutex ownership wrapper
    // allowing deferred locking, time-constrained attempts at locking,
    // recursive locking, transfer of lock ownership, and use with condition
    // variables.
    std::unique_lock<std::mutex> lk(m);
    while(!frame_ready)
    {
        lk.unlock();
        lk.lock();
    }
    // ...
    frame_ready=false;
    lk.unlock();
}
