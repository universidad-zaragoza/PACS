void WriteToFile(const std::string& message) {
    // |mutex| is to protect access to |file| (which is shared across threads).
    static std::mutex mutex;

    // Lock |mutex| before accessing |file|.
    std::lock_guard<std::mutex> lock(mutex);

    // Try to open file.
    std::ofstream file("example.txt");
    if (!file.is_open()) {
        throw std::runtime_error("unable to open file");
    }

    // Write |message| to |file|.
    file << message << std::endl;

    // |file| will be closed first when leaving scope (regardless of exception)
    // mutex will be unlocked second (from lock destructor) when leaving scope
    // (regardless of exception).
}

