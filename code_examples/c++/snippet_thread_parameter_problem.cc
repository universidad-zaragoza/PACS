// Copied from C++ Concurrency in Action

void f(int i,std::string const& s);

void oops(int some_param)
{
    char buffer[1024];
    sprintf(buffer, "%i",some_param);
    std::thread t(f,3,buffer);
    t.detach();
}

// What happens if oops ends before t starts execution
// Please remember that the thread constructor will copy the parameters.
// Excerpt from the book
// "Note that even though f takes a std::string as the second parameter, the
// string literal is passed as a char const* and converted to a std::string
// only in the context of the new thread."
