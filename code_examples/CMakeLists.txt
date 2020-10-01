cmake_minimum_required (VERSION 3.1)
project (PACS_EXAMPLES)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_executable(hello_pacs hello_pacs.cc)
target_link_libraries(hello_pacs Threads::Threads)
add_executable(thread_joinable thread_joinable.cc)
target_link_libraries(thread_joinable Threads::Threads)
add_executable(hello_pacs_two_threads hello_pacs_two_threads.cc)
target_link_libraries(hello_pacs_two_threads Threads::Threads)
add_executable(hello_pacs_args hello_pacs_args.cc)
target_link_libraries(hello_pacs_args Threads::Threads)
add_executable(lambda_example lambda_example.cc)
add_executable(saxpy saxpy.cc)
target_link_libraries(saxpy Threads::Threads)
add_executable(saxpy_scaling saxpy_scaling.cc)
target_link_libraries(saxpy_scaling Threads::Threads)
