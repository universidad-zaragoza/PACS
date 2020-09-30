cmake_minimum_required (VERSION 3.1)
project (PACS_EXAMPLES)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_executable(hello_pacs hello_pacs.cc)
add_executable(thread_joinable thread_joinable.cc)
add_executable(hello_pacs_two_threads hello_pacs_two_threads.cc)
