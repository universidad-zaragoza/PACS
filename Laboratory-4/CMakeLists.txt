ADD_PACS_EXECUTABLE(TARGET smallpt_thread_pool SOURCES smallpt_thread_pool.cpp)
target_include_directories(smallpt_thread_pool
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
