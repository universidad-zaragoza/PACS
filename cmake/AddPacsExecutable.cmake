macro(ADD_PACS_EXECUTABLE)
    set(options "")
    set(oneValueArgs TARGET KERNEL)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(ADD_PACS_EXECUTABLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    add_executable("${ADD_PACS_EXECUTABLE_TARGET}" ${ADD_PACS_EXECUTABLE_SOURCES})
    target_link_libraries("${ADD_PACS_EXECUTABLE_TARGET}"  Threads::Threads)  
endmacro(ADD_PACS_EXECUTABLE)
