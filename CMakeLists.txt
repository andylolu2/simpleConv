cmake_minimum_required(VERSION 3.25.2)

project(simpleConv CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

function(compile name source_file)
    add_executable(${name} ${source_file})
    set_target_properties(
        ${name} PROPERTIES 
        CUDA_ARCHITECTURES native
    )
    target_compile_options(
        ${name} PRIVATE
        -Wno-unused-function
        --expt-relaxed-constexpr
    )
    target_include_directories(
        ${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
    )
endfunction()

compile(main src/main.cu)
compile(correctness test/correctness.cu)