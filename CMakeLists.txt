cmake_minimum_required(VERSION 3.10)

project(matrizes C)

find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)

include_directories(include ${MPI_INCLUDE_PATH})

set(SOURCES
    matrizes.c
    src/work_pool.c
    src/master_slave.c
)

add_executable(matrizes ${SOURCES})

target_link_libraries(matrizes ${MPI_LIBRARIES})
