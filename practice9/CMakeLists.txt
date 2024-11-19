cmake_minimum_required(VERSION 3.16)
project(PlutoSDR CXX C)

# Find libiio
find_library(LIBIIO_LIBRARIES iio)
message(${LIBIIO_LIBRARIES})

# Include directories
include_directories(src/include)

# Add the my_utils library
add_library(my_utils SHARED src/my_utils/my_utils.cpp)

# Set the main source files
set(MAIN_SOURCE_FILES
    src/main.cpp
)

# Add the executable
add_executable(main ${MAIN_SOURCE_FILES})

# Link libraries
target_link_libraries(main my_utils ${LIBIIO_LIBRARIES} m)