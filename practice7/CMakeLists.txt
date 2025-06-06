# Минимальная версия Qt, которая поддерживает Qt Charts
cmake_minimum_required(VERSION 3.16)
project(PlutoSDR CXX C)

# Опция для включения или отключения установки зависимостей
option(INSTALL_DEPS "Установить зависимости" ON)

find_package(Qt5 COMPONENTS Widgets Charts REQUIRED)
find_package(Gnuradio REQUIRED)

find_library(LIBIIO_LIBRARIES iio)
message(${LIBIIO_LIBRARIES})


set(MAIN_SOURCE_FILES
    src/main.cpp
)
set(TEST_SOURCE_FILES
    tests/chat_test.cpp
)

# Путь до необходимых библиотек
# include_directories(${PATH}/libiio)
# link_directories(${PATH}/libiio)


# Добавляем исполняемый файл
add_executable(main ${MAIN_SOURCE_FILES})
add_executable(chat_test ${TEST_SOURCE_FILES})
add_executable(ex_1 tests/rxtx_1_example.cpp)
add_executable(ex_2 tests/rxtx_2_example.cpp)

# Линковка с библиотеками (Qt)
target_link_libraries(main Qt5::Core Qt5::Widgets Qt5::Charts)
target_link_libraries(main ${LIBIIO_LIBRARIES})
target_link_libraries(chat_test ${LIBIIO_LIBRARIES})
target_link_libraries(ex_1  ${LIBIIO_LIBRARIES})
target_link_libraries(ex_2  ${LIBIIO_LIBRARIES})
