# Минимальная версия Qt, которая поддерживает Qt Charts
cmake_minimum_required(VERSION 3.16)
project(PlutoSDR CXX C)

set(CMAKE_BUILD_TYPE Debug)

# Опция для включения или отключения установки зависимостей
option(INSTALL_DEPS "Установить зависимости" ON)
option(UNIT_TESTS_ENABLED "Build unit tests" ON)
option(PLUTO_TIMESTAMP "Build pluto with timestamp" ON)

# Для работы с модулями Qt и Gnuradio
# find_package(Qt5 COMPONENTS Widgets Charts REQUIRED)
# find_package(Gnuradio REQUIRED)

# RF frontend driver
find_package(SoapySDR REQUIRED)

set(MAIN_SOURCE_FILES
    src/main.c
)
# Путь до необходимых библиотек
# include_directories(${PATH}/libiio)
# link_directories(${PATH}/libiio)


# Добавляем исполняемый файл
add_executable(main ${MAIN_SOURCE_FILES})

# Линковка с библиотеками (Qt)
#   Для работы с модулями Qt и Gnuradio
# target_link_libraries(main Qt5::Core Qt5::Widgets Qt5::Charts)
target_link_libraries(main ${SoapySDR_LIBRARIES})


if(UNIT_TESTS_ENABLED)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()
