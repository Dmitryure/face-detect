cmake_minimum_required(VERSION 3.5.0)
project(oop VERSION 0.1.0)
set(CMAKE_CXX_STANDART 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(IMAGE_DIR "images")
set(VIDEO_DIR "videos")
set(ASSETS_DIR "assets")
set(BIN_DIR "${CMAKE_BINARY_DIR}/")

file(COPY ${IMAGE_DIR} DESTINATION ${BIN_DIR})
file(COPY ${VIDEO_DIR} DESTINATION ${BIN_DIR})
file(COPY "${ASSETS_DIR}/" DESTINATION ${BIN_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

file(GLOB lib_src
    "src/*.cpp"
    "src/*/*.cpp"
)

add_library(${PROJECT_NAME} STATIC
    ${lib_src}
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

file(GLOB src_files
    "*.cpp"
)

add_executable(runner ${src_files})

target_link_libraries(runner
    ${PROJECT_NAME}
)