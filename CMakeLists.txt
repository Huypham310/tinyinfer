cmake_minimum_required(VERSION 3.24)

project(tinyinfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TINYINFER_BUILD_TESTS "Build unit tests" ON)
option(TINYINFER_NATIVE "Compile with -march=native" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Warnings stay on for everything we own; third-party targets get their own flags.
add_library(tinyinfer_warnings INTERFACE)
if(MSVC)
  target_compile_options(tinyinfer_warnings INTERFACE /W4 /permissive-)
else()
  target_compile_options(tinyinfer_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
endif()

configure_file(include/tinyinfer/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/tinyinfer/version.hpp @ONLY)

add_library(tinyinfer
  src/cpu_features.cpp
  src/layout.cpp
  src/shape.cpp
  src/tensor.cpp
)
target_include_directories(tinyinfer
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(tinyinfer PRIVATE tinyinfer_warnings)
if(TINYINFER_NATIVE AND NOT MSVC)
  target_compile_options(tinyinfer PRIVATE -march=native)
endif()

add_executable(tinyinfer_cli tools/tinyinfer_cli.cpp)
set_target_properties(tinyinfer_cli PROPERTIES OUTPUT_NAME tinyinfer)
target_link_libraries(tinyinfer_cli PRIVATE tinyinfer tinyinfer_warnings)

if(TINYINFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
