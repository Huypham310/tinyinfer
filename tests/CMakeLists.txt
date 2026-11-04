include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)
# keep gtest on the same CRT as us on MSVC
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tinyinfer_tests
  test_cpu_features.cpp
  test_tensor.cpp
)
target_link_libraries(tinyinfer_tests PRIVATE tinyinfer tinyinfer_warnings GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tinyinfer_tests)
