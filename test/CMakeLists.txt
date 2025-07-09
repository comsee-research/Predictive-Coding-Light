# 'Google_test' is the subproject name
project(tests)

message("-----------------------------------------------------------------------------------------")
message("\t[TEST] Neuvisys - ${PROJECT_NAME}")
message("-----------------------------------------------------------------------------------------")

set(FIXTURE_SOURCES
        SpikingNetworkTest.cpp
        NetworkHandleTest.cpp)

add_executable(neuvisys_tests ${FIXTURE_SOURCES})

target_include_directories(neuvisys_tests PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(neuvisys_tests gtest gtest_main ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${HDF5_LIBRARIES} cnpy z network)