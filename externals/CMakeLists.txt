#json
add_library(json INTERFACE)
target_include_directories(json INTERFACE json/)

#cnpy
add_library(cnpy SHARED IMPORTED GLOBAL)
set_target_properties(cnpy PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/cnpy/libcnpy.so")

if (BUILD_MOTOR_CONTROL OR (BUILD_GUI OR BUILD_GUI_SIMULATION))
    add_subdirectory(libserie)
endif ()

