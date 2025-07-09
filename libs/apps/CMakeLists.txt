#=================== includes app subdirectory =================================
#neuvisys
add_subdirectory(neuvisys)

#motor-control
if (BUILD_MOTOR_CONTROL)
    add_subdirectory(motor_control)
endif ()

#simulator
if (BUILD_SIMULATOR)
    add_subdirectory(simulator)
endif ()

#gui
if (BUILD_GUI)
    add_subdirectory(gui)
endif ()

#gui-simu
if (BUILD_GUI_SIMULATION)
    add_subdirectory(gui_simulation)
endif()



