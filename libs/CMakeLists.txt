#=================== includes libs subdirectory ================================
#network
add_subdirectory(network)

#gui
if (BUILD_GUI)
    add_subdirectory(gui)
endif ()

#=================== includes apps subdirectory ================================
add_subdirectory(apps)



