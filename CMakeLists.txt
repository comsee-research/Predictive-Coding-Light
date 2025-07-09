cmake_minimum_required(VERSION 3.10.0)

# Extract version from header, done first to satisfy CMP0048,
# see `cmake --help-policy CMP0048` for more information.
file(
	STRINGS                         # Read in a file to memory.
	libs/version.h           		# File to parse for version number.
	NEUVISYS_PROJECT_VERSION          	# Where to store the results (should just be one line)
	REGEX "#define NEUVISYS_VERSION " 	# The space after is important to ignore 'NEUVISYS_VERSION_INT'
)

# Transform '#define NEUVISYS_VERSION "X.Y.Z"' into 'X.Y.Z'
string(REGEX REPLACE ".*\"(.*)\".*" "\\1" NEUVISYS_PROJECT_VERSION "${NEUVISYS_PROJECT_VERSION}")

# Project name and version
project(neuvisys
        VERSION ${NEUVISYS_PROJECT_VERSION}
        LANGUAGES C CXX)

# Ensure proper build type
if (NOT CMAKE_BUILD_TYPE)
  message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
else (NOT CMAKE_BUILD_TYPE)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("\n=================================================================================")
    message("\n-- Build type: Debug. Performance will be terrible!")
    message("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
    message("\n=================================================================================")
  endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
endif (NOT CMAKE_BUILD_TYPE)

# Options
option(BUILD_GUI "Enable to use graphical user interface" OFF)

#================= extra building definitions ==============================
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT CMAKE_INSTALL_PREFIX)
    IF (UNIX AND NOT APPLE)
        SET(CMAKE_INSTALL_PREFIX "/usr" CACHE STRING "CMake default install prefix, set to /usr on Unix/Linux")
    ELSEIF (APPLE)
        SET(CMAKE_INSTALL_PREFIX "/usr/local" CACHE STRING "CMake default install prefix, set to /usr/local on macOS")
    ELSE ()
        MESSAGE(FATAL_ERROR "CMAKE_INSTALL_PREFIX is not set")
    ENDIF ()
ENDIF ()

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-O3)
endif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")

include(GNUInstallDirs)

message("-----------------------------------------------------------------------------------------")
message("${PROJECT_NAME}")
message("-----------------------------------------------------------------------------------------")

#===============================================================================
#=================== build dependencies ========================================
add_subdirectory(externals)
include_directories(PUBLIC externals/)

#===============================================================================
#=================== build sources and libs ====================================
add_subdirectory(libs)
#===============================================================================

#===============================================================================
#=================== build tests ===============================================
add_subdirectory(test)
#===============================================================================

message("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
