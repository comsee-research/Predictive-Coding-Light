set(MY_LIB_NAME network)

message("-----------------------------------------------------------------------------------------")
message("\t[LIB] Neuvisys - ${MY_LIB_NAME}")
message("-----------------------------------------------------------------------------------------")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
find_package(PythonLibs REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(OpenCV REQUIRED)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
set(SRC_BASE_PATH "src")
set(HDRS_BASE_PATH "include/${MY_LIB_NAME}")

set(NEUVISYS_NETWORK_SOURCES        
		${SRC_BASE_PATH}/SpikingNetwork.cpp
        ${SRC_BASE_PATH}/NetworkHandle.cpp
		${SRC_BASE_PATH}/SurroundSuppression.cpp
        #---------
        ${SRC_BASE_PATH}/neurons/Neuron.cpp
        ${SRC_BASE_PATH}/neurons/SimpleNeuron.cpp
        ${SRC_BASE_PATH}/neurons/ComplexNeuron.cpp
		#---------
		${SRC_BASE_PATH}/utils/Util.cpp
		${SRC_BASE_PATH}/utils/WeightMatrix.cpp
		${SRC_BASE_PATH}/utils/WeightMap.cpp
		#---------
		${SRC_BASE_PATH}/config/Config.cpp
		${SRC_BASE_PATH}/config/DefaultConfig.cpp
)

set(NEUVISYS_NETWORK_HEADERS       
		${HDRS_BASE_PATH}/Event.hpp
		${HDRS_BASE_PATH}/SpikingNetwork.hpp
        ${HDRS_BASE_PATH}/NetworkHandle.hpp
		${HDRS_BASE_PATH}/SurroundSuppression.hpp
        #---------
        ${HDRS_BASE_PATH}/neurons/Neuron.hpp
        ${HDRS_BASE_PATH}/neurons/SimpleNeuron.hpp
        ${HDRS_BASE_PATH}/neurons/ComplexNeuron.hpp
		#---------
		${HDRS_BASE_PATH}/utils/Types.hpp
		${HDRS_BASE_PATH}/utils/Util.hpp
		${HDRS_BASE_PATH}/utils/WeightMatrix.hpp
		${HDRS_BASE_PATH}/utils/WeightMap.hpp
		#---------
		${HDRS_BASE_PATH}/config/Config.hpp
		${HDRS_BASE_PATH}/config/DefaultConfig.hpp
)

set(NEUVISYS_NETWORK_LIBS
	${OpenCV_LIBS} 
	${PYTHON_LIBRARIES} 
	${HDF5_LIBRARIES}
	cnpy
	z
	-lstdc++fs
)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
add_library(${MY_LIB_NAME} STATIC ${NEUVISYS_NETWORK_SOURCES} ${NEUVISYS_NETWORK_HEADERS})

target_include_directories(
	${MY_LIB_NAME} 
	PUBLIC ${HDF5_INCLUDE_DIRS} include/
	PROTECTED ${PYTHON_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
	PRIVATE ${HDRS_BASE_PATH}/neurons ${HDRS_BASE_PATH}/utils ${HDRS_BASE_PATH}/config ${HDRS_BASE_PATH}
)
target_link_libraries(
	${MY_LIB_NAME} 
	PUBLIC ${NEUVISYS_NETWORK_LIBS}
)

message("\n\t-- Done.\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
