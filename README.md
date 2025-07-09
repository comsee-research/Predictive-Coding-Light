# Neuvisys Project

The Neuvisys project stands for Neuromorphic Vision System. It is a library offering access to the Predictive Coding Light network.
The library is written in C++.
It can be launched with command lines or via a Qt gui. 

## Requirements

### Neuvisys

- Python
- OpenCV
- HDF5

Neuvisys uses libraries such as Eigen, a json parser, cnpy and caer all linked locally from src/dependencies

### OpenCV

To install Opencv:
``sudo apt install libopencv-dev python3-opencv``

### HDF5

To install HDF5:
``sudo apt-get install libhdf5-dev``

The HDF5 format is used to store event files, that can then be used by the network.
The format should be as follows:
- a group named "events"
- 5 dataset in that group: "t" for timestamps, "x" for pixel width axis, "y" for pixel height axis, "p" for polarities and "c" for camera (0 for left camera, 1 for right camera).

### Datasets and example networks

Datasets and example already trained networks can be found at https://drive.uca.fr/library/0656421d-ad64-4de5-8371-51852594fca1/Datasets_PCL/ 

### Qt
install QT 5 with the **Qt Charts** module:

``sudo apt install qt5-default``
``sudo apt install libqt5charts5-dev``

## Neuvisys libraries

By default, only the neuvisys library core is compiled.
There is one more libraries that adds functionnality:

- GUI: allows the use of a graphical user interface.

## Launch

To compile the Neuvisys library, in the root folder:
- Run ``mkdir build``, ``cd build``, ``cmake -DCMAKE_BUILD_TYPE=Release ..``

If you want to use some of the abovementioned functionnalities, you can compile them with:
- ``cmake -DBUILD_GUI=ON -DCMAKE_BUILD_TYPE=Release ..``
(put ``OFF`` on the functionnalities you do not want to use and compile).

The core neuvisys library does not need any installation requirements except OPENCV. But adding more functionnalities means installing the adequate libraries (see Requirements section).

If there is some errors, you may have to install the following python packages:
``pip install empy``
``pip install catkin-pkg``

- Run ``make -j`` to compile all targets

or

- Run ``make [target-name]`` to compile only one target. possible targets are:
- ``neuvisys_tests``  launches a test network that validates that the program works correctly.
- ``neuvisys-exe`` is the command line executable.
- ``neuvisys-qt`` is similar to neuvisys but with an added Qt interface. Please use the C++ code to choose the network and event paths in the libraries of the NeuvisysGui/NeuvisysThread.cpp

Compiled target are found in the "build/src" folder.

An example of use with the ``neuvisys-exe`` target:

- ``./neuvisys-exe [m_networkPath] [eventPath] [nbPass]``

``m_networkPath`` correspond to the path of the network structure. This must link to the network config file, such as: ``./network/configs/network_config.json``.

``eventPath`` is the relative path to an event file in the .npz format.

``nbPass`` is the number of times the events will be presented to the network.

### Create empty Network

You can generate an empty spiking network ready to use from:

- Run ``cd  build``, ``./neuvisys-exe [networkDirectory]``

The parameters will be set to their default values, but you can change them afterwards using the gui or directly via the json config files.

## Quick dev guide

Here is a quick guide to use the snn in your c++ code:

```
#include "src/network/NetworkHandle.hpp"

std::string m_networkPath = "/path/to/network_folder/";
NetworkConfig::createNetwork(m_networkPath);
```

Creates an empty network folder at the given path with a default configuration. You can modify the configurations stored in the configs folder (refer to the Configuration part).

```
std::string eventsPath = "/path/to/events.h5";
NetworkHandle network(m_networkPath, eventsPath);
```

Defines the path to the event file and creates the network. This might take a while depending on the number of layers, neurons and connections.

```
std::vector<Event> events;
while (network.loadEvents(events, 1)) {
    network.feedEvents(events);
}
```

Load the events chunk by chunk from the event file and feed them to the network.

```
network.save(eventsPath, 1);
```

Save the network weights and other information to the network folder.

```
#include "src/network/NetworkHandle.hpp"

std::string m_networkPath = "/path/to/network_folder/";
NetworkConfig::createNetwork(m_networkPath);

std::string eventsPath = "/path/to/events.h5";
NetworkHandle network(m_networkPath + "configs/network_config.json", eventsPath);

std::vector<Event> events;
while (network.loadEvents(events, 1)) {
    network.feedEvents(events);
}

network.save(eventsPath, 1);
```

## Configuration guide

The network parameters are saved in json configuration files:

- network_config.json : describes the network architecture, number of layers, neurons, types of newLocalInhibitoryEvent...
- simple_cell_config.json : simple neuron parameters
- complex_cell_config.json : complex neuron parameters

See the PCL paper for simple cell and complex cell configuration parameters.

### List of configuration parameters with explanation

[] : defines the range/type of the parameter

() : indicates that this a list, with one parameter for each layer.

#### Network config

| parameter name | type | range | explanation |
| ------ | ------ | ------ | ------ |
| nbCameras | integer | [1, 2] | for mono or stereo applications. |
| neuron1Synapses | integer | [1 - inf] | number of synapses between the pixel array and the first layer |
| sharingType | string | ["patch"] | type of weight sharing."patch" = weights shared between patches/regions of neurons |
<<<<<<< HEAD
| neuronType | list string | (["SimpleCell", "ComplexCell"], ...) | type of neuron used for each layer |
| inhibitions | list string | (["none", "local", "topdown", "lateral"], ...) | type of newLocalInhibitoryEvent |
| interConnections | list integer | ([0 - inf], ...) | indicates to which layer the indicated one is connected to. The first layer is the layer 0 and is always connected to the pixel array (-1) |
| patches | list of integer | (([0 - inf], [0 - inf], [0 - inf]), ...) | x, y and z coordinates of the patches |
| sizes | list of integer | (([0 - inf], [0 - inf], [0 - inf]), ...) | width, height and depth of each neuronal layer |
=======
| layerCellTypes | list string | (["SimpleCell", "ComplexCell"], ...) | type of neuron used for each layer |
| layerInhibitions | list string | (["none", "local", "topdown", "lateral"], ...) | type of newLocalInhibitoryEvent |
| interLayerConnections | list integer | ([0 - inf], ...) | indicates to which layer the indicated one is connected to. The first layer is the layer 0 and is always connected to the pixel array (-1) |
| layerPatches | list of integer | (([0 - inf], [0 - inf], [0 - inf]), ...) | x, y and z coordinates of the patches |
| layerSizes | list of integer | (([0 - inf], [0 - inf], [0 - inf]), ...) | width, height and depth of each neuronal layer |
>>>>>>> origin/surround-suppression
| neuronSizes | list of integer | (([0 - inf], [0 - inf], [0 - inf]), ...) | width, height and depth of the neurons receptive fields |
| neuronOverlap | list of integer | (([0 - inf], [0 - inf], [0 - inf]), ...) | x, y and z overlap between neuronal receptive fields |
