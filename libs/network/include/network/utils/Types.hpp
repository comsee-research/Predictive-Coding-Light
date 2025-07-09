//
// Created by thomas on 06/07/22.
//

#ifndef NEUVISYS_TYPES_HPP
#define NEUVISYS_TYPES_HPP

#include <vector>
#include <unordered_map>
#include <array>
#include <queue>
#include <stack>

#include <boost/circular_buffer.hpp>
#include "H5Cpp.h"
#include "hdf5.h"
#include "json/json.hpp"
#include "cnpy/cnpy.h"

#include "network/Event.hpp"

using Events = std::vector<Event>;
using NetConf = std::vector<nlohmann::json>;

#endif //NEUVISYS_TYPES_HPP
