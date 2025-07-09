//
// Created by thomas on 12/07/2022.
//

#include "utils/WeightMap.hpp"

#include <utility>

std::random_device WeightMap::m_rd;
std::mt19937 WeightMap::m_generator(WeightMap::m_rd());
std::uniform_real_distribution<double> WeightMap::m_uniformRealDistr(0.0, 1.0);

WeightMap::WeightMap() : m_data(), m_dimensions() {
    m_size = 0;
}

WeightMap::WeightMap(std::vector<size_t> dimensions) : m_data(), m_dimensions(std::move(dimensions)) {
    m_size = 0;
}

double WeightMap::getNorm() {
    double norm = 0;
    for (const auto &element : m_data) {
        norm += pow(element.second, 2);
        // std::cout << "element = " << element.second << " ; pow = " << pow(element.second, 2) << std::endl;
    }
    
    return sqrt(norm);
}

double WeightMap::getL1Norm() {
    double norm = 0;
    for (const auto &element : m_data) {
        norm += std::abs(element.second); 
    }
    return norm;
}

double WeightMap::getAverage() {
    double avg = 0;
    for (const auto &element : m_data) {
        avg += std::abs(element.second); 
    }
    return avg/m_data.size();
}

double WeightMap::getMinimum() {
    double minimum = 10000000;
    for (const auto &element : m_data) {
        if(element.second < minimum) {
            minimum = element.second;
        } 
    }
    return minimum;
}

double WeightMap::getMaximum() {
    double maximum = 0;
    for (const auto &element : m_data) {
        if(element.second > maximum) {
            maximum = std::abs(element.second);
        } 
    }
    return maximum;
}

void WeightMap::normalize(const double normFactor) {
    auto norm = getNorm();
    if (norm != 0) {
        for (auto &element : m_data) {
            element.second *= normFactor / norm;
        }
    }
}

void WeightMap::normalizeL1Norm(const double normFactor) {
    auto norm = getL1Norm();
    if (norm != 0) {
        for (auto &element : m_data) {
            element.second *= normFactor / norm;
        }
    }
}

void WeightMap::thresh(const double threshold) {
    for (auto &element : m_data) {
        if(element.second > threshold) {
            element.second = threshold;
        }
    }
}

void WeightMap::reInit(const double value) {
    for (auto &element : m_data) {
        element.second = value / m_data.size();
    }
}

void WeightMap::loadNumpyFile(const std::string &filePath) {
    // auto *weights = cnpy::npy_load(filePath).data<double>();
    cnpy::NpyArray arr = cnpy::npy_load(filePath);
    double *weights = arr.data<double>();
    weightsToMap(m_data, weights);
}

void WeightMap::loadNumpyFile(cnpy::npz_t &arrayNPZ, const std::string &arrayName) {
    auto *weights = arrayNPZ[arrayName].data<double>();
    weightsToMap(m_data, weights);
}

void WeightMap::saveWeightsToNumpyFile(const std::string &filePath) {
    std::vector<double> data(static_cast<size_t>(m_data.size()));
    mapToWeights(m_data, data);
    auto shape = std::vector<size_t>(1, m_size);
    if (!m_dimensions.empty()) {
        shape = m_dimensions;
    }
    cnpy::npy_save(filePath + ".npy", &data[0], shape, "w");
}

void WeightMap::saveWeightsToNumpyFile(const std::string &filePath, const std::string &arrayName) {
    std::vector<double> data(static_cast<size_t>(m_data.size()));
    mapToWeights(m_data, data);
    auto shape = std::vector<size_t>(1, m_size);
    if (!m_dimensions.empty()) {
        shape = m_dimensions;
    }
    cnpy::npz_save(filePath, arrayName, &data[0], shape, "a");
}

void WeightMap::mapToWeights(const std::unordered_map<size_t, double> &map, std::vector<double> &data) {
    size_t count = 0;
    for (auto const &element: map) {
        data[count] = element.second;
        ++count;
    }
}

void WeightMap::weightsToMap(std::unordered_map<size_t, double> &map, const double *weights) {
    size_t count = 0;
    for (auto &element: map) {
        element.second = weights[count];
        ++count;
    }
}

double &WeightMap::at(size_t id) {
    return m_data.at(id);
}

void WeightMap::setSeed(size_t seed) {
    m_generator.seed(seed);
}

void WeightMap::addWeight(size_t id, bool uniform) {
    if (uniform) {
        m_data[id] = 1; //m_uniformRealDistr(m_generator);
    } else {
        m_data[id] = 0;
    }
    ++m_size;
}

std::vector<size_t> WeightMap::getKeys() {
    std::vector<size_t> keys;
    for (auto &kv : m_data) {
        keys.push_back(kv.first);
    }
    return keys;
}