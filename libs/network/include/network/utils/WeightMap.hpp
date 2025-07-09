//
// Created by thomas on 12/07/2022.
//

#ifndef NEUVISYS_WEIGHTMAP_HPP
#define NEUVISYS_WEIGHTMAP_HPP

#include <random>
#include <utility>

#include "cnpy/cnpy.h"

#include "Types.hpp"

class WeightMap {
    std::unordered_map<size_t, double> m_data;
    size_t m_size{};
    std::vector<size_t> m_dimensions;

    static std::random_device m_rd;
    static std::mt19937 m_generator;
    static std::uniform_real_distribution<double> m_uniformRealDistr;

public:
    WeightMap();

    explicit WeightMap(std::vector<size_t> dimensions);

    static void setSeed(size_t seed);

    void addWeight(size_t id, bool uniform);

    void normalize(double normFactor);

    void normalizeL1Norm(const double normFactor);

    void thresh(const double threshold);

    void reInit(const double value);

    std::vector<size_t> getDimensions() { return m_dimensions; }

    size_t getSize() const { return m_size; }

    double &at(size_t id);

    void loadNumpyFile(const std::string &filePath);

    void loadNumpyFile(cnpy::npz_t &arrayNPZ, const std::string &arrayName);

    void saveWeightsToNumpyFile(const std::string &filePath);

    void saveWeightsToNumpyFile(const std::string &filePath, const std::string &arrayName);

    std::vector<size_t> getKeys();

    double getNorm();

    double getL1Norm();

    double getAverage();

    double getMinimum();

    double getMaximum();


private:

    static void mapToWeights(const std::unordered_map<size_t, double> &map, std::vector<double> &data);

    static void weightsToMap(std::unordered_map<size_t, double> &map, const double *weights);
};


#endif //NEUVISYS_WEIGHTMAP_HPP
