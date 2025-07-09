//
// Created by thomas on 06/07/22.
//

#ifndef NEUVISYS_WEIGHTMATRIX_HPP
#define NEUVISYS_WEIGHTMATRIX_HPP

#include <random>
#include <utility>

#include "cnpy/cnpy.h"

#include "Types.hpp"

class WeightMatrix {
    std::vector<double> m_data;
    std::vector<size_t> m_dimensions;

    static std::random_device m_rd;
    static std::mt19937 m_generator;
    static std::uniform_real_distribution<double> m_uniformRealDistr;
    static std::uniform_real_distribution<double> m_uniformRealDistrConst;

public:
    WeightMatrix();

    WeightMatrix(std::vector<size_t> dimensions);

    WeightMatrix(std::vector<size_t> dimensions, bool uniform, double normFactor);

    static void setSeed(size_t seed);

    double getMinimum();

    double getMaximum();

    int getNMaximum(int x, int y, int n);

    double getMedian();

    void antiNoise();

    void normalize(double normFactor);

    void MinMaxNorm(double normFactor);

    void pruning(double thresh);

    void operator=(WeightMatrix weight);

    void normalizeL1Norm(double normFactor);

    std::vector<size_t> getDimensions() const { return m_dimensions; }

    size_t getSize() const { return m_data.size(); }

    std::vector<double> getMatrix() { return m_data; }

    double &get(size_t a);

    double &get(size_t a, size_t b);

    double &get(size_t a, size_t b, size_t c);

    double &get(size_t a, size_t b, size_t c, size_t d);

    double &get(size_t a, size_t b, size_t c, size_t d, size_t e);

    void loadNumpyFile(const std::string &filePath);

    void loadNumpyFile(cnpy::npz_t &arrayNPZ, const std::string &arrayName);

    void saveWeightsToNumpyFile(const std::string &filePath);

    void saveWeightsToNumpyFile(const std::string &filePath, const std::string &arrayName);

    double getNorm();

    double getL1Norm();

private:

    static void mapToWeights(const std::vector<double> &map, std::vector<double> &data);

    static void weightsToMap(std::vector<double> &map, const double *weights);

};

#endif //NEUVISYS_WEIGHTMATRIX_HPP
