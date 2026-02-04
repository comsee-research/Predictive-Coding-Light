//
// Created for generalized spike recording
//

#ifndef NEUVISYS_DV_DATASET_SCANNER_HPP
#define NEUVISYS_DV_DATASET_SCANNER_HPP

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <regex>

#if __GNUC__ > 8
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif

/**
 * Represents a single class/category in a dataset
 */
struct DatasetClass {
    std::string name;           // Class name (e.g., "hand_clapping")
    std::string path;           // Full path to class folder
    int label;                  // Numeric label (0-indexed)
    size_t sampleCount;         // Number of samples in this class
    std::vector<std::string> samplePaths;  // Paths to individual samples
};

/**
 * Represents the structure of an entire dataset
 */
struct DatasetStructure {
    std::string basePath;
    std::vector<DatasetClass> classes;
    size_t totalSamples = 0;
    bool isValid = false;
    std::string errorMessage;
    
    [[nodiscard]] size_t getNumClasses() const { return classes.size(); }
    
    [[nodiscard]] std::vector<std::string> getClassNames() const {
        std::vector<std::string> names;
        names.reserve(classes.size());
        for (const auto& cls : classes) {
            names.push_back(cls.name);
        }
        return names;
    }
    
    [[nodiscard]] std::vector<int> getLabels() const {
        std::vector<int> labels;
        labels.reserve(classes.size());
        for (const auto& cls : classes) {
            labels.push_back(cls.label);
        }
        return labels;
    }
    
    [[nodiscard]] std::vector<std::string> getClassPaths() const {
        std::vector<std::string> paths;
        paths.reserve(classes.size());
        for (const auto& cls : classes) {
            paths.push_back(cls.path);
        }
        return paths;
    }
};

/**
 * Utility class for scanning and auto-detecting dataset structure from a path.
 * Supports multiple folder naming conventions:
 * - Numeric prefix: "0_class_name", "1_class_name"
 * - Plain names: "class_name" (alphabetical ordering)
 * - Nested structure: dataset/train/, dataset/test/
 */
class DatasetScanner {
public:
    /**
     * Scan a dataset directory and return its structure.
     * Automatically detects folder naming convention and assigns labels.
     * 
     * @param datasetPath Path to the dataset root directory
     * @param fileExtensions Optional filter for file extensions (e.g., {".h5", ".npz"})
     * @return DatasetStructure containing all discovered classes and samples
     */
    static DatasetStructure scan(const std::string& datasetPath, 
                                  const std::vector<std::string>& fileExtensions = {".h5", ".npz"});
    
    /**
     * Check if a path looks like a labeled dataset (has subdirectories with samples)
     */
    static bool isLabeledDataset(const std::string& path);
    
    /**
     * Check if a path is a flat directory of samples (no class subdirectories)
     */
    static bool isFlatDataset(const std::string& path);
    
    /**
     * Print dataset structure to stdout for debugging
     */
    static void printStructure(const DatasetStructure& structure);

private:
    /**
     * Extract numeric prefix from folder name if present
     * Returns -1 if no numeric prefix found
     * Examples: "0_hand_clapping" -> 0, "10_other" -> 10, "hand_clapping" -> -1
     */
    static int extractNumericPrefix(const std::string& folderName);
    
    /**
     * Extract class name from folder name (strips numeric prefix if present)
     * Examples: "0_hand_clapping" -> "hand_clapping", "hand_clapping" -> "hand_clapping"
     */
    static std::string extractClassName(const std::string& folderName);
    
    /**
     * Count files in a directory matching the given extensions
     */
    static std::vector<std::string> getSamplePaths(const std::string& dirPath, 
                                                    const std::vector<std::string>& extensions);
    
    /**
     * Check if folder contains valid sample files
     */
    static bool containsSamples(const std::string& dirPath, 
                                const std::vector<std::string>& extensions);
};

#endif //NEUVISYS_DV_DATASET_SCANNER_HPP
