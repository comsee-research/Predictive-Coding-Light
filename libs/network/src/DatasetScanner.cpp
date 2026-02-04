//
// Created for generalized spike recording
//

#include "DatasetScanner.hpp"

DatasetStructure DatasetScanner::scan(const std::string& datasetPath, 
                                       const std::vector<std::string>& fileExtensions) {
    DatasetStructure structure;
    structure.basePath = datasetPath;
    
    // Check if path exists
    if (!fs::exists(datasetPath)) {
        structure.errorMessage = "Dataset path does not exist: " + datasetPath;
        return structure;
    }
    
    if (!fs::is_directory(datasetPath)) {
        structure.errorMessage = "Dataset path is not a directory: " + datasetPath;
        return structure;
    }
    
    // Collect all subdirectories that contain samples
    std::vector<std::pair<std::string, std::string>> classFolders;  // (folder_name, full_path)
    
    for (const auto& entry : fs::directory_iterator(datasetPath)) {
        if (fs::is_directory(entry.path())) {
            std::string folderName = entry.path().filename().string();
            std::string fullPath = entry.path().string();
            
            // Check if this folder contains sample files
            if (containsSamples(fullPath, fileExtensions)) {
                classFolders.emplace_back(folderName, fullPath);
            }
        }
    }
    
    if (classFolders.empty()) {
        // Check if it's a flat dataset (samples directly in the folder)
        if (containsSamples(datasetPath, fileExtensions)) {
            structure.errorMessage = "Dataset appears to be flat (no class subdirectories). "
                                    "Use a labeled dataset structure with class folders.";
        } else {
            structure.errorMessage = "No class folders with valid samples found in: " + datasetPath + 
                                    "\nExpected structure: dataset/class_0/*.h5, dataset/class_1/*.h5, ...";
        }
        return structure;
    }
    
    // Check if folders have numeric prefixes
    bool hasNumericPrefixes = true;
    for (const auto& [folderName, _] : classFolders) {
        if (extractNumericPrefix(folderName) < 0) {
            hasNumericPrefixes = false;
            break;
        }
    }
    
    // Sort folders: by numeric prefix if available, otherwise alphabetically
    if (hasNumericPrefixes) {
        std::sort(classFolders.begin(), classFolders.end(), 
            [](const auto& a, const auto& b) {
                return extractNumericPrefix(a.first) < extractNumericPrefix(b.first);
            });
    } else {
        std::sort(classFolders.begin(), classFolders.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
    }
    
    // Build class structures
    int label = 0;
    for (const auto& [folderName, fullPath] : classFolders) {
        DatasetClass cls;
        cls.name = extractClassName(folderName);
        cls.path = fullPath;
        cls.label = label++;
        cls.samplePaths = getSamplePaths(fullPath, fileExtensions);
        cls.sampleCount = cls.samplePaths.size();
        
        structure.totalSamples += cls.sampleCount;
        structure.classes.push_back(std::move(cls));
    }
    
    structure.isValid = true;
    return structure;
}

bool DatasetScanner::isLabeledDataset(const std::string& path) {
    if (!fs::exists(path) || !fs::is_directory(path)) {
        return false;
    }
    
    // Check if at least one subdirectory contains samples
    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_directory(entry.path())) {
            if (containsSamples(entry.path().string(), {".h5", ".npz"})) {
                return true;
            }
        }
    }
    return false;
}

bool DatasetScanner::isFlatDataset(const std::string& path) {
    if (!fs::exists(path) || !fs::is_directory(path)) {
        return false;
    }
    return containsSamples(path, {".h5", ".npz"}) && !isLabeledDataset(path);
}

void DatasetScanner::printStructure(const DatasetStructure& structure) {
    std::cout << "\n===== Dataset Structure =====" << std::endl;
    std::cout << "Base path: " << structure.basePath << std::endl;
    
    if (!structure.isValid) {
        std::cout << "ERROR: " << structure.errorMessage << std::endl;
        return;
    }
    
    std::cout << "Number of classes: " << structure.getNumClasses() << std::endl;
    std::cout << "Total samples: " << structure.totalSamples << std::endl;
    std::cout << "\nClasses:" << std::endl;
    
    for (const auto& cls : structure.classes) {
        std::cout << "  [" << cls.label << "] " << cls.name 
                  << " (" << cls.sampleCount << " samples)" << std::endl;
    }
    std::cout << "============================\n" << std::endl;
}

int DatasetScanner::extractNumericPrefix(const std::string& folderName) {
    // Match patterns like "0_name", "10_name", "123_name"
    std::regex prefixRegex("^(\\d+)_.*");
    std::smatch match;
    
    if (std::regex_match(folderName, match, prefixRegex)) {
        return std::stoi(match[1].str());
    }
    return -1;
}

std::string DatasetScanner::extractClassName(const std::string& folderName) {
    // Remove numeric prefix if present
    std::regex prefixRegex("^\\d+_(.*)");
    std::smatch match;
    
    if (std::regex_match(folderName, match, prefixRegex)) {
        return match[1].str();
    }
    return folderName;
}

std::vector<std::string> DatasetScanner::getSamplePaths(const std::string& dirPath, 
                                                         const std::vector<std::string>& extensions) {
    std::vector<std::string> paths;
    
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (fs::is_regular_file(entry.path())) {
            std::string ext = entry.path().extension().string();
            // Convert to lowercase for comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            for (const auto& validExt : extensions) {
                std::string validExtLower = validExt;
                std::transform(validExtLower.begin(), validExtLower.end(), 
                             validExtLower.begin(), ::tolower);
                if (ext == validExtLower) {
                    paths.push_back(entry.path().string());
                    break;
                }
            }
        }
    }
    
    // Sort paths for consistent ordering
    std::sort(paths.begin(), paths.end());
    return paths;
}

bool DatasetScanner::containsSamples(const std::string& dirPath, 
                                      const std::vector<std::string>& extensions) {
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (fs::is_regular_file(entry.path())) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            for (const auto& validExt : extensions) {
                std::string validExtLower = validExt;
                std::transform(validExtLower.begin(), validExtLower.end(), 
                             validExtLower.begin(), ::tolower);
                if (ext == validExtLower) {
                    return true;
                }
            }
        }
    }
    return false;
}
