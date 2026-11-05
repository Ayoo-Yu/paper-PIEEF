#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>

void modifyBoundaryTypes(const std::string& filePath, const std::unordered_map<std::string, std::string>& boundaryUpdates) {
    std::ifstream infile(filePath.c_str());
    if (!infile.is_open()) {
        std::cerr << "Failed to open the file: " << filePath << std::endl;
        return;
    }

    std::stringstream buffer;
    std::string line;
    std::string currentBoundary;
    bool inBoundaryBlock = false;

    while (std::getline(infile, line)) {
        if (boundaryUpdates.find(line) != boundaryUpdates.end()) {
            currentBoundary = line;
            inBoundaryBlock = true;
        }

        if (inBoundaryBlock && line.find("type") != std::string::npos) {
            std::string updatedType = boundaryUpdates.at(currentBoundary);
            line = "        type            " + updatedType + ";";
            inBoundaryBlock = false;
        }

        buffer << line << "\n";
    }

    infile.close();

    std::ofstream outfile(filePath.c_str());
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the file for writing: " << filePath << std::endl;
        return;
    }

    outfile << buffer.str();
    outfile.close();
}

int main() {
    std::unordered_map<std::string, std::string> boundaryUpdates = {
        {"    inlet", "patch"},
        {"    outlet", "patch"},
        {"    front", "symmetry"},      // <--- 修改
        {"    back", "symmetry"},       // <--- 修改
        {"    bot", "wall"},
        {"    top", "symmetry"}         // <--- 修改
    };

    std::string boundaryFilePath = "constant/polyMesh/boundary";
    modifyBoundaryTypes(boundaryFilePath, boundaryUpdates);

    return 0;
}

