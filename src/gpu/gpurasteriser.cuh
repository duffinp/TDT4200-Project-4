#pragma once

#include <string>
#include <vector>

std::vector<unsigned char> rasteriseGPU(std::string inputFile,
                                      unsigned int width,
                                      unsigned int height,
                                      unsigned int depthLimit = 1);