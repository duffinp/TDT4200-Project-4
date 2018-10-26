#pragma once

#include <string>
#include "cuda_runtime.h"
#include "geometry.hpp"

std::vector<GPUMesh> loadWavefrontGPU(std::string const srcFile, bool quiet = true);