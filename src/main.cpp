#include <iostream>
#include "common/arrrgh.hpp"
#include "common/lodepng.h"
#include "cpu/cpurasteriser.hpp"
#include "gpu/gpurasteriser.cuh"

int main(int argc, const char **argv) {
	const std::string defaultInput("../input/spheres.obj");
	const std::string defaultOutput("../output/sphere.png");
	const unsigned int defaultWidth = 1920;
	const unsigned int defaultHeight = 1080;
	const unsigned int defaultDepth = 3;
	arrrgh::parser parser("gpurender", "Renders raster images on the CPU or GPU");
	const auto& showHelp = parser.add<bool>(
		"help", 
		"Show this help message", 
		'h', 
		arrrgh::Optional, false);
	const auto& inputFile = parser.add<std::string>(
		"input", 
		"The location of the input model file.", 
		'i', 
		arrrgh::Optional, defaultInput);
	const auto& outputFile = parser.add<std::string>(
		"output",
		"The location where the output image should be written to.",
		'o',
		arrrgh::Optional, defaultOutput);
	const auto& forceGPU = parser.add<bool>(
		"enable-gpu",
		"Run the algorithm on the GPU",
		'g',
		arrrgh::Optional, false);
	const auto& width = parser.add<int>(
		"width",
		"Set the width of the output image in pixels",
		'w',
		arrrgh::Optional, defaultWidth);
	const auto& height = parser.add<int>(
		"height",
		"Set the height of the output image in pixels",
		'v',
		arrrgh::Optional, defaultHeight);
	const auto& depth = parser.add<int>(
		"depth",
		"Set the recursion depth of the sierpinski carpet",
		'd',
		arrrgh::Optional, defaultDepth);
	
	try
	{
		parser.parse(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error parsing arguments: " << e.what() << std::endl;
		parser.show_usage(std::cerr);
		exit(1);
	}

	// Show help if desired
	if(showHelp.value())
	{
		return 0;
	}

	std::vector<unsigned char> frameBuffer;

	if(forceGPU.value()) {
		frameBuffer = rasteriseGPU(inputFile.value(), width.value(), height.value(), depth.value());
	} else {
		frameBuffer = rasteriseCPU(inputFile.value(), width.value(), height.value(), depth.value());
	}

	std::cout << "Writing image to '" << outputFile.value() << "'..." << std::endl;

	unsigned error = lodepng::encode(outputFile.value(), frameBuffer, width.value(), height.value());

	if(error)
	{
		std::cout << "An error occurred while writing the image file: " << error << ": " << lodepng_error_text(error) << std::endl;
	}

	return 0;
}
