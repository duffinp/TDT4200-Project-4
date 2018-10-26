#include "gpurasteriser.cuh"
#include "utilities/OBJLoader.hpp"
#include <vector>
#include <iomanip>
#include <chrono>
#include <limits>
#include <iostream>
#include <algorithm>
#include "cuda_runtime.h"
#include "utilities/cuda_error_helper.hpp"


// UTILITY FUNCTIONS HAVE BEEN MOVED INTO THE KERNEL SOURCE FILE ITSELF
// CUDA relocatable and separable compilation is possible, but due to the many possible
// problems it can cause on different platforms, I decided to take the safe route instead
// and make sure it would compile fine for everyone. That implies moving everything into
// one file unfortunately.

class globalLight {
public:
	float3 direction;
	float3 colour;
	__host__ __device__ globalLight(float3 const vdirection, float3 const vcolour) : direction(vdirection), colour(vcolour) {}
};

__host__ __device__ float dotGPU(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 normalizeGPU(float3 v)
{
    float invLen = 1.0f / sqrtf(dotGPU(v, v));
    v.x *= invLen;
    v.y *= invLen;
    v.z *= invLen;
    return v;
}

// Utility function if you'd like to convert the depth buffer to an integer format.
__host__ __device__ int depthFloatToInt(float value) {
	value = (value + 1.0f) * 0.5f;
    return static_cast<int>(static_cast<double>(value) * static_cast<double>(16777216)); 
}

__host__ __device__ bool isPointInTriangle(
		float4 const &v0, float4 const &v1, float4 const &v2,
		unsigned int const x, unsigned int const y,
		float &u, float &v, float &w) {
		u = (((v1.y - v2.y) * (x    - v2.x)) + ((v2.x - v1.x) * (y    - v2.y))) /
				 	 (((v1.y - v2.y) * (v0.x - v2.x)) + ((v2.x - v1.x) * (v0.y - v2.y)));
		if (u < 0) {
			return false;
		}
		v = (((v2.y - v0.y) * (x    - v2.x)) + ((v0.x - v2.x) * (y    - v2.y))) /
					(((v1.y - v2.y) * (v0.x - v2.x)) + ((v2.x - v1.x) * (v0.y - v2.y)));
		if (v < 0) {
			return false;
		}
		w = 1 - u - v;
		if (w < 0) {
			return false;
		}
		return true;
}

__host__ __device__ float3 computeInterpolatedNormal(
		float3 const &normal0,
		float3 const &normal1,
		float3 const &normal2,
		float3 const &weights
	) {
	float3 weightedN0, weightedN1, weightedN2;

	weightedN0.x = (normal0.x * weights.x);
	weightedN0.y = (normal0.y * weights.x);
	weightedN0.z = (normal0.z * weights.x);

	weightedN1.x = (normal1.x * weights.y);
	weightedN1.y = (normal1.y * weights.y);
	weightedN1.z = (normal1.z * weights.y);

	weightedN2.x = (normal2.x * weights.z);
	weightedN2.y = (normal2.y * weights.z);
	weightedN2.z = (normal2.z * weights.z);

	float3 weightedNormal;

	weightedNormal.x = weightedN0.x + weightedN1.x + weightedN2.x;
	weightedNormal.y = weightedN0.y + weightedN1.y + weightedN2.y;
	weightedNormal.z = weightedN0.z + weightedN1.z + weightedN2.z;

	return normalizeGPU(weightedNormal);
}

__host__ __device__ float computeDepth(
		float4 const &v0, float4 const &v1, float4 const &v2,
		float3 const &weights) {
	return weights.x * v0.z + weights.y * v1.z + weights.z * v2.z;
}





// ORIGINAL SOURCE FILE IS STARTING HERE

struct workItemGPU {
    float scale;
    float3 distanceOffset;

    workItemGPU(float& scale_, float3& distanceOffset_) : scale(scale_), distanceOffset(distanceOffset_) {}
    workItemGPU() : scale(1), distanceOffset(make_float3(0, 0, 0)) {}
};

void runVertexShader( float4 &vertex,
                      float3 positionOffset,
                      float scale,
					  unsigned int const width,
					  unsigned int const height,
				  	  float const rotationAngle = 0)
{
	float const pi = 3.1415926f;
	// The matrices defined below are the ones used to transform the vertices and normals.

	// This projection matrix assumes a 16:9 aspect ratio, and an field of view (FOV) of 90 degrees.
	mat4x4 const projectionMatrix(
		0.347270,   0, 			0, 		0,
		0,	  		0.617370, 	0,		0,
		0,	  		0,			-1, 	-0.2f,
		0,	  		0,			-1,		0);

	mat4x4 translationMatrix(
		1,			0,			0,			0 + positionOffset.x /*X*/,
		0,			1,			0,			0 + positionOffset.y /*Y*/,
		0,			0,			1,			-10 + positionOffset.z /*Z*/,
		0,			0,			0,			1);

	mat4x4 scaleMatrix(
		scale/*X*/,	0,			0,				0,
		0, 			scale/*Y*/, 0,				0,
		0, 			0,			scale/*Z*/, 	0,
		0, 			0,			0,				1);

	mat4x4 const rotationMatrixX(
		1,			0,				0, 				0,
		0, 			cosf(0), 	-sinf(0),	0,
		0, 			sinf(0),	cosf(0), 	0,
		0, 			0,				0,				1);

	float const rotationAngleRad = (pi / 4.0f) + (rotationAngle / (180.0f/pi));

	mat4x4 const rotationMatrixY(
		cosf(rotationAngleRad), 0, sinf(rotationAngleRad), 0,
		0, 1, 0, 0,
		-sinf(rotationAngleRad), 0, cosf(rotationAngleRad), 	0,
		0, 0, 0, 1);

	mat4x4 const rotationMatrixZ(
		cosf(pi),	-sinf(pi),	0,			0,
		sinf(pi), 	cosf(pi), 	0,			0,
		0,				0,				1,			0,
		0, 				0,				0,			1);

	mat4x4 const MVP =
		projectionMatrix * translationMatrix * rotationMatrixX * rotationMatrixY * rotationMatrixZ * scaleMatrix;

		float4 transformed = (MVP * vertex);

    vertex.x = transformed.x / transformed.w;
    vertex.y = transformed.y / transformed.w;
    vertex.z = transformed.z / transformed.w;
    vertex.w = 1.0;

    vertex.x = (vertex.x + 0.5f) * (float) width;
    vertex.y = (vertex.y + 0.5f) * (float) height;
}


void runFragmentShader( unsigned char* frameBuffer,
						unsigned int const baseIndex,
						GPUMesh &mesh,
						unsigned int triangleIndex,
						float3 const &weights)
{
	float3 normal = computeInterpolatedNormal(
            mesh.normals[3 * triangleIndex + 0],
            mesh.normals[3 * triangleIndex + 1],
            mesh.normals[3 * triangleIndex + 2],
			weights);

    float3 colour = make_float3(0.0f, 0.0f, 0.0f);

    const unsigned int lightSourceCount = 1;
    const globalLight lightSources[lightSourceCount] = {{make_float3(0.3f, 0.5f, 1.0f), make_float3(1.0f, 1.0f, 1.0f)}};

	for (unsigned int lightSource = 0; lightSource < lightSourceCount; lightSource++) {
		globalLight l = lightSources[lightSource];
		float lightNormalDotProduct = 
			normal.x * l.direction.x + normal.y * l.direction.y + normal.z * l.direction.z;

		float3 diffuseReflectionColour;
		diffuseReflectionColour.x = mesh.objectDiffuseColour.x * l.colour.x;
		diffuseReflectionColour.y = mesh.objectDiffuseColour.y * l.colour.y;
		diffuseReflectionColour.z = mesh.objectDiffuseColour.z * l.colour.z;

		colour.x += diffuseReflectionColour.x * lightNormalDotProduct;
		colour.y += diffuseReflectionColour.y * lightNormalDotProduct;
		colour.z += diffuseReflectionColour.z * lightNormalDotProduct;
	}

    colour.x = fminf(fmaxf(colour.x, 0.0f), 1.0f);
    colour.y = fminf(fmaxf(colour.y, 0.0f), 1.0f);
    colour.z = fminf(fmaxf(colour.z, 0.0f), 1.0f);

    frameBuffer[4 * baseIndex + 0] = colour.x * 255.0f;
    frameBuffer[4 * baseIndex + 1] = colour.y * 255.0f;
    frameBuffer[4 * baseIndex + 2] = colour.z * 255.0f;
    frameBuffer[4 * baseIndex + 3] = 255;

}

/**
 * The main procedure which rasterises all triangles on the framebuffer
 * @param transformedMesh         Transformed mesh object
 * @param frameBuffer             frame buffer for the rendered image
 * @param depthBuffer             depth buffer for every pixel on the image
 * @param width                   width of the image
 * @param height                  height of the image
 */
void rasteriseTriangle( float4 &v0, float4 &v1, float4 &v2,
                        GPUMesh &mesh,
                        unsigned int triangleIndex,
                        unsigned char* frameBuffer,
                        int* depthBuffer,
                        unsigned int const width,
                        unsigned int const height ) {

    // Compute the bounding box of the triangle.
    // Pixels that are intersecting with the triangle can only lie in this rectangle
	unsigned int minx = unsigned(floorf(fminf(fminf(v0.x, v1.x), v2.x)));
	unsigned int maxx = unsigned(ceilf(fmaxf(fmaxf(v0.x, v1.x), v2.x)));
	unsigned int miny = unsigned(floorf(fminf(fminf(v0.y, v1.y), v2.y)));
	unsigned int maxy = unsigned(ceilf(fmaxf(fmaxf(v0.y, v1.y), v2.y)));

	// Make sure the screen coordinates stay inside the window
    // This ensures parts of the triangle that are outside the
    // view of the camera are not drawn.
	minx = fmaxf(minx, (unsigned int) 0);
	maxx = fminf(maxx, width);
	miny = fmaxf(miny, (unsigned int) 0);
	maxy = fminf(maxy, height);

	// We iterate over each pixel in the triangle's bounding box
	for (unsigned int x = minx; x < maxx; x++) {
		for (unsigned int y = miny; y < maxy; y++) {
			float u, v, w;
			// For each point in the bounding box, determine whether that point lies inside the triangle
			if (isPointInTriangle(v0, v1, v2, x, y, u, v, w)) {
				// If it does, compute the distance between that point on the triangle and the screen
				float pixelDepth = computeDepth(v0, v1, v2, make_float3(u, v, w));
				// If the point is closer than any point we have seen thus far, render it.
				// Otherwise it is hidden behind another object, and we can throw it away
				// Because it will be invisible anyway.
                if (pixelDepth >= -1 && pixelDepth <= 1) {
					int pixelDepthConverted = depthFloatToInt(pixelDepth);
                 	if (pixelDepthConverted < depthBuffer[y * width + x]) {
					    // If it is, we update the depth buffer to the new depth.
					    depthBuffer[y * width + x] = pixelDepthConverted;

					    // And finally we determine the colour of the pixel, now that 
					    // we know our pixel is the closest we have seen thus far.
						runFragmentShader(frameBuffer, x + (width * y), mesh, triangleIndex, make_float3(u, v, w));
					}
				}
			}
		}
	}
}


void renderMeshes(
        unsigned long totalItemsToRender,
        workItemGPU* workQueue,
        GPUMesh* meshes,
        unsigned int meshCount,
        unsigned int width,
        unsigned int height,
        unsigned char* frameBuffer,
        int* depthBuffer
) {

    for(unsigned int item = 0; item < totalItemsToRender; item++) {
        workItemGPU objectToRender = workQueue[item];
        for (unsigned int meshIndex = 0; meshIndex < meshCount; meshIndex++) {
            for(unsigned int triangleIndex = 0; triangleIndex < meshes[meshIndex].vertexCount / 3; triangleIndex++) {
                float4 v0 = meshes[meshIndex].vertices[triangleIndex * 3 + 0];
                float4 v1 = meshes[meshIndex].vertices[triangleIndex * 3 + 1];
                float4 v2 = meshes[meshIndex].vertices[triangleIndex * 3 + 2];

                runVertexShader(v0, objectToRender.distanceOffset, objectToRender.scale, width, height);
                runVertexShader(v1, objectToRender.distanceOffset, objectToRender.scale, width, height);
                runVertexShader(v2, objectToRender.distanceOffset, objectToRender.scale, width, height);

                rasteriseTriangle(v0, v1, v2, meshes[meshIndex], triangleIndex, frameBuffer, depthBuffer, width, height);
            }
        }
    }
}



void fillWorkQueue(
        workItemGPU* workQueue,
        float largestBoundingBoxSide,
        int depthLimit,
        unsigned long* nextIndexInQueue,
        float scale = 1.0,
        float3 distanceOffset = {0, 0, 0}) {

    // Queue a work item at the current scale and location
    workQueue[*nextIndexInQueue] = {scale, distanceOffset};
    (*nextIndexInQueue)++;

    // Check whether we've reached the recursive depth of the fractal we want to reach
    depthLimit--;
    if(depthLimit == 0) {
        return;
    }

    // Now we recursively draw the meshes in a smaller size
    for(int offsetX = -1; offsetX <= 1; offsetX++) {
        for(int offsetY = -1; offsetY <= 1; offsetY++) {
            for(int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                float3 offset = make_float3(offsetX,offsetY,offsetZ);
                // We draw the new objects in a grid around the "main" one.
                // We thus skip the location of the object itself.
                if(offsetX == 0 && offsetY == 0 && offsetZ == 0) {
                    continue;
                }

                float smallerScale = scale / 3.0f;
                float3 displacedOffset = make_float3(
                        distanceOffset.x + offset.x * (largestBoundingBoxSide / 2.0f) * scale,
                        distanceOffset.y + offset.y * (largestBoundingBoxSide / 2.0f) * scale,
                        distanceOffset.z + offset.z * (largestBoundingBoxSide / 2.0f) * scale
                );

                fillWorkQueue(workQueue, largestBoundingBoxSide, depthLimit, nextIndexInQueue, smallerScale, displacedOffset);
            }
        }
    }

}

// This function kicks off the rasterisation process.
std::vector<unsigned char> rasteriseGPU(std::string inputFile, unsigned int width, unsigned int height, unsigned int depthLimit) {
    std::cout << "Rendering an image on the GPU.." << std::endl;
    std::cout << "Loading '" << inputFile << "' file... " << std::endl;

    std::vector<GPUMesh> meshes = loadWavefrontGPU(inputFile, false);

    // We first need to allocate some buffers.
    // The framebuffer contains the image being rendered.
    unsigned char* frameBuffer = new unsigned char[width * height * 4];
    // The depth buffer is used to make sure that objects closer to the camera occlude/obscure objects that are behind it
    for (unsigned int i = 0; i < (4 * width * height); i+=4) {
		frameBuffer[i + 0] = 0;
		frameBuffer[i + 1] = 0;
		frameBuffer[i + 2] = 0;
		frameBuffer[i + 3] = 255;
	}

	int* depthBuffer = new int[width * height];
	for(unsigned int i = 0; i < width * height; i++) {
    	depthBuffer[i] = 16777216; // = 2 ^ 24
    }

    float3 boundingBoxMin = make_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    float3 boundingBoxMax = make_float3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

    std::cout << "Rendering image... " << std::endl;

    for(unsigned int i = 0; i < meshes.size(); i++) {
        for(unsigned int vertex = 0; vertex < meshes.at(i).vertexCount; vertex++) {
            boundingBoxMin.x = std::min(boundingBoxMin.x, meshes.at(i).vertices[vertex].x);
            boundingBoxMin.y = std::min(boundingBoxMin.y, meshes.at(i).vertices[vertex].y);
            boundingBoxMin.z = std::min(boundingBoxMin.z, meshes.at(i).vertices[vertex].z);

            boundingBoxMax.x = std::max(boundingBoxMax.x, meshes.at(i).vertices[vertex].x);
            boundingBoxMax.y = std::max(boundingBoxMax.y, meshes.at(i).vertices[vertex].y);
            boundingBoxMax.z = std::max(boundingBoxMax.z, meshes.at(i).vertices[vertex].z);
        }
    }

    float3 boundingBoxDimensions = make_float3(
            boundingBoxMax.x - boundingBoxMin.x,
            boundingBoxMax.y - boundingBoxMin.y,
            boundingBoxMax.z - boundingBoxMin.z);
    float largestBoundingBoxSide = std::max(std::max(boundingBoxDimensions.x, boundingBoxDimensions.y), boundingBoxDimensions.z);

    // Each recursion level splits up the lowest level nodes into 28 smaller ones.
    // This regularity means we can calculate the total number of objects we need to render
    // which we can of course preallocate
    unsigned long totalItemsToRender = 0;
    for(unsigned long level = 0; level < depthLimit; level++) {
        totalItemsToRender += std::pow(26ul, level);
    }

    workItemGPU* workQueue = new workItemGPU[totalItemsToRender];

    std::cout << "Number of items to be rendered: " << totalItemsToRender << std::endl;

    unsigned long counter = 0;
    fillWorkQueue(workQueue, largestBoundingBoxSide, depthLimit, &counter);

	renderMeshes(
			totalItemsToRender, workQueue,
			meshes.data(), meshes.size(),
			width, height, frameBuffer, depthBuffer);

    std::cout << "Finished!" << std::endl;

    // Copy the output picture into a vector so that the image dump code is happy :)
    std::vector<unsigned char> outputFramebuffer(frameBuffer, frameBuffer + (width * height * 4));

    return outputFramebuffer;
}