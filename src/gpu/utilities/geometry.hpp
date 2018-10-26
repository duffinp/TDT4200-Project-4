#pragma once

#include <string>
#include <vector>
#include "mat4.hpp"
#include "cuda_runtime.h"

class Material {
	public:
		static Material None;
		std::string name;
		float Ns;
		float3 Ka;
		float3 Kd;
		float3 Ks;
		float3 Ke;
		float Ni;
		float d;
		unsigned int illum;

		__host__ Material() : name("noname"), Ns(100.0f), Ka(make_float3(1.0f, 1.0f, 1.0f)),
			Kd(make_float3(0.5f, 0.5f, 0.5f)), Ks(make_float3(0.5f, 0.5f, 0.5f)), Ke(make_float3(0.0f, 0.0f, 0.0f)),
			Ni(1.0), d(1.0), illum(2) {}

		__host__ Material(std::string _name) : name(_name), Ns(100.0f), Ka(make_float3(1.0f, 1.0f, 1.0f)),
			Kd(make_float3(0.5f, 0.5f, 0.5f)), Ks(make_float3(0.5f, 0.5f, 0.5f)), Ke(make_float3(0.0f, 0.0f, 0.0f)),
			Ni(1.0), d(1.0), illum(2) {}
};

class GPUMesh {
public:
	float4* vertices;
	float3* normals;

	unsigned long vertexCount = 0;

	float3 objectDiffuseColour;

	bool hasNormals;

	__host__ GPUMesh clone();
};
