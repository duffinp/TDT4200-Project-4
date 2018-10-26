#include <iostream>
#include "geometry.hpp"

Material Material::None = Material();

GPUMesh GPUMesh::clone() {
	GPUMesh clonedMesh;

	clonedMesh.vertices = new float4[this->vertexCount];
	std::copy(this->vertices, this->vertices + this->vertexCount, clonedMesh.vertices);

	clonedMesh.normals = new float3[this->vertexCount];
	std::copy(this->normals, this->normals + this->vertexCount, clonedMesh.normals);
	
	clonedMesh.objectDiffuseColour = this->objectDiffuseColour;

	clonedMesh.hasNormals = this->hasNormals;

	clonedMesh.vertexCount = this->vertexCount;

	return clonedMesh;
}

