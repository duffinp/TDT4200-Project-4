#include "OBJLoader.hpp"
#include <algorithm>
#include <exception>
#include <regex>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include "cuda_runtime.h"

inline std::vector<std::string> split(std::string target, std::string delimiter)
{
	std::vector<std::string> res;
	size_t pos = 0;
	while ((pos = target.find(delimiter)) != std::string::npos) {
		res.push_back(target.substr(0, pos));
		target.erase(0, pos + delimiter.length());
	}
	res.push_back(target);
	return res;
}

inline std::vector<Material> loadMaterials(std::string const srcFile, bool quiet) {
	std::vector<Material> materials;
	std::ifstream objFile(srcFile);
	if (objFile.is_open()) {
		std::string line;
		while (std::getline(objFile, line)) {
			std::vector<std::string> parts = split(line, " ");
			if (parts.at(0) == "newmtl" && parts.size() >= 2) {
				materials.emplace_back(parts.at(1));
			} else if (materials.size() > 0) {
				Material &mtl = materials.back();
				if (parts.at(0) == "Ns" && parts.size() >= 2) {
					mtl.Ns = std::stof(parts.at(1));
				} else if (parts.at(0) == "Ni" && parts.size() >= 2) {
					mtl.Ni = std::stof(parts.at(1));
				} else if (parts.at(0) == "d" && parts.size() >= 2) {
					mtl.d = std::stof(parts.at(1));
				} else if (parts.at(0) == "illum" && parts.size() >= 2) {
					mtl.illum = std::stoi(parts.at(1));
				} else if (parts.at(0) == "Ka" && parts.size() >= 4) {
					mtl.Ka.x = std::stof(parts.at(1));
					mtl.Ka.y = std::stof(parts.at(2));
					mtl.Ka.z = std::stof(parts.at(3));
				} else if (parts.at(0) == "Kd" && parts.size() >= 4) {
					mtl.Kd.x = std::stof(parts.at(1));
					mtl.Kd.y = std::stof(parts.at(2));
					mtl.Kd.z = std::stof(parts.at(3));
				} else if (parts.at(0) == "Ks" && parts.size() >= 4) {
					mtl.Ks.x = std::stof(parts.at(1));
					mtl.Ks.y = std::stof(parts.at(2));
					mtl.Ks.z = std::stof(parts.at(3));
				} else if (parts.at(0) == "Ke" && parts.size() >= 4) {
					mtl.Ke.y = std::stof(parts.at(1));
					mtl.Ke.y = std::stof(parts.at(2));
					mtl.Ke.z = std::stof(parts.at(3));
				}
			}
		}
	} else {
		if (!quiet)
			std::cout << "[WARNING] file '" << srcFile << "' not found!" << std::endl;
	}
	return materials;
}

void copyGeometryDataIntoLastMesh(std::vector<GPUMesh> &meshes, std::vector<float4> &meshVertices,
                              std::vector<float3> &meshNormals) {
	float4* permanentMeshVertices = new float4[meshVertices.size()];
	float3* permanentMeshNormals = new float3[meshNormals.size()];

	copy(meshVertices.begin(), meshVertices.end(), permanentMeshVertices);
	copy(meshNormals.begin(), meshNormals.end(), permanentMeshNormals);

    meshes.back().vertexCount = meshVertices.size();

    meshVertices.resize(0);
    meshNormals.resize(0);

    meshes.back().normals = permanentMeshNormals;
    meshes.back().vertices = permanentMeshVertices;

}

std::vector<GPUMesh> loadWavefrontGPU(std::string const srcFile, bool quiet)
{
	std::vector<GPUMesh> meshes;
	std::ifstream objFile(srcFile);
	std::vector<float4> vertices;
	std::vector<float3> normals;

	std::vector<float4> meshVertices;
	std::vector<float3> meshNormals;

	std::vector<Material> materials = loadMaterials(std::regex_replace(srcFile, std::regex("\\b.obj$"), ".mtl"), quiet);

	if (objFile.is_open()) {
		std::string line;
		while (std::getline(objFile, line)) {

			std::vector<std::string> parts = split(line, " ");
			if (parts.size() > 0) {
				// New Mesh object
				if (parts.at(0) == "o" && parts.size() >= 2) {
					if(meshes.size() > 0) {
						copyGeometryDataIntoLastMesh(meshes, meshVertices, meshNormals);
					}
					meshes.emplace_back();
				} else if (parts.at(0) == "v" && parts.size() >= 4) {
					vertices.push_back(make_float4(
						std::stof(parts.at(1)),
						std::stof(parts.at(2)),
						std::stof(parts.at(3)),
						(parts.size() >= 5) ? std::stof(parts.at(4)) : 1.0f
					));
				} else if (parts.at(0) == "vn" && parts.size() >= 4) {
				   normals.push_back(make_float3(
					   std::stof(parts.at(1)),
					   std::stof(parts.at(2)),
					   std::stof(parts.at(3))
				   ));
			   } else if (parts.at(0) == "usemtl" && parts.size() >= 2) {
				   if (meshes.size() == 0) {
					   	if (!quiet) {
					   		std::cout << "[WARNING] mtl definition found, but no object" << std::endl;
							std::cout << "[WARNING] creating object 'noname'" << std::endl;
						}
						meshes.emplace_back();
						//continue;
				   }
				   GPUMesh &mesh = meshes.back();
				   bool found = false;
				   for (Material &m : materials) {
					   if (m.name == parts.at(1)) {
						   mesh.objectDiffuseColour = m.Kd;
						   found = true;
						   break;
					   }
				   }
				   if (!found && !quiet)
					   std::cout << "[WARNING] material '" << parts.at(1) << "' not found" << std::endl;
			   } else if (parts.at(0) == "f" && parts.size() >= 4) {
				   if (meshes.size() == 0) {
					   if (!quiet) {
						   	std::cout << "[WARNING] face definition found, but no object" << std::endl;
							//std::cout << "[WARNING] creating object 'noname'" << std::endl;
					   }
					   //meshes.emplace_back();
					   continue;
				   }

					GPUMesh &mesh = meshes.back();

					bool quadruple = (parts.size() >= 5) ? true : false;

					std::vector<std::string> parts1 = split(parts.at(1),"/");
					std::vector<std::string> parts2 = split(parts.at(2),"/");
					std::vector<std::string> parts3 = split(parts.at(3),"/");
					std::vector<std::string> parts4;
					if (quadruple) {
						parts4 = split(parts.at(4),"/");
					}

					if (parts1.size() < 1 || parts1.size() != parts2.size() || parts2.size() != parts3.size() || (quadruple && parts4.size() != parts1.size())) {
						if (!quiet)
							std::cout << "[WARNING] invalid face defintion '" << line << "'" << std::endl;
						continue;
					}

					mesh.hasNormals = (parts1.size() >= 3) ? true : false;

					size_t n1_index, n2_index, n3_index, n4_index;
					size_t v4_index;
					size_t v1_index = std::stoi(parts1.at(0)) - 1;
					size_t v2_index = std::stoi(parts2.at(0)) - 1;
					size_t v3_index = std::stoi(parts3.at(0)) - 1;

					if (quadruple) {
						v4_index = std::stoi(parts4.at(0)) - 1;
					}

					if (v1_index >= vertices.size() ||
						v2_index >= vertices.size() ||
						v3_index >= vertices.size() ||
						(quadruple && v4_index >= vertices.size())) {
								continue;
					}




					if (mesh.hasNormals) {
						n1_index = std::stoi(parts1.at(2)) - 1;
						n2_index = std::stoi(parts2.at(2)) - 1;
						n3_index = std::stoi(parts3.at(2)) - 1;
						if (quadruple) {
							n4_index = std::stoi(parts4.at(2)) - 1;
						}
						if (n1_index >= normals.size() ||
							n2_index >= normals.size() ||
							n3_index >= normals.size() ||
							(quadruple && n4_index >= normals.size())) {
									continue;
						}
					}

					if (quadruple) {
						meshVertices.push_back(vertices.at(v1_index));
						meshVertices.push_back(vertices.at(v3_index));
						meshVertices.push_back(vertices.at(v4_index));
						if (mesh.hasNormals) {
							meshNormals.push_back(normals.at(n1_index));
							meshNormals.push_back(normals.at(n3_index));
							meshNormals.push_back(normals.at(n4_index));
						} else {
							meshNormals.insert(meshNormals.end(), { 0.0f, 0.0f, 0.0f });
						}
						// mesh.faces.emplace_back(mesh.vertices.end()[-3],mesh.vertices.end()[-2],mesh.vertices.end()[-1],
						// 						mesh.textures.end()[-3],mesh.textures.end()[-2],mesh.textures.end()[-1],
						// 						mesh.normals.end()[-3],mesh.normals.end()[-2],mesh.normals.end()[-1]);
					}


					meshVertices.push_back(vertices.at(v1_index));
					meshVertices.push_back(vertices.at(v2_index));
					meshVertices.push_back(vertices.at(v3_index));
					if (mesh.hasNormals){
						meshNormals.push_back(normals.at(n1_index));
						meshNormals.push_back(normals.at(n2_index));
						meshNormals.push_back(normals.at(n3_index));
					} else {
						meshNormals.insert(meshNormals.end(), { 0.0f, 0.0f, 0.0f });
					}
				}
			}
		}
	} else {
		throw std::runtime_error("Reading OBJ file failed. This is usually because the operating system can't find it. Check if the relative path (to your terminal's working directory) is correct.");
	}
	copyGeometryDataIntoLastMesh(meshes, meshVertices, meshNormals);

	return meshes;
}
