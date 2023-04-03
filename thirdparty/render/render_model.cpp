#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "render_model.h"

Model::Model(const char *filename) : verts_(), faces_() {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  if (in.fail())
    return;
  std::string line;
  double scale = 0.10;
  while (!in.eof()) {
    std::getline(in, line);
    std::istringstream iss(line.c_str());
    char trash;
    if (!line.compare(0, 2, "v ")) {
      iss >> trash;
      Eigen::Vector3d v;
      for (int i = 0; i < 3; i++) {
        float ft;
        iss >> ft;
        v(i) = scale * (double)ft;
      }
      verts_.push_back(v);
    } else if (!line.compare(0, 2, "f ")) {
      std::vector<int> f;
      // int itrash, idx;
      int idx;
      iss >> trash;
      // while (iss >> idx >> trash >> itrash >> trash >> itrash) {
      while (iss >> idx) {
        idx--; // in wavefront obj all indices start at 1, not zero
        f.push_back(idx);
      }
      faces_.push_back(f);
    }
  }
  std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << std::endl;
}

Model::~Model() {}

int Model::nverts() { return (int)verts_.size(); }

int Model::nfaces() { return (int)faces_.size(); }

std::vector<int> Model::face(int idx) { return faces_[idx]; }

Eigen::Vector3d Model::vert(int i) { return verts_[i]; }
