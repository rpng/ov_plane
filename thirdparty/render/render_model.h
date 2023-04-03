#ifndef __MODEL_H__
#define __MODEL_H__

#include "Eigen/Eigen"
#include <vector>

class Model {
private:
  std::vector<Eigen::Vector3d> verts_;
  std::vector<std::vector<int>> faces_;

public:
  Model(const char *filename);
  ~Model();
  int nverts();
  int nfaces();
  Eigen::Vector3d vert(int i);
  std::vector<int> face(int idx);
};

#endif //__MODEL_H__