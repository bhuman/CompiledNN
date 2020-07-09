/**
 * @file ONNX.cpp
 *
 * This file implements a class that reads ONNX models.
 *
 * @author Arne Hasselbring
 */

#include "ONNX.h"
#include "Platform/BHAssert.h"
#include <onnx.pb.h>
#include <fstream>
#include <iostream>

namespace NeuralNetwork
{
  void ONNX::read(const std::string &file)
  {
    std::vector<char> binary;
    {
      std::ifstream f(file, std::ifstream::binary);
      if(!f.is_open())
        FAIL("Model \"" << file << "\" could not be opened.");
      f.seekg(0, std::ifstream::end);
      binary.resize(f.tellg());
      f.seekg(0);
      f.read(binary.data(), binary.size());
      f.close();
    }

    onnx::ModelProto model;
    model.ParseFromArray(binary.data(), binary.size());
    ASSERT(model.has_graph());

    // const auto& graph = model.graph();
    // ...
    FAIL("Not yet implemented.");
  }
}

