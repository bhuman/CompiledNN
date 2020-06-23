/**
 * @file ZeroPadding2D.cpp
 *
 * This file defines a test for the ZeroPadding2D layer.
 *
 * @author Arne Hasselbring
 */

#include "CompiledNN/CompiledNN.h"
#include "CompiledNN/SimpleNN.h"
#include "CompiledNN/Model.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <tuple>

using namespace NeuralNetwork;

class ZeroPadding2DTest : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>>
{
  static const Node& buildNode(ZeroPadding2DLayer* l, const std::array<unsigned int, 4>& padding, unsigned int height, unsigned int width, unsigned int channels)
  {
    l->nodes.clear();
    l->padding = padding;

    l->nodes.emplace_back(l);
    Node& n = l->nodes.back();
    n.inputs.emplace_back(nullptr, 0, 0);
    n.inputDimensions.push_back({height, width, channels});
    l->calcOutputDimensions(n);
    for(std::size_t i = 0; i < n.outputDimensions.size(); ++i)
      n.outputs.emplace_back(l, 0, i);
    return n;
  }

  mutable std::mt19937 generator;

public:
  float getError() const
  {
    CompiledNN c;
    CompilationSettings settings;
    settings.useX64 = false;

    std::vector<TensorXf> testOutputTensors(1);

    std::uniform_real_distribution<float> inputDist(-1.f, 1.f);

    ZeroPadding2DLayer l;
    const Node& n = buildNode(&l, {std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), std::get<3>(GetParam())},
                              std::get<4>(GetParam()), std::get<5>(GetParam()), std::get<6>(GetParam()));

    float absError = 0.f;
    for(unsigned int i = 0; i < 5; ++i)
    {
      c.compile(n, settings);

      // To make sure that the layer fills in zeros correctly, put another value in the output tensor.
      std::fill(c.output(0).begin(), c.output(0).end(), 42.f);

      for(auto p = c.input(0).begin(); p < c.input(0).end(); p++)
        *p = inputDist(generator);

      SimpleNN::apply({TensorXf(c.input(0))}, testOutputTensors, n);
      c.apply();

      const float err = testOutputTensors[0].maxAbsError(c.output(0));
      if(err > absError)
        absError = err;
    }
    return absError;
  }
};

TEST_P(ZeroPadding2DTest, ProducesSameOutputAsSimpleNN)
{
  EXPECT_EQ(getError(), 0.f);
}

INSTANTIATE_TEST_CASE_P(Layers, ZeroPadding2DTest,
                        ::testing::Combine(/* padding[TOP] */ ::testing::Values(0u, 1u), /* padding[BOTTOM] */ ::testing::Values(0u, 1u),
                                           /* padding[LEFT] */ ::testing::Values(0u, 1u, 2u), /* padding[RIGHT] */ ::testing::Values(0u, 1u, 2u),
                                           /* height */ ::testing::Values(1u, 8u), /* width */ ::testing::Values(1u, 8u), /* channels */ ::testing::Values(1u, 4u, 8u)));
