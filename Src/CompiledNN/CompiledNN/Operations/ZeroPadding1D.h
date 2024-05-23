/**
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct ZeroPadding1DCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        std::array<unsigned int, 2> padding;

        bool operator==(const Parameters& other) const
        {
          return padding == other.padding;
        }
      };
      const Parameters p;

      ZeroPadding1DCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override { return true; }
      void initialize() override {}
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 2);
        return {{
            inputDimensions[0] + p.padding[ZeroPadding1DLayer::LEFT] + p.padding[ZeroPadding1DLayer::RIGHT],
            inputDimensions[1]
          }};
      }
    };
  }
}
