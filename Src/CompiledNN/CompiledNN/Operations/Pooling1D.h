/**
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct Pooling1DCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        unsigned int kernelSize;
        unsigned int stride;
        PoolingMethod method;
        PaddingType padding;

        bool operator==(const Parameters& other) const
        {
          return kernelSize == other.kernelSize &&
                 stride == other.stride &&
                 method == other.method &&
                 padding == other.padding;
        }
      };
      const Parameters p;

      Pooling1DCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override
      {
        return p.stride >= p.kernelSize;
      }

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 2);
        return {{
          (inputDimensions[0] - (p.padding == PaddingType::valid ? p.kernelSize - 1 : 0) + p.stride - 1) / p.stride,
          inputDimensions[1]
        }};
      }

    private:
      void pool(x86::Assembler& a, const unsigned int padding, const unsigned int channels, bool& helperRegInitialized) const;
    };
  }
}
