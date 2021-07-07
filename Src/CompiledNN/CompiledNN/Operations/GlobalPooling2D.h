/**
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct GlobalPooling2DCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        PoolingMethod method;
        unsigned int imageSize = 0;

        bool operator==(const Parameters& other) const
        {
          return method == other.method &&
                 imageSize == other.imageSize;
        }
      };
      const Parameters p;

      GlobalPooling2DCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override
      {
        return true;
      }

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 3);
        return {inputDimensions[2]};
      }
    };
  }
}
