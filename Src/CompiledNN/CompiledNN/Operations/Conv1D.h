/**
 * @author Felix Thielke
 */

#pragma once

#include "../ActivationFunctions.h"
#include "../CompiledNNImplBase.h"
#include "BatchNormalization.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct Conv1DCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        const BatchNormalizationCompiler::Parameters* batchNormalization = nullptr;
        const Tensor<float, 1>* weights;
        const std::vector<float>* biases;
        unsigned int stride;
        ActivationFunctionDescriptor postActivation;

        bool operator==(const Parameters& other) const
        {
          return batchNormalization == other.batchNormalization &&
                 weights == other.weights &&
                 biases == other.biases &&
                 stride == other.stride &&
                 postActivation == other.postActivation;
        }
      };
      const Parameters p;

      Conv1DCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline bool canBeInplace() const override
      {
        return false;
      }

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 2);
        return {{(inputDimensions[0] - p.weights->dims(0) + p.stride) / p.stride, p.weights->dims(2)}};
      }
    };
  }
}
