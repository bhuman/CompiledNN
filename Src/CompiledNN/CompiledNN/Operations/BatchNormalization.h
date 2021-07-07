/**
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct BatchNormalizationCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        const std::vector<float>* factor;
        const std::vector<float>* offset;
        unsigned int dimension;
        unsigned int inputSize;

        bool operator==(const Parameters& other) const
        {
          return factor == other.factor &&
                 offset == other.offset &&
                 dimension == other.dimension &&
                 inputSize == other.inputSize;
        }
      };
      const Parameters p;

      BatchNormalizationCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override { return true; }

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

    private:
      unsigned int paramLength;
    };
  }
}
