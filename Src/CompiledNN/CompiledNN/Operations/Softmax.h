/**
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct SoftmaxCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        unsigned int dimension;

        bool operator==(const Parameters& other) const
        {
          return dimension == other.dimension;
        }
      };
      const Parameters p;

      SoftmaxCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override { return true; }

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;
    };
  }
}
