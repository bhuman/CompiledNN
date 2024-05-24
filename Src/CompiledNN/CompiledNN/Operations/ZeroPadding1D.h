/**
 * @author Felix Thielke
 */

#pragma once

#include <vector>
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

    private:
      int copyLoopPacked(x86::Assembler& a, const int size, const int numRegs, const bool inputAligned, const bool outputAligned) const;
      void copyLoopSingle(x86::Assembler& a, const int size) const;
      int zeroLoopPacked(x86::Assembler& a, const int size, const int numRegs, const bool aligned, std::vector<bool>& xmmIsZero) const;
      void zeroLoopSingle(x86::Assembler& a, const int size, std::vector<bool>& xmmIsZero) const;
    };
  }
}
