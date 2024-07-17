/**
 * @author Felix Thielke
 */

#pragma once

#include "../ActivationFunctions.h"
#include "../CompiledNNImplBase.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct QuantizedInputConvStrided4x4WithReLUCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        const Tensor<float, 1>* weights;
        const std::vector<float>* biases;
        uint8_t scale;
        bool outputAsFloat;

        bool operator==(const Parameters& other) const
        {
          return weights == other.weights &&
                 biases == other.biases &&
                 scale == other.scale &&
                 outputAsFloat == other.outputAsFloat;
        }
      };
      const Parameters p;

      QuantizedInputConvStrided4x4WithReLUCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override
      {
        return false;
      }

      void initialize() override;
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 3);
        return {{inputDimensions[0] / 4, inputDimensions[1] / 4, p.weights->dims(3)}};
      }

    private:
      void convolutionForPixel(x86::Assembler& a, unsigned int pixelId) const;
      void outputAsFloat(x86::Assembler& a, unsigned int destOffset) const;
    };
  }
}
