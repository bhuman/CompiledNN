/**
 * @author Felix Thielke
 */

#include "Conv1D.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    void Conv1DCompiler::initialize()
    {
      // Declare constants
      constants.clear();

      // Store weights
      constants.emplace_back();
      NetworkConstants& weights = constants.back();
      weights.data.clear();
      ASSERT(p.weights->rank() == 3);
      unsigned int outputBatchSize = 4 * (settings.xmmRegs() - std::max(2u, ActivationFunctionHandler::neededSpares(p.postActivation)));
      for(unsigned int outputOffset = 0; outputOffset < p.weights->dims(2); outputOffset += outputBatchSize)
      {
        const unsigned int outputBatchEnd = std::min(outputOffset + outputBatchSize, p.weights->dims(2));

        for(unsigned int input = 0; input < p.weights->dims(0) * p.weights->dims(1); input += 4)
        {
          const unsigned int remainingInputs = std::min(4u, p.weights->dims(0) * p.weights->dims(1) - input);

          for(unsigned int shuffle = remainingInputs; shuffle; --shuffle)
          {
            for(unsigned int output = outputOffset; output < outputBatchEnd; output += 4)
            {
              const unsigned int remainingOutputs = std::min(4u, outputBatchEnd - output);

              for(unsigned int i = 0; i < remainingOutputs; i++)
              {
                const float w = (*p.weights)[(input + ((remainingInputs - shuffle + i) % remainingInputs)) * p.weights->dims(2) + output + i];
                if(p.batchNormalization)
                  weights.data.emplace_back(w * (*p.batchNormalization->factor)[output + i]);
                else
                  weights.data.emplace_back(w);
              }
              for(unsigned int i = remainingOutputs; i < 4; i++)
                weights.data.emplace_back(0.f);
            }
          }
        }
      }

      // Store biases
      if(p.biases || p.batchNormalization) {
        constants.emplace_back();
        NetworkConstants& biases = constants.back();
        if(p.biases)
          biases.data = *p.biases;
        else
          biases.data.resize(p.weights->dims(2), 0.f);
        if(p.batchNormalization)
        {
          for(size_t i = 0; i < biases.data.size(); i++)
            biases.data[i] = biases.data[i] * (*p.batchNormalization->factor)[i] + (*p.batchNormalization->offset)[i];
        }
      }
    }

    void Conv1DCompiler::compile(x86::Assembler& a, ActivationFunctionHandler&, const TensorPointerXf& input,
                                 [[maybe_unused]] const TensorPointerXf& output) const
    {
      ASSERT(input.rank() == 2);
      ASSERT(output.rank() == 2);
      ASSERT(input.dims(1) == p.weights->dims(1));
      ASSERT(output.dims(1) == p.weights->dims(2));

      //const NetworkConstants& weights = constants[0];

      // Load input/output base addresses
      a.mov(a.zsi(), imm(input.data()));
      a.mov(a.zdi(), a.zsi());

      FAIL("Not implemented");
    }
  }
}
