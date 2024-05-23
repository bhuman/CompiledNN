/**
 * @author Felix Thielke
 */

#include "ZeroPadding1D.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    void ZeroPadding1DCompiler::compile(x86::Assembler&, ActivationFunctionHandler&, const TensorPointerXf& input, const TensorPointerXf& output) const
    {
      ASSERT(input.rank() == 2);
      ASSERT(output.rank() == 2);
      ASSERT(input.dims(0) + p.padding[ZeroPadding1DLayer::LEFT] + p.padding[ZeroPadding1DLayer::RIGHT] == output.dims(0));
      ASSERT(input.dims(1) == output.dims(1));
      ASSERT(input.data() == output.data());

      if(p.padding[ZeroPadding1DLayer::LEFT] > 0)
      {
        // Copy data
        input.size();
      }

      FAIL("Not implemented");
    }
  }
}
