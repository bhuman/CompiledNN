/**
 * @author Felix Thielke
 */

#include "Pooling1D.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    void Pooling1DCompiler::initialize()
    {
      if(p.method == PoolingMethod::average && p.kernelSize > 1)
      {
        constants.resize(1);
        constants.back().data.clear();
        const float factor = 1.f / static_cast<float>(p.kernelSize);
        for(unsigned int i = 4; i; --i)
          constants.back().data.emplace_back(factor);
      }
    }

    void Pooling1DCompiler::compile(x86::Assembler& a, ActivationFunctionHandler&, const TensorPointerXf& input, const TensorPointerXf& output) const
    {
      ASSERT(input.rank() == 2);
      ASSERT(output.rank() == 2);

      if(p.kernelSize <= 1 && p.stride <= 1)
        return;

      // Calculate padding (cf. https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/CXX11/src/Tensor/TensorImagePatch.h#L262)
      const bool validPadding = p.padding == PaddingType::valid;
      //const unsigned int paddingLeft = validPadding ? 0 : ((output.dims(0) - 1) * p.stride + p.kernelSize - input.dims(0)) / 2;
      if(validPadding)
        ASSERT(output.dims(0) == (input.dims(0) - p.kernelSize + p.stride) / p.stride);
      else
        ASSERT(output.dims(0) == (input.dims(0) + p.stride - 1) / p.stride);

      // Load input/output base addresses
      a.mov(a.zsi(), imm(input.data()));
      if(input.data() == output.data())
        a.mov(a.zdi(), a.zsi());
      else
        a.mov(a.zdi(), imm(output.data()));

      FAIL("Not implemented");
    }
  }
}
