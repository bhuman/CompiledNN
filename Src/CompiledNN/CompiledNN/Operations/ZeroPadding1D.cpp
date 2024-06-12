/**
 * @author Felix Thielke
 */

#include "ZeroPadding1D.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    int ZeroPadding1DCompiler::copyLoopPacked(x86::Assembler& a, const int size, const int numRegs, const bool inputAligned, const bool outputAligned) const
    {
      const int stepSize = numRegs * 4;
      if(size < stepSize || numRegs < 1)
        return size;

      const int numIterations = size / stepSize;

      Label loop;
      if(numIterations > 1)
      {
        loop = a.newLabel();
        a.mov(a.zcx(), imm(numIterations));
        a.bind(loop);
      }

      for(int i = 0; i < numRegs; i++)
      {
        if(inputAligned)
          a.movaps(x86::xmm(i), a.ptr_zsi(((i * 4) - stepSize) * sizeof(float)));
        else
          a.movups(x86::xmm(i), a.ptr_zsi(((i * 4) - stepSize) * sizeof(float)));
      }
      for(int i = 0; i < numRegs; i++)
      {
        if(outputAligned)
          a.movaps(a.ptr_zdi(((i * 4) - stepSize) * sizeof(float)), x86::xmm(i));
        else
          a.movups(a.ptr_zdi(((i * 4) - stepSize) * sizeof(float)), x86::xmm(i));
      }

      a.sub(a.zsi(), imm(stepSize * sizeof(float)));
      a.sub(a.zdi(), imm(stepSize * sizeof(float)));

      if(numIterations > 1)
      {
        a.dec(a.zcx());
        a.jnz(loop);
      }

      return size % stepSize;
    }

    void ZeroPadding1DCompiler::copyLoopSingle(x86::Assembler& a, const int size) const
    {
      for(int i = 0; i < size; i++)
      {
        a.movss(x86::xmm(i), a.ptr_zsi((i - size) * sizeof(float)));
      }
      for(int i = 0; i < size; i++)
      {
        a.movss(a.ptr_zdi((i - size) * sizeof(float)), x86::xmm(i));
      }
    }

    int ZeroPadding1DCompiler::zeroLoopPacked(x86::Assembler& a, const int size, const int numRegs, const bool aligned, std::vector<bool>& xmmIsZero) const
    {
      const int stepSize = numRegs * 4;
      if(size < stepSize || numRegs < 1)
        return size;

      const int numIterations = size / stepSize;

      for(int i = 0; i < numRegs; i++)
      {
        if(!xmmIsZero[i])
        {
          a.pxor(x86::xmm(i), x86::xmm(i));
          xmmIsZero[i] = true;
        }
      }

      Label loop;
      if(numIterations > 1)
      {
        loop = a.newLabel();
        a.mov(a.zcx(), imm(numIterations));
        a.bind(loop);
      }

      for(int i = 0; i < numRegs; i++)
      {
        if(aligned)
          a.movaps(a.ptr_zdi((i * 4) * sizeof(float)), x86::xmm(i));
        else
          a.movups(a.ptr_zdi((i * 4) * sizeof(float)), x86::xmm(i));
      }

      a.add(a.zdi(), imm(stepSize * sizeof(float)));

      if(numIterations > 1)
      {
        a.dec(a.zcx());
        a.jnz(loop);
      }

      return size % stepSize;
    }

    void ZeroPadding1DCompiler::zeroLoopSingle(x86::Assembler& a, const int size, std::vector<bool>& xmmIsZero) const
    {
      if(size > 0 && !xmmIsZero[0])
      {
        a.pxor(x86::xmm(0), x86::xmm(0));
        xmmIsZero[0] = true;
      }
      for(int i = 0; i < size; i++)
      {
        a.movss(a.ptr_zdi(i * sizeof(float)), x86::xmm(xmmIsZero[i] ? i : 0));
      }
    }

    void ZeroPadding1DCompiler::compile(x86::Assembler& a, ActivationFunctionHandler&, const TensorPointerXf& input, const TensorPointerXf& output) const
    {
      ASSERT(input.rank() == 2);
      ASSERT(output.rank() == 2);
      ASSERT(input.dims(0) + p.padding[ZeroPadding1DLayer::LEFT] + p.padding[ZeroPadding1DLayer::RIGHT] == output.dims(0));
      ASSERT(input.dims(1) == output.dims(1));

      std::vector<bool> xmmIsZero(settings.xmmRegs(), false);

      if(p.padding[ZeroPadding1DLayer::LEFT] > 0)
      {
        // Copy data
        a.mov(a.zsi(), imm(input.data() + input.size()));
        a.mov(a.zdi(), imm(output.data() + (input.size() + p.padding[ZeroPadding1DLayer::LEFT] * input.dims(1))));

        const bool inputAligned = (input.size() % 4) == 0;
        const bool outputAligned = ((input.size() + p.padding[ZeroPadding1DLayer::LEFT] * input.dims(1)) % 4) == 0;
        int remainingSize = copyLoopPacked(a, static_cast<int>(input.size()), settings.xmmRegs(), inputAligned, outputAligned);
        if(remainingSize > 0)
        {
          remainingSize = copyLoopPacked(a, remainingSize, remainingSize / 4, inputAligned, outputAligned);
          if(remainingSize > 0)
            copyLoopSingle(a, remainingSize);
        }

        // Set left border to zero
        a.mov(a.zdi(), imm(output.data()));
        remainingSize = zeroLoopPacked(a, p.padding[ZeroPadding1DLayer::LEFT] * input.dims(1), settings.xmmRegs(), true, xmmIsZero);
        if(remainingSize > 0)
        {
          remainingSize = zeroLoopPacked(a, remainingSize, remainingSize / 4, true, xmmIsZero);
          if(remainingSize > 0)
            zeroLoopSingle(a, remainingSize, xmmIsZero);
        }
      }

      if(p.padding[ZeroPadding1DLayer::RIGHT] > 0)
      {
        // Set right border to zero
        a.mov(a.zdi(), imm(output.data() + (input.size() + p.padding[ZeroPadding1DLayer::LEFT] * input.dims(1))));
        const bool aligned = ((input.size() + p.padding[ZeroPadding1DLayer::LEFT] * input.dims(1)) % 4) == 0;
        int remainingSize = zeroLoopPacked(a, p.padding[ZeroPadding1DLayer::RIGHT] * input.dims(1), settings.xmmRegs(), aligned, xmmIsZero);
        if(remainingSize > 0)
        {
          remainingSize = zeroLoopPacked(a, remainingSize, remainingSize / 4, aligned, xmmIsZero);
          if(remainingSize > 0)
            zeroLoopSingle(a, remainingSize, xmmIsZero);
        }
      }
    }
  }
}
