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

    void Pooling1DCompiler::pool(x86::Assembler& a, const unsigned int padding, const unsigned int channels, bool& helperRegInitialized) const
    {
      const bool aligned = channels % 4 == 0;
      const bool isPadded = padding > 0;
      const unsigned int regsPerStep = aligned && !(isPadded && p.method == PoolingMethod::max) ? settings.xmmRegs() : settings.xmmRegs() - 1;
      const x86::Xmm helperReg = aligned ? x86::xmm(settings.xmmRegs() - 1) : x86::xmm(settings.xmmRegs() - 2);

      if(!helperRegInitialized && (channels + 3) / 4 < (aligned ? settings.xmmRegs() : settings.xmmRegs() - 1))
      {
        if(isPadded && p.method == PoolingMethod::max)
          a.xorps(helperReg, helperReg);
        else if(p.method == PoolingMethod::average && p.kernelSize > 1)
          a.movaps(helperReg, x86::ptr(constants.back().label));

        helperRegInitialized = true;
      }

      for(unsigned int channelOffset = 0; channelOffset < channels; channelOffset += 4 * regsPerStep)
      {
        const unsigned int processedChannels = std::min(regsPerStep * 4, channels - channelOffset);
        const unsigned int stepSize = (processedChannels + 3) / 4;

        // Apply filter
        bool first = true;
        for(unsigned int filterIndex = 0; filterIndex < p.kernelSize - padding; filterIndex++)
        {
          unsigned int offset = (filterIndex * channels + channelOffset) * sizeof(float);
          if(first)
          {
            for(unsigned int step = 0; step < stepSize; step++)
            {
              if(aligned)
                a.movaps(x86::xmm(step), a.ptr_zsi(offset));
              else
                a.movups(x86::xmm(step), a.ptr_zsi(offset));
              offset += 4 * sizeof(float);
            }

            first = false;
          }
          else
          {
            if(aligned)
            {
              for(unsigned int step = 0; step < stepSize; step++)
              {
                if(p.method == PoolingMethod::average)
                  a.addps(x86::xmm(step), a.ptr_zsi(offset));
                else // method == Pooling2DLayer::PoolingMethod::max
                  a.maxps(x86::xmm(step), a.ptr_zsi(offset));
                offset += 4 * sizeof(float);
              }
            }
            else
            {
              const unsigned int helperOffset = stepSize;
              const unsigned int helperCount = settings.xmmRegs() - stepSize;
              unsigned int helper = 0;
              for(unsigned int step = 0; step < stepSize;)
              {
                a.movups(x86::xmm(helperOffset + helper), a.ptr_zsi(offset));
                step++;
                offset += 4 * sizeof(float);
                helper++;

                if(helper == helperCount)
                {
                  for(helper = 0; helper < helperCount; helper++)
                  {
                    if(p.method == PoolingMethod::average)
                      a.addps(x86::xmm(step - helperCount + helper), x86::xmm(helperOffset + helper));
                    else // method == Pooling2DLayer::PoolingMethod::max
                      a.maxps(x86::xmm(step - helperCount + helper), x86::xmm(helperOffset + helper));
                  }

                  helper = 0;
                }
              }

              if(helper != 0)
              {
                for(unsigned int i = 0; i < helper; i++)
                {
                  if(p.method == PoolingMethod::average)
                    a.addps(x86::xmm(stepSize - (helper - i)), x86::xmm(helperOffset + i));
                  else // method == Pooling2DLayer::PoolingMethod::max
                    a.maxps(x86::xmm(stepSize - (helper - i)), x86::xmm(helperOffset + i));
                }
              }
            }
          }
        }

        if(isPadded && p.method == PoolingMethod::max)
        {
          if(!helperRegInitialized)
          {
            a.xorps(x86::xmm(settings.xmmRegs() - 1), x86::xmm(settings.xmmRegs() - 1));
            for(unsigned int step = 0; step < stepSize; step++)
              a.maxps(x86::xmm(step), x86::xmm(settings.xmmRegs() - 1));
          }
          else
          {
            a.xorps(helperReg, helperReg);
            for(unsigned int step = 0; step < stepSize; step++)
              a.maxps(x86::xmm(step), helperReg);
          }
        }
        if(p.method == PoolingMethod::average && p.kernelSize > 1)
        {
          if(!helperRegInitialized)
          {
            for(unsigned int step = 0; step < stepSize; step++)
              a.mulps(x86::xmm(step), x86::ptr(constants.back().label));
          }
          else
          {
            for(unsigned int step = 0; step < stepSize; step++)
              a.mulps(x86::xmm(step), helperReg);
          }
        }

        // Store results
        for(unsigned int step = 0; step < stepSize; step++)
        {
          if(aligned)
            a.movaps(a.ptr_zdi((channelOffset + step * 4) * sizeof(float)), x86::xmm(step));
          else
            a.movups(a.ptr_zdi((channelOffset + step * 4) * sizeof(float)), x86::xmm(step));
        }
      }

      a.add(a.zdi(), imm(channels * sizeof(float)));
    }

    void Pooling1DCompiler::compile(x86::Assembler& a, ActivationFunctionHandler&, const TensorPointerXf& input, const TensorPointerXf& output) const
    {
      ASSERT(input.rank() == 2);
      ASSERT(output.rank() == 2);
      const unsigned int inputWidth = input.dims(0);
      const unsigned int outputWidth = output.dims(0);
      const unsigned int channels = input.dims(1);

      if(p.kernelSize <= 1 && p.stride <= 1)
        return;

      // Calculate padding (cf. https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/CXX11/src/Tensor/TensorImagePatch.h#L262)
      const bool validPadding = p.padding == PaddingType::valid;
      const unsigned int paddingLeft = validPadding ? 0 : ((outputWidth - 1) * p.stride + p.kernelSize - inputWidth) / 2;
      if(validPadding)
        ASSERT(outputWidth == (inputWidth - p.kernelSize + p.stride) / p.stride);
      else
        ASSERT(outputWidth == (inputWidth + p.stride - 1) / p.stride);

      // Load input/output base addresses
      a.mov(a.zsi(), imm(input.data()));
      if(input.data() == output.data())
        a.mov(a.zdi(), a.zsi());
      else
        a.mov(a.zdi(), imm(output.data()));

      bool helperRegInitialized = false;

      // Pool left-padded cells
      unsigned int inputCol = 0;
      unsigned int outputCol = 0;
      for(; inputCol < paddingLeft; inputCol += p.stride, outputCol++)
      {
        pool(a, paddingLeft - inputCol, channels, helperRegInitialized);
      }
      if(inputCol > paddingLeft)
      {
        a.add(a.zsi(), imm((inputCol - paddingLeft) * channels * sizeof(float)));
      }

      // Calculate number of non-padded cols
      unsigned int nonPaddedCols = 0;
      for(; inputCol < paddingLeft + inputWidth - p.kernelSize + 1; inputCol += p.stride, outputCol++, nonPaddedCols++);

      if(nonPaddedCols)
      {
        // Begin loop over image cols
        Label inputColLoop;
        if(nonPaddedCols > 1)
        {
          a.mov(a.zcx(), imm(nonPaddedCols));
          inputColLoop = a.newLabel();
          a.bind(inputColLoop);
        }

        // Pool current cell
        pool(a, 0, channels, helperRegInitialized);

        // Set input offset to next column, respecting the stride
        a.add(a.zsi(), imm(p.stride * channels * sizeof(float)));

        // End loop over image cols
        if(nonPaddedCols > 1)
        {
          a.dec(a.zcx());
          a.jnz(inputColLoop);
        }
      }

      // Pool right-padded cells
      for(; outputCol < outputWidth; inputCol += p.stride, outputCol++)
      {
        pool(a, inputCol + p.kernelSize - (paddingLeft + inputWidth), channels, helperRegInitialized);

        if(outputCol < outputWidth - 1)
          a.add(a.zsi(), imm(p.stride * channels * sizeof(float)));
      }
    }
  }
}
