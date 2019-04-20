/**
 * @author Arne Hasselbring
 */

#include "Concatenate.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    void ConcatenateCompiler::compileCopyPaste(X86Assembler& a, const std::vector<TensorPointerXf>& input, const TensorPointerXf& output, std::size_t innerSize) const
    {
      const bool isInplace = input[0].data() == output.data();
      std::size_t offset = isInplace ? (innerSize * input[0].dims(p.dimension)) : 0;
      a.mov(a.zdi(), imm_ptr<>(output.data() + offset));
      for(std::size_t i = isInplace ? 1 : 0; i < input.size(); ++i)
      {
        std::size_t remainingChannels = innerSize * input[i].dims(p.dimension);
        const bool aligned = (offset % 4) == 0;
        const std::size_t alignmentOffset = 4 - (offset % 4);
        offset += remainingChannels;

        a.mov(a.zsi(), imm_ptr<>(input[i].data()));
        if(!aligned)
        {
          a.movaps(x86::xmm0, a.ptr_zsi());
          a.movups(a.ptr_zdi(), x86::xmm0);
          if(remainingChannels <= 4)
          {
            if(i != input.size() - 1)
              a.add(a.zdi(), imm_u(remainingChannels * sizeof(float)));
            continue;
          }
          remainingChannels -= alignmentOffset;
          a.add(a.zsi(), imm_u(alignmentOffset * sizeof(float)));
          a.add(a.zdi(), imm_u(alignmentOffset * sizeof(float)));
        }

        for(unsigned int stepSize = settings.xmmRegs(); stepSize; --stepSize)
        {
          const unsigned int channelsPerStep = stepSize * 4;
          if(remainingChannels < channelsPerStep)
            continue;

          Label loop;
          if(remainingChannels >= channelsPerStep * 2)
          {
            loop = a.newLabel();
            a.mov(a.zcx(), imm_u(remainingChannels / channelsPerStep));
            a.bind(loop);
          }

          for(unsigned int step = 0; step < stepSize; step++)
            if(!aligned)
              a.movups(x86::xmm(step), a.ptr_zsi(step * 4 * sizeof(float)));
            else
              a.movaps(x86::xmm(step), a.ptr_zsi(step * 4 * sizeof(float)));
          for(unsigned int step = 0; step < stepSize; step++)
            a.movaps(a.ptr_zdi(step * 4 * sizeof(float)), x86::xmm(step));

          if(remainingChannels != channelsPerStep)
            a.add(a.zsi(), imm_u(stepSize * 4 * sizeof(float)));
          if(remainingChannels != channelsPerStep || i != input.size() - 1)
            a.add(a.zdi(), imm_u(stepSize * 4 * sizeof(float)));

          if(remainingChannels >= channelsPerStep * 2)
          {
            a.dec(a.zcx());
            a.jne(loop);
          }

          remainingChannels %= channelsPerStep;
        }
        if(remainingChannels == 1)
        {
          a.movss(x86::xmm0, a.ptr_zsi());
          a.movss(a.ptr_zdi(), x86::xmm0);
        }
        else
        {
          ASSERT(remainingChannels < 4);
          if(!aligned)
            a.movups(x86::xmm0, a.ptr_zsi());
          else
            a.movaps(x86::xmm0, a.ptr_zsi());
          a.movaps(a.ptr_zdi(), x86::xmm0);
        }
        if(i != input.size() - 1 && remainingChannels)
          a.add(a.zdi(), imm_u(remainingChannels * sizeof(float)));
      }
    }

    void ConcatenateCompiler::compile(X86Assembler& a, ActivationFunctionHandler& afHandler,
                                      const std::vector<TensorPointerXf>& input, const std::vector<TensorPointerXf>& output) const
    {
      ASSERT(output.size() == 1);

      std::size_t outerSize = 1, innerSize = 1;
      for(std::size_t i = 0; i < p.dimension; ++i)
        outerSize *= output[0].dims(i);
      for(std::size_t i = p.dimension + 1; i < output[0].rank(); ++i)
        innerSize *= output[0].dims(i);

      if(outerSize == 1)
        compileCopyPaste(a, input, output[0], innerSize);
      else
      {
        const std::array<X86Gp::Id, 3> regs = {X86Gp::kIdSi, X86Gp::kIdBx, X86Gp::kIdDx};
        std::size_t outputOffset = 0;
        for(std::size_t i = 0; i < input.size(); i += regs.size())
        {
          const std::size_t maxInput = std::min(input.size(), i + regs.size());
          a.mov(a.zdi(), imm_ptr<>(output[0].data() + outputOffset));
          for(std::size_t j = i; j < maxInput; ++j)
          {
            a.mov(a.gpzRef(regs[j - i]), imm_ptr<>(input[j].data()));
            outputOffset += innerSize * input[j].dims(p.dimension);
            ASSERT((outputOffset % 4) == 0); // TODO -> see below
          }
          a.mov(a.zax(), imm_u(outerSize));
          Label outerLoop = a.newLabel();
          a.bind(outerLoop);

          std::size_t channelsToSkip = 0;
          if(input.size() > regs.size())
          {
            for(std::size_t j = 0; j < i; ++j)
              channelsToSkip += innerSize * input[j].dims(p.dimension);
            for(std::size_t j = maxInput; j < input.size(); ++j)
              channelsToSkip += innerSize * input[j].dims(p.dimension);
          }

          for(std::size_t j = i; j < maxInput; ++j)
          {
            const X86Gp::Id ptrRegister = regs[j - i];
            std::size_t remainingChannels = innerSize * input[j].dims(p.dimension);
            ASSERT((remainingChannels % 4) == 0); // TODO -> currently only works for multiples of 4

            for(unsigned int stepSize = settings.xmmRegs(); stepSize; --stepSize)
            {
              const unsigned int channelsPerStep = stepSize * 4;
              if(remainingChannels < channelsPerStep)
                continue;

              Label innerLoop;
              if(remainingChannels >= channelsPerStep * 2)
              {
                innerLoop = a.newLabel();
                a.mov(a.zcx(), imm_u(remainingChannels / channelsPerStep));
                a.bind(innerLoop);
              }

              for(unsigned int step = 0; step < stepSize; step++)
                a.movaps(x86::xmm(step), a.ptr_base(ptrRegister, step * 4 * sizeof(float)));
              for(unsigned int step = 0; step < stepSize; step++)
                a.movaps(a.ptr_zdi(step * 4 * sizeof(float)), x86::xmm(step));

              a.add(a.gpzRef(ptrRegister), imm_u(stepSize * 4 * sizeof(float)));
              if(remainingChannels == channelsPerStep && j == maxInput - 1 && channelsToSkip)
              {
                a.add(a.zdi(), imm_u((stepSize * 4 + channelsToSkip) * sizeof(float)));
                channelsToSkip = 0;
              }
              else
                a.add(a.zdi(), imm_u(stepSize * 4 * sizeof(float)));

              if(remainingChannels >= channelsPerStep * 2)
              {
                a.dec(a.zcx());
                a.jne(innerLoop);
              }

              remainingChannels %= channelsPerStep;
            }
            ASSERT(!remainingChannels);
          }

          if(channelsToSkip)
            a.add(a.zdi(), imm_u(channelsToSkip * sizeof(float)));

          a.dec(a.zax());
          a.jnz(outerLoop);
        }
      }
    }
  }
}
