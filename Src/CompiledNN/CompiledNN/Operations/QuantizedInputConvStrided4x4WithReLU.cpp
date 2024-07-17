/**
 * @author Felix Thielke
 */

#include "QuantizedInputConvStrided4x4WithReLU.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    void QuantizedInputConvStrided4x4WithReLUCompiler::initialize()
    {
      // Declare constants
      constants.resize(2);

      // Store weights
      NetworkConstants& weights = constants[0];
      weights.data.resize(p.weights->size() / sizeof(float));
      unsigned int i = 0;
      for (unsigned int y = 0; y < p.weights->dims(0); y++)
      {
        for (unsigned int c = 0; c < p.weights->dims(3); c++)
        {
          for (unsigned int x = 0; x < p.weights->dims(1); x++)
          {
            reinterpret_cast<int8_t*>(weights.data.data())[i++] = static_cast<int8_t>((*p.weights)(y, x, 0, c) * static_cast<float>(1 << p.scale));
          }
        }
      }

      // Store biases
      NetworkConstants& biases = constants[1];
      biases.data.resize(p.biases->size() * sizeof(int16_t) / sizeof(float));
      i = 0;
      for (const float bias : *p.biases) {
        reinterpret_cast<int16_t*>(biases.data.data())[i++] = static_cast<int16_t>(bias);
      }
    }

    void QuantizedInputConvStrided4x4WithReLUCompiler::convolutionForPixel(x86::Assembler& a, unsigned int pixelId) const
    {
      a.movdqa(x86::xmm0, x86::xmm8);
      a.movdqa(x86::xmm2, x86::xmm9);
      a.movdqa(x86::xmm4, x86::xmm10);
      a.movdqa(x86::xmm6, x86::xmm11);

      const auto shuffleConfig = imm(pixelId | (pixelId << 2) | (pixelId << 4) | (pixelId << 6));
      a.shufps(x86::xmm0, x86::xmm0, shuffleConfig);
      a.shufps(x86::xmm2, x86::xmm2, shuffleConfig);
      a.shufps(x86::xmm4, x86::xmm4, shuffleConfig);
      a.shufps(x86::xmm6, x86::xmm6, shuffleConfig);

      a.movdqa(x86::xmm1, x86::xmm0);
      a.movdqa(x86::xmm3, x86::xmm2);
      a.movdqa(x86::xmm5, x86::xmm4);
      a.movdqa(x86::xmm7, x86::xmm6);

      a.pmaddubsw(x86::xmm0, a.ptr_zbx());
      a.pmaddubsw(x86::xmm1, a.ptr_zbx(0x10));
      a.pmaddubsw(x86::xmm2, a.ptr_zbx(0x20));
      a.pmaddubsw(x86::xmm3, a.ptr_zbx(0x30));
      a.pmaddubsw(x86::xmm4, a.ptr_zbx(0x40));
      a.pmaddubsw(x86::xmm5, a.ptr_zbx(0x50));
      a.pmaddubsw(x86::xmm6, a.ptr_zbx(0x60));
      a.pmaddubsw(x86::xmm7, a.ptr_zbx(0x70));

      a.paddsw(x86::xmm0, x86::xmm2);
      a.paddsw(x86::xmm1, x86::xmm3);
      a.paddsw(x86::xmm4, x86::xmm6);
      a.paddsw(x86::xmm5, x86::xmm7);
      a.paddsw(x86::xmm0, x86::xmm4);
      a.paddsw(x86::xmm1, x86::xmm5);
      a.phaddsw(x86::xmm0, x86::xmm1);

      a.psraw(x86::xmm0, imm(p.scale));
      a.paddsw(x86::xmm0, x86::xmm12);
    }

    void QuantizedInputConvStrided4x4WithReLUCompiler::outputAsFloat(x86::Assembler& a, unsigned int destOffset) const
    {
      a.movdqa(x86::xmm2, x86::xmm13);
      a.punpcklbw(x86::xmm13, x86::xmm14);
      a.punpckhbw(x86::xmm2, x86::xmm14);
      a.movdqa(x86::xmm1, x86::xmm13);
      a.movdqa(x86::xmm3, x86::xmm2);
      a.punpcklwd(x86::xmm13, x86::xmm14);
      a.punpckhwd(x86::xmm1, x86::xmm14);
      a.punpcklwd(x86::xmm2, x86::xmm14);
      a.punpckhwd(x86::xmm3, x86::xmm14);
      for (unsigned int i = 0; i < 4; i++)
        a.cvtdq2ps(x86::xmm(i == 0 ? 13 : i), x86::xmm(i == 0 ? 13 : i));
      for (unsigned int i = 0; i < 4; i++)
        a.movaps(a.ptr_zdi(i * 4 * sizeof(float) + destOffset), x86::xmm(i == 0 ? 13 : i));
    }

    void QuantizedInputConvStrided4x4WithReLUCompiler::compile(x86::Assembler& a, ActivationFunctionHandler&, const TensorPointerXf& input, const TensorPointerXf& output) const
    {
      ASSERT(input.dims(1) % 16 == 0);
      ASSERT(settings.xmmRegs() > 14);

      if(p.outputAsFloat)
        a.pxor(x86::xmm14, x86::xmm14);

      // Load input/output base addresses
      a.mov(a.zsi(), imm(input.data()));
      a.mov(a.zdi(), imm(output.data()));

      // Load weights address
      a.lea(a.zbx(), x86::ptr(constants[0].label));

      // Load biases into XMM12
      a.movdqa(x86::xmm12, x86::ptr(constants[1].label));

      // Begin loop over rows
      Label rowLoop;
      if (input.dims(0) > 4)
      {
        a.mov(a.zax(), imm(input.dims(0) / 4));
        rowLoop = a.newLabel();
        a.bind(rowLoop);
      }

      // Begin loop over columns
      Label colLoop;
      if (input.dims(1) > 16)
      {
        a.mov(a.zcx(), imm(input.dims(1) / 16));
        colLoop = a.newLabel();
        a.bind(colLoop);
      }

      // Load 16 pixels (4 output pixels) from 4 consecutive rows
      a.movdqa(x86::xmm8, a.ptr_zsi());
      a.movdqa(x86::xmm9, a.ptr_zsi(input.dims(1)));
      a.movdqa(x86::xmm10, a.ptr_zsi(2 * input.dims(1)));
      a.movdqa(x86::xmm11, a.ptr_zsi(3 * input.dims(1)));

      // Calculate pixels 0 and 1
      convolutionForPixel(a, 0);
      a.movdqa(x86::xmm13, x86::xmm0);
      convolutionForPixel(a, 1);
      a.packuswb(x86::xmm13, x86::xmm0);
      if(p.outputAsFloat)
        outputAsFloat(a, 0);
      else
        a.movdqa(a.ptr_zdi(), x86::xmm13);

      // Calculate pixels 2 and 3
      convolutionForPixel(a, 2);
      a.movdqa(x86::xmm13, x86::xmm0);
      convolutionForPixel(a, 3);
      a.packuswb(x86::xmm13, x86::xmm0);
      if (p.outputAsFloat)
        outputAsFloat(a, 0x40);
      else
        a.movdqa(a.ptr_zdi(0x10), x86::xmm13);

      // Advance output pointer
      a.add(a.zdi(), imm(p.outputAsFloat ? 0x80 : 0x20));

      // Next column
      if (input.dims(1) > 16)
      {
        a.add(a.zsi(), imm(0x10));
        a.dec(a.zcx());
        a.jnz(colLoop);
      }

      // Next row
      if (input.dims(0) > 4)
      {
        // Move to next row (cursor is currently at first pixel of first skipped row)
        a.add(a.zsi(), imm(input.dims(1) * 3));

        a.dec(a.zax());
        a.jnz(rowLoop);
      }
    }
  }
}
