/**
 * @author Felix Thielke
 */

#include "Im2Col2D.h"
#include "Platform/BHAssert.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    void Im2Col2DCompiler::compile(x86::Assembler& a, ActivationFunctionHandler&, const TensorPointerXf& input, const TensorPointerXf& output) const
    {
      ASSERT(input.rank() == 3);
      ASSERT(output.rank() == 3);
      ASSERT(p.paddingType == PaddingType::valid || p.paddingType == PaddingType::same);
      ASSERT(p.kernelSize[0] >= 1 && p.kernelSize[1] >= 1);
      ASSERT(p.kernelSize[0] > 1 || p.kernelSize[1] > 1); // Im2Col for 1x1-kernels would be a no-op
      ASSERT(p.strides[0] >= 1 && p.strides[1] >= 1);
      ASSERT(p.dilation[0] >= 1 && p.dilation[1] >= 1);

      // Calculate padding
      const unsigned int verticalPadding = p.paddingType == PaddingType::same ? (output.dims(0) - 1) * p.strides[0] + p.kernelSize[0] + (p.kernelSize[0] - 1) * (p.dilation[0] - 1) - input.dims(0) : 0;
      const unsigned int horizontalPadding = p.paddingType == PaddingType::same ? (output.dims(1) - 1) * p.strides[1] + p.kernelSize[1] + (p.kernelSize[1] - 1) * (p.dilation[1] - 1) - input.dims(1) : 0;
      const std::array<unsigned int, 4> padding
      {
        {
          verticalPadding / 2,
          verticalPadding - verticalPadding / 2,
          horizontalPadding / 2,
          horizontalPadding - horizontalPadding / 2
        }
      };
      hasPadding = padding[Side::TOP] > 0 || padding[Side::BOTTOM] > 0 || padding[Side::LEFT] > 0 || padding[Side::RIGHT] > 0;

      // Set xmm0 to zero for padding
      if(hasPadding)
        a.xorps(x86::xmm0, x86::xmm0);

      // Load input/output base addresses
      a.mov(a.zsi(), imm(input.data()));
      a.mov(a.zdi(), imm(output.data()));

      // Top padding
      int inputY = -static_cast<int>(padding[Side::TOP]);
      for(; inputY < 0; inputY += p.strides[0])
        compileIm2ColRow(a, {{static_cast<unsigned int>(-inputY), 0}}, padding, input);
      if(inputY > 0)
        a.add(a.zsi(), imm(static_cast<unsigned long long>(inputY) * input.dims(1) * input.dims(2) * sizeof(float)));

      // No top or bottom padding
      unsigned int inputYEnd = static_cast<unsigned int>(inputY) + (p.kernelSize[0] - 1) * p.dilation[0];
      unsigned int nRows;
      for(nRows = 0; inputYEnd < input.dims(0); nRows++, inputYEnd += p.strides[0]);
      if(nRows <= 2)
      {
        // Unroll loop over rows
        for(unsigned int i = nRows; i; i--)
          compileIm2ColRow(a, {{0, 0}}, padding, input);
      }
      else
      {
        // Loop over rows
        a.mov(a.zax(), imm(nRows));
        Label rowLoop = a.newLabel();
        a.bind(rowLoop);
        compileIm2ColRow(a, {{0, 0}}, padding, input);
        a.dec(a.zax());
        a.jnz(rowLoop);
      }

      // Bottom padding
      for(; inputYEnd < input.dims(0) + padding[Side::BOTTOM]; inputYEnd += p.strides[0])
        compileIm2ColRow(a, {{0, inputYEnd + 1 - input.dims(0)}}, padding, input);
    }

    void Im2Col2DCompiler::compileIm2ColRow(x86::Assembler& a, const std::array<unsigned int, 2> rowPadding, const std::array<unsigned int, 4>& padding, const TensorPointerXf& input) const
    {
      int inputX = -static_cast<int>(padding[Side::LEFT]);

      // Left padding
      for(; inputX < 0; inputX += p.strides[1])
        compileIm2ColKernel(a, { {rowPadding[Side::TOP], rowPadding[Side::BOTTOM], static_cast<unsigned int>(-inputX), 0}}, input, 0);
      if(inputX > 0)
        a.add(a.zsi(), imm(static_cast<unsigned long long>(inputX) * input.dims(2) * sizeof(float)));

      // No horizontal padding
      long long inputOffset = 0ll;
      unsigned int inputXEnd = inputX + (p.kernelSize[1] - 1) * p.dilation[1];
      unsigned int nCols;
      for(nCols = 0; inputXEnd < input.dims(1); nCols++, inputXEnd += p.strides[1]);
      if(nCols <= 2)
      {
        // Unroll loop over cols
        for(unsigned int i = nCols; i; i--)
        {
          compileIm2ColKernel(a, {{rowPadding[Side::TOP], rowPadding[Side::BOTTOM], 0, 0}}, input, inputOffset);

          // Advance input pointer to next column
          inputOffset += p.strides[1] * input.dims(2);
        }
      }
      else
      {
        // Loop over cols
        a.mov(a.zdx(), imm(nCols));
        Label colLoop = a.newLabel();
        a.bind(colLoop);
        compileIm2ColKernel(a, {{rowPadding[Side::TOP], rowPadding[Side::BOTTOM], 0, 0}}, input, inputOffset);
        a.add(a.zsi(), imm(p.strides[1] * input.dims(2) * sizeof(float)));
        a.dec(a.zdx());
        a.jnz(colLoop);
      }

      // Right padding
      for(; inputXEnd < input.dims(1) + padding[Side::RIGHT]; inputXEnd += p.strides[1])
      {
        compileIm2ColKernel(a, {{rowPadding[Side::TOP], rowPadding[Side::BOTTOM], 0, inputXEnd + 1 - input.dims(1)}}, input, inputOffset);

        // Advance input pointer to next column
        inputOffset += p.strides[1] * input.dims(2);
      }

      if(rowPadding[Side::TOP] > 0)
      {
        // Reset input pointer
        inputOffset -= (inputXEnd - (p.kernelSize[1] - 1) * p.dilation[1]) * input.dims(2);
      }
      else
      {
        // Advance input pointer to next row
        inputOffset += (p.strides[0] * input.dims(1) - (inputXEnd - (p.kernelSize[1] - 1) * p.dilation[1])) * input.dims(2);
      }
      if(inputOffset != 0ll)
        a.add(a.zsi(), imm(inputOffset * sizeof(float)));
    }

    void Im2Col2DCompiler::compileIm2ColKernel(x86::Assembler& a, const std::array<unsigned int, 4> kernelPadding, const TensorPointerXf& input, const long long inputOffset) const
    {
      // Calculate the number of cells in each area of the kernel
      const unsigned int topPaddingCells = (kernelPadding[Side::TOP] + p.dilation[0] - 1) / p.dilation[0];
      const unsigned int verticalKernelCells = ((p.kernelSize[0] - topPaddingCells) * p.dilation[0] - kernelPadding[Side::BOTTOM]) / p.dilation[0];
      const unsigned int bottomPaddingCells = p.kernelSize[0] - topPaddingCells - verticalKernelCells;
      const unsigned int leftPaddingCells = (kernelPadding[Side::LEFT] + p.dilation[1] - 1) / p.dilation[1];
      const unsigned int horizontalKernelCells = ((p.kernelSize[1] - leftPaddingCells) * p.dilation[1] - kernelPadding[Side::RIGHT]) / p.dilation[1];
      const unsigned int rightPaddingCells = p.kernelSize[1] - leftPaddingCells - horizontalKernelCells;

      const bool cellsAligned = (input.dims(2) % 4) == 0;

      // Define functions for padding
      auto pad = [&a, cellsAligned](const unsigned int n, const unsigned long long offset = 0ull)
      {
        if(n == 0)
          return offset;

        if(n <= 16)
        {
          // Unrolled loop
          for(unsigned int i = 0; i < n; i += 4)
          {
            if(i == n - 1)
              a.movss(a.ptr_zdi(static_cast<int>(i * sizeof(float) + offset)), x86::xmm0);
            else
            {
              if(cellsAligned)
                a.movaps(a.ptr_zdi(static_cast<int>(i * sizeof(float) + offset)), x86::xmm0);
              else
                a.movups(a.ptr_zdi(static_cast<int>(i * sizeof(float) + offset)), x86::xmm0);
            }
          }

          return offset + n * sizeof(float);
        }
        else
        {
          // Loop
          a.mov(a.zcx(), (n + 3) / 4);
          a.add(a.zdi(), imm(offset));
          Label padLoop = a.newLabel();
          a.bind(padLoop);
          if(cellsAligned)
            a.movaps(a.ptr_zdi(), x86::xmm0);
          else
            a.movups(a.ptr_zdi(), x86::xmm0);
          a.add(a.zdi(), imm(4 * sizeof(float)));
          a.dec(a.zcx());
          a.jnz(padLoop);

          if(n % 4 != 0)
            a.sub(a.zdi(), imm((4 - (n % 4)) * sizeof(float)));

          return 0ull;
        }
      };
      auto padAndAdvance = [&a, &pad](const unsigned int n, const unsigned long long offset = 0ull)
      {
        const unsigned long long newOffset = pad(n, offset);
        if(newOffset != 0)
          a.add(a.zdi(), imm(newOffset));
      };

      // Pad output for top part of kernel
      padAndAdvance(topPaddingCells * p.kernelSize[1] * input.dims(2));

      // Load input pointer address for the kernel
      a.lea(a.zbx(), a.ptr_zsi(static_cast<int>((((topPaddingCells * p.dilation[0] - kernelPadding[Side::TOP]) * input.dims(1) + leftPaddingCells * p.dilation[1] - kernelPadding[Side::LEFT]) * input.dims(2) + inputOffset) * sizeof(float))));

      unsigned long long outputOffset = 0ull;
      const unsigned int regOffset = hasPadding ? 1 : 0;
      const unsigned int availableRegisters = settings.xmmRegs() - regOffset;
      const unsigned int regsPerCol = (input.dims(2) + 3) / 4;
      if(p.dilation[1] == 1 && availableRegisters >= verticalKernelCells * ((input.dims(2) * horizontalKernelCells + 3) / 4))
      {
        // No horizontal dilation and the whole kernel fits in the registers

        // Gather values
        unsigned int curReg = regOffset;
        unsigned long long inputOffset = 0ll;
        for(unsigned int kernelYCount = verticalKernelCells; kernelYCount; kernelYCount--)
        {
          // Read channels for all cells in kernel
          for(unsigned int kernelCell = 0; kernelCell < input.dims(2) * horizontalKernelCells + 3; curReg++, kernelCell += 4)
          {
            if(cellsAligned)
              a.movaps(x86::xmm(curReg), a.ptr_zbx(static_cast<int>((inputOffset + kernelCell) * sizeof(float))));
            else
              a.movups(x86::xmm(curReg), a.ptr_zbx(static_cast<int>((inputOffset + kernelCell) * sizeof(float))));
          }

          // Set offset to next row
          inputOffset += p.dilation[0] * input.dims(1) * input.dims(2);
        }

        // Write values
        curReg = 0;
        for(size_t kernelYCount = verticalKernelCells; kernelYCount; kernelYCount--)
        {
          // Pad output for left part of kernel
          outputOffset = pad(leftPaddingCells * input.dims(2), outputOffset);

          // Write channels for all cells in kernel
          for(size_t kernelCellCount = (input.dims(2) * horizontalKernelCells + 3) / 4; kernelCellCount; curReg++, outputOffset += 4 * sizeof(float), kernelCellCount--)
          {
            if(cellsAligned)
              a.movaps(a.ptr_zdi(static_cast<int>(outputOffset)), x86::xmm(curReg));
            else
              a.movups(a.ptr_zdi(static_cast<int>(outputOffset)), x86::xmm(curReg));
          }
          if((input.dims(2) * horizontalKernelCells) % 4 != 0)
            outputOffset -= 4 - ((input.dims(2) * horizontalKernelCells) % 4);

          // Pad output for right part of kernel
          outputOffset = pad(rightPaddingCells * input.dims(2), outputOffset);
        }
      }
      else if(availableRegisters >= verticalKernelCells * regsPerCol * horizontalKernelCells)
      {
        // Horizontal dilation and the whole kernel fits in the registers

        // Gather values
        unsigned int curReg = regOffset;
        unsigned long long inputOffset = 0ll;
        for(unsigned int kernelYCount = verticalKernelCells; kernelYCount; kernelYCount--)
        {
          for(size_t kernelXCount = horizontalKernelCells; kernelXCount; kernelXCount--)
          {
            for(size_t kernelCCount = (input.dims(2) + 3) / 4; kernelCCount; curReg++, kernelCCount--)
            {
              if(cellsAligned)
                a.movaps(x86::xmm(curReg), a.ptr_zbx(static_cast<int>(inputOffset * sizeof(float))));
              else
                a.movups(x86::xmm(curReg), a.ptr_zbx(static_cast<int>(inputOffset * sizeof(float))));

              // Set offset to next cell
              inputOffset += 4;
            }

            // Set offset to next column
            inputOffset += p.dilation[1] * input.dims(2) - ((input.dims(2) + 3) & (~3));
          }

          // Set offset to next row
          inputOffset += (p.dilation[0] * input.dims(1) - horizontalKernelCells * p.dilation[1]) * input.dims(2);
        }

        // Write values
        curReg = 0;
        for(size_t kernelYCount = verticalKernelCells; kernelYCount; kernelYCount--)
        {
          // Pad output for left part of kernel
          outputOffset = pad(leftPaddingCells * input.dims(2), outputOffset);

          for(size_t kernelXCount = horizontalKernelCells; kernelXCount; kernelXCount--)
          {
            for(size_t kernelCCount = (input.dims(2) + 3) / 4; kernelCCount; curReg++, outputOffset += 4 * sizeof(float), kernelCCount--)
            {
              if(cellsAligned)
                a.movaps(a.ptr_zdi(static_cast<int>(outputOffset)), x86::xmm(curReg));
              else
                a.movups(a.ptr_zdi(static_cast<int>(outputOffset)), x86::xmm(curReg));
            }
            if(input.dims(2) % 4 != 0)
              outputOffset -= 4 - (input.dims(2) % 4);
          }

          // Pad output for right part of kernel
          outputOffset = pad(rightPaddingCells * input.dims(2), outputOffset);
        }
      }
      else if(p.dilation[1] == 1 && availableRegisters >= (input.dims(2) * horizontalKernelCells + 3) / 4)
      {
        // No horizontal dilation and one row of the kernel fits in the registers
        FAIL("not implemented");
      }
      else if(availableRegisters >= regsPerCol * horizontalKernelCells)
      {
        // Horizontal dilation and one row of the kernel fits in the registers
        FAIL("not implemented");
      }
      else if(availableRegisters >= regsPerCol)
      {
        // One cell of the kernel fits in the registers
        FAIL("not implemented");
      }
      else
      {
        // Less than a cell fits in the registers
        FAIL("not implemented");
      }

      // Pad output for bottom part of kernel
      padAndAdvance(bottomPaddingCells * p.kernelSize[1] * input.dims(2), outputOffset);
    }

    /**
     * Reference implementation.
     */
    void im2col(const TensorXf& input, TensorXf& output,
                const std::array<unsigned int, 2> kernelSize,
                const std::array<unsigned int, 2> strides,
                const std::array<unsigned int, 2> dilation,
                const std::array<unsigned int, 4> padding)
    {
      enum Side
      {
        TOP,
        BOTTOM,
        LEFT,
        RIGHT,
      };

      const float* inputPtr = input.data();
      float* outputPtr = output.data();

      auto im2colRow = [&](const std::array<unsigned int, 2> rowPadding)
      {
        auto im2colFilter = [&](const std::array<unsigned int, 4> filterPadding)
        {
          // Calculate the number of cells in each area of the kernel
          const unsigned int topPaddingCells = (filterPadding[Side::TOP] + dilation[0] - 1) / dilation[0];
          const unsigned int verticalKernelCells = ((kernelSize[0] - topPaddingCells) * dilation[0] - filterPadding[Side::BOTTOM]) / dilation[0];
          const unsigned int bottomPaddingCells = kernelSize[0] - topPaddingCells - verticalKernelCells;
          const unsigned int leftPaddingCells = (filterPadding[Side::LEFT] + dilation[1] - 1) / dilation[1];
          const unsigned int horizontalKernelCells = ((kernelSize[1] - leftPaddingCells) * dilation[1] - filterPadding[Side::RIGHT]) / dilation[1];
          const unsigned int rightPaddingCells = kernelSize[1] - leftPaddingCells - horizontalKernelCells;

          // Pad output for top part of kernel
          for(unsigned int i = topPaddingCells * kernelSize[1] * input.dims(2); i; i--)
            *outputPtr++ = 0.f;

          const float* kernelInputPtr = inputPtr + ((topPaddingCells * dilation[0] - filterPadding[Side::TOP]) * input.dims(1) + leftPaddingCells * dilation[1] - filterPadding[Side::LEFT]) * input.dims(2);

          for(size_t kernelYCount = verticalKernelCells; kernelYCount; kernelYCount--)
          {
            // Pad output for left part of kernel
            for(unsigned int i = leftPaddingCells * input.dims(2); i; i--)
              *outputPtr++ = 0.f;

            // Copy channels for all cells in kernel
            for(size_t kernelXCount = horizontalKernelCells; kernelXCount; kernelXCount--)
            {
              for(size_t filterC = 0; filterC < input.dims(2); filterC++)
                outputPtr[filterC] = kernelInputPtr[filterC];
              outputPtr += input.dims(2);

              // Advance kernelInputPtr
              kernelInputPtr += dilation[1] * input.dims(2);
            }

            // Pad output for right part of kernel
            for(unsigned int i = rightPaddingCells * input.dims(2); i; i--)
              *outputPtr++ = 0.f;

            // Set kernelInputPtr to next row, skipping padding
            kernelInputPtr += (dilation[0] * input.dims(1) - horizontalKernelCells * dilation[1]) * input.dims(2);
          }

          // Pad output for bottom part of kernel
          for(unsigned int i = bottomPaddingCells * kernelSize[1] * input.dims(2); i; i--)
            *outputPtr++ = 0.f;
        };

        int inputX = -static_cast<int>(padding[Side::LEFT]);

        // Left padding
        for(; inputX < 0; inputX += strides[1])
        {
          im2colFilter({{rowPadding[Side::TOP], rowPadding[Side::BOTTOM], static_cast<unsigned int>(-inputX), 0}});
        }
        if(inputX > 0)
          inputPtr += inputX * input.dims(2);

        // No horizontal padding
        unsigned int inputXEnd = inputX + (kernelSize[1] - 1) * dilation[1];
        for(; inputXEnd < input.dims(1); inputXEnd += strides[1])
        {
          im2colFilter({{rowPadding[Side::TOP], rowPadding[Side::BOTTOM], 0, 0}});

          // Advance inputPtr to next column
          inputPtr += strides[1] * input.dims(2);
        }

        // Right padding
        for(; inputXEnd < input.dims(1) + padding[Side::RIGHT]; inputXEnd += strides[1])
        {
          im2colFilter({{rowPadding[Side::TOP], rowPadding[Side::BOTTOM], 0, inputXEnd + 1 - input.dims(1)}});

          // Advance inputPtr to next column
          inputPtr += strides[1] * input.dims(2);
        }

        if(rowPadding[Side::TOP] > 0)
        {
          // Reset inputPtr
          inputPtr -= (inputXEnd - (kernelSize[1] - 1) * dilation[1]) * input.dims(2);
        }
        else
        {
          // Advance inputPtr to next row
          inputPtr += (strides[0] * input.dims(1) - (inputXEnd - (kernelSize[1] - 1) * dilation[1])) * input.dims(2);
        }
      };

      int inputY = -static_cast<int>(padding[Side::TOP]);

      // Top padding
      for(; inputY < 0; inputY += strides[0])
      {
        im2colRow({{static_cast<unsigned int>(-inputY), 0}});
      }
      if(inputY > 0)
        inputPtr += inputY * input.dims(1) * input.dims(2);

      // No top or bottom padding
      unsigned int inputYEnd = static_cast<unsigned int>(inputY) + (kernelSize[0] - 1) * dilation[0];
      for(; inputYEnd < input.dims(0); inputYEnd += strides[0])
      {
        im2colRow({{0, 0}});
      }

      // Bottom padding
      for(; inputYEnd < input.dims(0) + padding[Side::BOTTOM]; inputYEnd += strides[0])
      {
        im2colRow({{0, inputYEnd + 1 - input.dims(0)}});
      }
    }
  }
}
