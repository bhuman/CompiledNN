/**
 * @file Im2Col2D.h
 *
 * Defines a compiler for the im2col operation (aka tf.extract_image_patches).
 *
 * @author Felix Thielke
 */

#pragma once

#include "../CompiledNNImplBase.h"
#include "../../Model.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    struct Im2Col2DCompiler : public SISOOperationCompiler
    {
      struct Parameters final
      {
        std::array<unsigned int, 2> kernelSize;
        std::array<unsigned int, 2> strides;
        std::array<unsigned int, 2> dilation;
        PaddingType paddingType;
      };
      const Parameters p;

      Im2Col2DCompiler(const CompilationSettings& settings, const Parameters& p) : SISOOperationCompiler(settings), p(p) {}

      inline bool canBeInplace() const override
      {
        return false;
      }

      void initialize() override {}
      void compile(x86::Assembler& a, ActivationFunctionHandler& afHandler, const TensorPointerXf& input, const TensorPointerXf& output) const override;

      inline std::vector<unsigned int> calcOutputDimensions(const std::vector<unsigned int>& inputDimensions) const override
      {
        ASSERT(inputDimensions.size() == 3);
        ASSERT(p.paddingType == PaddingType::valid || p.paddingType == PaddingType::same);

        const unsigned int rowDiff = p.paddingType == PaddingType::valid ? p.kernelSize[0] + (p.kernelSize[0] - 1) * (p.dilation[0] - 1) : 1;
        const unsigned int colDiff = p.paddingType == PaddingType::valid ? p.kernelSize[1] + (p.kernelSize[1] - 1) * (p.dilation[1] - 1) : 1;

        return
        {
          {
            (inputDimensions[0] - rowDiff + p.strides[0]) / p.strides[0],
            (inputDimensions[1] - colDiff + p.strides[1]) / p.strides[1],
            p.kernelSize[0] * p.kernelSize[1] * inputDimensions[2]
          }
        };
      }

    private:
      enum Side
      {
        TOP,
        BOTTOM,
        LEFT,
        RIGHT,
      };

      mutable bool hasPadding;

      void compileIm2ColRow(x86::Assembler& a, const std::array<unsigned int, 2> rowPadding, const std::array<unsigned int, 4>& padding, const TensorPointerXf& input) const;
      void compileIm2ColKernel(x86::Assembler& a, const std::array<unsigned int, 4> kernelPadding, const TensorPointerXf& input, const long long inputOffset) const;
    };
  }
}
