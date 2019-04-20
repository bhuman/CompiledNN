/**
 * Utility functions for approximating exp(x) by exploiting the floating point
 * format as shown by to Schraudolph.
 * (https://nic.schraudolph.org/bib2html/b2hd-Schraudolph99.html)
 *
 * For the neural network use cases, this method has a mean absolute error of
 * about 0.02.
 *
 * @author Felix Thielke
 */

#include "ExpApprox.h"

namespace NeuralNetwork
{
  namespace CompiledNNImpl
  {
    namespace ExpApprox
    {
      template<bool single, typename FactorType, typename OffsetType>
      void apply(X86Assembler& a, const std::vector<X86Xmm>& values, const FactorType factor, const OffsetType offset)
      {
        for(const X86Xmm& value : values)
        {
          if(single)
            a.mulss(value, factor);
          else
            a.mulps(value, factor);
        }
        for(const X86Xmm& value : values)
          a.cvtps2dq(value, value);        // evil floating point bit level hacking
        for(const X86Xmm& value : values)
          a.paddd(value, offset);          // what the fuck?
      }

      template void apply<false>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Xmm factor, const X86Xmm offset);
      template void apply<true>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Xmm factor, const X86Xmm offset);
      template void apply<false>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Mem factor, const X86Xmm offset);
      template void apply<true>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Mem factor, const X86Xmm offset);
      template void apply<false>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Xmm factor, const X86Mem offset);
      template void apply<true>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Xmm factor, const X86Mem offset);
      template void apply<false>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Mem factor, const X86Mem offset);
      template void apply<true>(X86Assembler& a, const std::vector<X86Xmm>& values, const X86Mem factor, const X86Mem offset);
    }
  }
}
