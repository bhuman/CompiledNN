/**
 * @author Felix Thielke
 */

#include "CompilationSettings.h"
#include <asmjit/asmjit.h>

using namespace asmjit;

void NeuralNetwork::CompilationSettings::constrict()
{
  const CpuInfo& cpuInfo = CpuInfo::host();

  if(useX64 && !Environment(cpuInfo.arch(), cpuInfo.subArch()).is64Bit())
    useX64 = false;

  if(useSSE42 && !cpuInfo.features().x86().hasSSE4_2())
    useSSE42 = false;

  if(useAVX2 && !cpuInfo.features().x86().hasAVX2())
    useAVX2 = false;

  if(useFMA3 && !cpuInfo.features().x86().hasFMA())
    useFMA3 = false;
}
