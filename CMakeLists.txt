cmake_minimum_required(VERSION 3.10)
project(CompiledNN)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(CompiledNN STATIC
    Src/Model.cpp
    Src/CompiledNN.cpp
    Src/SimpleNN.cpp
    Src/CompiledNN/ActivationFunctions.cpp
    Src/CompiledNN/CompilationSettings.cpp
    Src/CompiledNN/Operations/Activation.cpp
    Src/CompiledNN/Operations/Arithmetic.cpp
    Src/CompiledNN/Operations/BatchNormalization.cpp
    Src/CompiledNN/Operations/Concatenate.cpp
    Src/CompiledNN/Operations/Conv2D.cpp
    Src/CompiledNN/Operations/DConv2D.cpp
    Src/CompiledNN/Operations/Dense.cpp
    Src/CompiledNN/Operations/GlobalPooling2D.cpp
    Src/CompiledNN/Operations/Padding.cpp
    Src/CompiledNN/Operations/Pooling2D.cpp
    Src/CompiledNN/Operations/Softmax.cpp
    Src/CompiledNN/Operations/UInt8Input.cpp
    Src/CompiledNN/Operations/UpSampling2D.cpp
    Src/CompiledNN/Util/ExpApprox.cpp
    Src/Platform/BHAssert.cpp
    Src/Tools/Streams/InOut.cpp
    Src/Tools/Streams/InStreams.cpp
    Src/Tools/Streams/SimpleMap.cpp

    3rdParty/asmjit/src/asmjit/base/arch.cpp
    3rdParty/asmjit/src/asmjit/base/assembler.cpp
    3rdParty/asmjit/src/asmjit/base/codebuilder.cpp
    3rdParty/asmjit/src/asmjit/base/codecompiler.cpp
    3rdParty/asmjit/src/asmjit/base/codeemitter.cpp
    3rdParty/asmjit/src/asmjit/base/codeholder.cpp
    3rdParty/asmjit/src/asmjit/base/constpool.cpp
    3rdParty/asmjit/src/asmjit/base/cpuinfo.cpp
    3rdParty/asmjit/src/asmjit/base/func.cpp
    3rdParty/asmjit/src/asmjit/base/globals.cpp
    3rdParty/asmjit/src/asmjit/base/inst.cpp
    3rdParty/asmjit/src/asmjit/base/logging.cpp
    3rdParty/asmjit/src/asmjit/base/operand.cpp
    3rdParty/asmjit/src/asmjit/base/osutils.cpp
    3rdParty/asmjit/src/asmjit/base/regalloc.cpp
    3rdParty/asmjit/src/asmjit/base/runtime.cpp
    3rdParty/asmjit/src/asmjit/base/string.cpp
    3rdParty/asmjit/src/asmjit/base/utils.cpp
    3rdParty/asmjit/src/asmjit/base/vmem.cpp
    3rdParty/asmjit/src/asmjit/base/zone.cpp
    3rdParty/asmjit/src/asmjit/x86/x86assembler.cpp
    3rdParty/asmjit/src/asmjit/x86/x86builder.cpp
    3rdParty/asmjit/src/asmjit/x86/x86compiler.cpp
    3rdParty/asmjit/src/asmjit/x86/x86inst.cpp
    3rdParty/asmjit/src/asmjit/x86/x86instimpl.cpp
    3rdParty/asmjit/src/asmjit/x86/x86internal.cpp
    3rdParty/asmjit/src/asmjit/x86/x86logging.cpp
    3rdParty/asmjit/src/asmjit/x86/x86operand_regs.cpp
    3rdParty/asmjit/src/asmjit/x86/x86regalloc.cpp
)

target_include_directories(CompiledNN PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Src")
target_include_directories(CompiledNN SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/asmjit/src")
target_compile_definitions(CompiledNN PRIVATE ASMJIT_EMBED ASMJIT_BUILD_X86 ASMJIT_DISABLE_BUILDER ASMJIT_DISABLE_COMPILER
    ASMJIT_DISABLE_LOGGING ASMJIT_DISABLE_TEXT ASMJIT_DISABLE_VALIDATION)

find_package(HDF5 REQUIRED)
target_link_libraries(CompiledNN PRIVATE ${HDF5_C_LIBRARIES})
target_include_directories(CompiledNN SYSTEM PRIVATE ${HDF5_C_INCLUDE_DIRS})
target_compile_definitions(CompiledNN PRIVATE ${HDF5_C_DEFINITIONS})
