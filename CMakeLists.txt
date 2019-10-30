cmake_minimum_required(VERSION 3.10)
project(CompiledNN)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(CompiledNN STATIC
    Src/CompiledNN.cpp
    Src/CompiledNN.h
    Src/Model.cpp
    Src/Model.h
    Src/SimpleNN.cpp
    Src/SimpleNN.h
    Src/Tensor.h
    Src/CompiledNN/ActivationFunctions.cpp
    Src/CompiledNN/ActivationFunctions.h
    Src/CompiledNN/CompilationSettings.cpp
    Src/CompiledNN/CompilationSettings.h
    Src/CompiledNN/CompiledNNImpl.h
    Src/CompiledNN/CompiledNNImplBase.h
    Src/CompiledNN/TensorPointer.h
    Src/CompiledNN/Operations/Activation.cpp
    Src/CompiledNN/Operations/Activation.h
    Src/CompiledNN/Operations/Arithmetic.cpp
    Src/CompiledNN/Operations/Arithmetic.h
    Src/CompiledNN/Operations/BatchNormalization.cpp
    Src/CompiledNN/Operations/BatchNormalization.h
    Src/CompiledNN/Operations/Concatenate.cpp
    Src/CompiledNN/Operations/Concatenate.h
    Src/CompiledNN/Operations/Conv2D.cpp
    Src/CompiledNN/Operations/Conv2D.h
    Src/CompiledNN/Operations/Cropping2D.cpp
    Src/CompiledNN/Operations/Cropping2D.h
    Src/CompiledNN/Operations/DConv2D.cpp
    Src/CompiledNN/Operations/DConv2D.h
    Src/CompiledNN/Operations/Dense.cpp
    Src/CompiledNN/Operations/Dense.h
    Src/CompiledNN/Operations/GlobalPooling2D.cpp
    Src/CompiledNN/Operations/GlobalPooling2D.h
    Src/CompiledNN/Operations/Pooling2D.cpp
    Src/CompiledNN/Operations/Pooling2D.h
    Src/CompiledNN/Operations/Softmax.cpp
    Src/CompiledNN/Operations/Softmax.h
    Src/CompiledNN/Operations/UInt8Input.cpp
    Src/CompiledNN/Operations/UInt8Input.h
    Src/CompiledNN/Operations/UpSampling2D.cpp
    Src/CompiledNN/Operations/UpSampling2D.h
    Src/CompiledNN/Operations/ZeroPadding2D.cpp
    Src/CompiledNN/Operations/ZeroPadding2D.h
    Src/CompiledNN/Util/ExpApprox.cpp
    Src/CompiledNN/Util/ExpApprox.h

    3rdParty/asmjit/src/asmjit/asmjit.h
    3rdParty/asmjit/src/asmjit/core.h
    3rdParty/asmjit/src/asmjit/x86.h
    3rdParty/asmjit/src/asmjit/core/arch.cpp
    3rdParty/asmjit/src/asmjit/core/arch.h
    3rdParty/asmjit/src/asmjit/core/assembler.cpp
    3rdParty/asmjit/src/asmjit/core/assembler.h
    3rdParty/asmjit/src/asmjit/core/build.h
    3rdParty/asmjit/src/asmjit/core/builder.cpp
    3rdParty/asmjit/src/asmjit/core/builder.h
    3rdParty/asmjit/src/asmjit/core/callconv.cpp
    3rdParty/asmjit/src/asmjit/core/callconv.h
    3rdParty/asmjit/src/asmjit/core/codebufferwriter_p.h
    3rdParty/asmjit/src/asmjit/core/codeholder.cpp
    3rdParty/asmjit/src/asmjit/core/codeholder.h
    3rdParty/asmjit/src/asmjit/core/compiler.cpp
    3rdParty/asmjit/src/asmjit/core/compiler.h
    3rdParty/asmjit/src/asmjit/core/constpool.cpp
    3rdParty/asmjit/src/asmjit/core/constpool.h
    3rdParty/asmjit/src/asmjit/core/cpuinfo.cpp
    3rdParty/asmjit/src/asmjit/core/cpuinfo.h
    3rdParty/asmjit/src/asmjit/core/emitter.cpp
    3rdParty/asmjit/src/asmjit/core/emitter.h
    3rdParty/asmjit/src/asmjit/core/features.h
    3rdParty/asmjit/src/asmjit/core/func.cpp
    3rdParty/asmjit/src/asmjit/core/func.h
    3rdParty/asmjit/src/asmjit/core/globals.cpp
    3rdParty/asmjit/src/asmjit/core/globals.h
    3rdParty/asmjit/src/asmjit/core/inst.cpp
    3rdParty/asmjit/src/asmjit/core/inst.h
    3rdParty/asmjit/src/asmjit/core/jitallocator.cpp
    3rdParty/asmjit/src/asmjit/core/jitallocator.h
    3rdParty/asmjit/src/asmjit/core/jitruntime.cpp
    3rdParty/asmjit/src/asmjit/core/jitruntime.h
    3rdParty/asmjit/src/asmjit/core/logging.cpp
    3rdParty/asmjit/src/asmjit/core/logging.h
    3rdParty/asmjit/src/asmjit/core/misc_p.h
    3rdParty/asmjit/src/asmjit/core/operand.cpp
    3rdParty/asmjit/src/asmjit/core/operand.h
    3rdParty/asmjit/src/asmjit/core/osutils.cpp
    3rdParty/asmjit/src/asmjit/core/osutils.h
    3rdParty/asmjit/src/asmjit/core/raassignment_p.h
    3rdParty/asmjit/src/asmjit/core/rabuilders_p.h
    3rdParty/asmjit/src/asmjit/core/radefs_p.h
    3rdParty/asmjit/src/asmjit/core/ralocal.cpp
    3rdParty/asmjit/src/asmjit/core/ralocal_p.h
    3rdParty/asmjit/src/asmjit/core/rapass.cpp
    3rdParty/asmjit/src/asmjit/core/rapass_p.h
    3rdParty/asmjit/src/asmjit/core/rastack.cpp
    3rdParty/asmjit/src/asmjit/core/rastack_p.h
    3rdParty/asmjit/src/asmjit/core/string.cpp
    3rdParty/asmjit/src/asmjit/core/string.h
    3rdParty/asmjit/src/asmjit/core/support.cpp
    3rdParty/asmjit/src/asmjit/core/support.h
    3rdParty/asmjit/src/asmjit/core/target.cpp
    3rdParty/asmjit/src/asmjit/core/target.h
    3rdParty/asmjit/src/asmjit/core/type.cpp
    3rdParty/asmjit/src/asmjit/core/type.h
    3rdParty/asmjit/src/asmjit/core/virtmem.cpp
    3rdParty/asmjit/src/asmjit/core/virtmem.h
    3rdParty/asmjit/src/asmjit/core/zone.cpp
    3rdParty/asmjit/src/asmjit/core/zone.h
    3rdParty/asmjit/src/asmjit/core/zonehash.cpp
    3rdParty/asmjit/src/asmjit/core/zonehash.h
    3rdParty/asmjit/src/asmjit/core/zonelist.cpp
    3rdParty/asmjit/src/asmjit/core/zonelist.h
    3rdParty/asmjit/src/asmjit/core/zonestack.cpp
    3rdParty/asmjit/src/asmjit/core/zonestack.h
    3rdParty/asmjit/src/asmjit/core/zonetree.cpp
    3rdParty/asmjit/src/asmjit/core/zonetree.h
    3rdParty/asmjit/src/asmjit/core/zonevector.cpp
    3rdParty/asmjit/src/asmjit/core/zonevector.h
    3rdParty/asmjit/src/asmjit/x86/x86assembler.cpp
    3rdParty/asmjit/src/asmjit/x86/x86assembler.h
    3rdParty/asmjit/src/asmjit/x86/x86builder.cpp
    3rdParty/asmjit/src/asmjit/x86/x86builder.h
    3rdParty/asmjit/src/asmjit/x86/x86callconv.cpp
    3rdParty/asmjit/src/asmjit/x86/x86callconv_p.h
    3rdParty/asmjit/src/asmjit/x86/x86compiler.cpp
    3rdParty/asmjit/src/asmjit/x86/x86compiler.h
    3rdParty/asmjit/src/asmjit/x86/x86emitter.h
    3rdParty/asmjit/src/asmjit/x86/x86features.cpp
    3rdParty/asmjit/src/asmjit/x86/x86features.h
    3rdParty/asmjit/src/asmjit/x86/x86globals.h
    3rdParty/asmjit/src/asmjit/x86/x86instapi.cpp
    3rdParty/asmjit/src/asmjit/x86/x86instapi_p.h
    3rdParty/asmjit/src/asmjit/x86/x86instdb.cpp
    3rdParty/asmjit/src/asmjit/x86/x86instdb.h
    3rdParty/asmjit/src/asmjit/x86/x86instdb_p.h
    3rdParty/asmjit/src/asmjit/x86/x86internal.cpp
    3rdParty/asmjit/src/asmjit/x86/x86internal_p.h
    3rdParty/asmjit/src/asmjit/x86/x86logging.cpp
    3rdParty/asmjit/src/asmjit/x86/x86logging_p.h
    3rdParty/asmjit/src/asmjit/x86/x86opcode_p.h
    3rdParty/asmjit/src/asmjit/x86/x86operand.cpp
    3rdParty/asmjit/src/asmjit/x86/x86operand.h
    3rdParty/asmjit/src/asmjit/x86/x86rapass.cpp
    3rdParty/asmjit/src/asmjit/x86/x86rapass_p.h

    3rdParty/B-Human/Platform/BHAssert.cpp
    3rdParty/B-Human/Platform/BHAssert.h
    3rdParty/B-Human/Tools/Math/BHMath.h
    3rdParty/B-Human/Tools/Math/NeumaierSum.h
    3rdParty/B-Human/Tools/Streams/InOut.cpp
    3rdParty/B-Human/Tools/Streams/InOut.h
    3rdParty/B-Human/Tools/Streams/InStreams.cpp
    3rdParty/B-Human/Tools/Streams/InStreams.h
    3rdParty/B-Human/Tools/Streams/SimpleMap.cpp
    3rdParty/B-Human/Tools/Streams/SimpleMap.h
)

target_include_directories(CompiledNN PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Src")
target_include_directories(CompiledNN SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/B-Human")
target_include_directories(CompiledNN SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/asmjit/src")
target_compile_definitions(CompiledNN PRIVATE ASMJIT_STATIC ASMJIT_BUILD_X86 ASMJIT_NO_BUILDER ASMJIT_NO_COMPILER
    ASMJIT_NO_LOGGING ASMJIT_NO_TEXT ASMJIT_NO_INST_API)
target_link_libraries(CompiledNN PUBLIC $<$<PLATFORM_ID:Linux>:pthread> $<$<PLATFORM_ID:Linux>:rt>)

find_package(HDF5 REQUIRED)
target_link_libraries(CompiledNN PRIVATE ${HDF5_C_LIBRARIES})
target_include_directories(CompiledNN SYSTEM PRIVATE ${HDF5_C_INCLUDE_DIRS})
target_compile_definitions(CompiledNN PRIVATE ${HDF5_C_DEFINITIONS})
