cmake_minimum_required(VERSION 3.15)
project(CompiledNN VERSION 1.0.0 LANGUAGES C CXX)

option(WITH_APPLICATIONS "Build applications." OFF)
option(WITH_TESTS "Build tests." OFF)
option(WITH_COVERAGE "Enable coverage." OFF)
option(WITH_KERAS_HDF5 "Enable Keras HDF5 model format." ON)
option(WITH_ONNX "Enable ONNX model format." ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(CompiledNN
    Src/CompiledNN/CompiledNN.cpp
    Src/CompiledNN/CompiledNN.h
    Src/CompiledNN/Model.cpp
    Src/CompiledNN/Model.h
    Src/CompiledNN/SimpleNN.cpp
    Src/CompiledNN/SimpleNN.h
    Src/CompiledNN/Tensor.h
    Src/CompiledNN/CompiledNN/ActivationFunctions.cpp
    Src/CompiledNN/CompiledNN/ActivationFunctions.h
    Src/CompiledNN/CompiledNN/CompilationSettings.cpp
    Src/CompiledNN/CompiledNN/CompilationSettings.h
    Src/CompiledNN/CompiledNN/CompiledNNImpl.h
    Src/CompiledNN/CompiledNN/CompiledNNImplBase.h
    Src/CompiledNN/CompiledNN/TensorPointer.h
    Src/CompiledNN/CompiledNN/Operations/Activation.cpp
    Src/CompiledNN/CompiledNN/Operations/Activation.h
    Src/CompiledNN/CompiledNN/Operations/Arithmetic.cpp
    Src/CompiledNN/CompiledNN/Operations/Arithmetic.h
    Src/CompiledNN/CompiledNN/Operations/BatchNormalization.cpp
    Src/CompiledNN/CompiledNN/Operations/BatchNormalization.h
    Src/CompiledNN/CompiledNN/Operations/Concatenate.cpp
    Src/CompiledNN/CompiledNN/Operations/Concatenate.h
    Src/CompiledNN/CompiledNN/Operations/Conv2D.cpp
    Src/CompiledNN/CompiledNN/Operations/Conv2D.h
    Src/CompiledNN/CompiledNN/Operations/Cropping2D.cpp
    Src/CompiledNN/CompiledNN/Operations/Cropping2D.h
    Src/CompiledNN/CompiledNN/Operations/DConv2D.cpp
    Src/CompiledNN/CompiledNN/Operations/DConv2D.h
    Src/CompiledNN/CompiledNN/Operations/Dense.cpp
    Src/CompiledNN/CompiledNN/Operations/Dense.h
    Src/CompiledNN/CompiledNN/Operations/GlobalPooling2D.cpp
    Src/CompiledNN/CompiledNN/Operations/GlobalPooling2D.h
    Src/CompiledNN/CompiledNN/Operations/Im2Col2D.cpp
    Src/CompiledNN/CompiledNN/Operations/Im2Col2D.h
    Src/CompiledNN/CompiledNN/Operations/Pooling2D.cpp
    Src/CompiledNN/CompiledNN/Operations/Pooling2D.h
    Src/CompiledNN/CompiledNN/Operations/Softmax.cpp
    Src/CompiledNN/CompiledNN/Operations/Softmax.h
    Src/CompiledNN/CompiledNN/Operations/UInt8Input.cpp
    Src/CompiledNN/CompiledNN/Operations/UInt8Input.h
    Src/CompiledNN/CompiledNN/Operations/UpSampling2D.cpp
    Src/CompiledNN/CompiledNN/Operations/UpSampling2D.h
    Src/CompiledNN/CompiledNN/Operations/ZeroPadding2D.cpp
    Src/CompiledNN/CompiledNN/Operations/ZeroPadding2D.h
    Src/CompiledNN/CompiledNN/Util/ExpApprox.cpp
    Src/CompiledNN/CompiledNN/Util/ExpApprox.h

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

target_compile_options(CompiledNN PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall>
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wextra>
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-fno-strict-aliasing>
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
)

if(WITH_COVERAGE)
  target_compile_options(CompiledNN PUBLIC
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fprofile-instr-generate>
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcoverage-mapping>)
  target_link_options(CompiledNN PUBLIC
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fprofile-instr-generate>
      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcoverage-mapping>)
  target_compile_options(CompiledNN PUBLIC $<$<CXX_COMPILER_ID:GNU>:--coverage>)
  target_link_options(CompiledNN PUBLIC $<$<CXX_COMPILER_ID:GNU>:--coverage>)
endif()

target_include_directories(CompiledNN PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Src>"
    "$<INSTALL_INTERFACE:include/CompiledNN>"
)
target_include_directories(CompiledNN SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/B-Human>"
    "$<INSTALL_INTERFACE:include/CompiledNN>"
)
target_include_directories(CompiledNN SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/asmjit/src"
)
target_compile_definitions(CompiledNN PRIVATE
    ASMJIT_STATIC ASMJIT_BUILD_X86 ASMJIT_NO_BUILDER ASMJIT_NO_COMPILER
    ASMJIT_NO_LOGGING ASMJIT_NO_TEXT ASMJIT_NO_VALIDATION ASMJIT_NO_INTROSPECTION
)
target_link_libraries(CompiledNN PUBLIC
    $<$<PLATFORM_ID:Linux>:pthread> $<$<PLATFORM_ID:Linux>:rt>
)
set_target_properties(CompiledNN PROPERTIES
    PUBLIC_HEADER "Src/CompiledNN/CompiledNN.h;Src/CompiledNN/Model.h;Src/CompiledNN/SimpleNN.h;Src/CompiledNN/Tensor.h"
)

if(WITH_KERAS_HDF5)
  find_package(HDF5 REQUIRED)
  if(NOT TARGET hdf5::hdf5-shared)
    add_library(hdf5::hdf5-shared INTERFACE IMPORTED)
    set_target_properties(hdf5::hdf5-shared PROPERTIES
        INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${HDF5_C_DEFINITIONS}"
    )
  endif()
  target_link_libraries(CompiledNN PRIVATE hdf5::hdf5-shared)
  target_compile_definitions(CompiledNN PRIVATE WITH_KERAS_HDF5)
  target_sources(CompiledNN PRIVATE
      Src/CompiledNN/Formats/KerasHDF5.cpp
      Src/CompiledNN/Formats/KerasHDF5.h
  )
endif()

if(WITH_ONNX)
  find_package(Protobuf REQUIRED)
  protobuf_generate_cpp(ONNX_PROTO_SOURCES ONNX_PROTO_HEADERS 3rdParty/onnx/onnx.proto)
  target_sources(CompiledNN PRIVATE ${ONNX_PROTO_SOURCES} ${ONNX_PROTO_HEADERS}
      Src/CompiledNN/Formats/ONNX.cpp
      Src/CompiledNN/Formats/ONNX.h
  )
  target_link_libraries(CompiledNN PRIVATE protobuf::libprotobuf)
  target_compile_definitions(CompiledNN PRIVATE WITH_ONNX)
  target_include_directories(CompiledNN PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(WITH_APPLICATIONS)
  add_executable(Benchmark Tests/Benchmark.cpp)
  target_link_libraries(Benchmark PRIVATE CompiledNN)

  add_executable(Check Tests/Check.cpp)
  target_link_libraries(Check PRIVATE CompiledNN)
endif()

if(WITH_TESTS)
  find_package(GTest)

  if(GTEST_FOUND)
    enable_testing()

    add_executable(LayerTests
        Tests/Layers/UpSampling2D.cpp
        Tests/Layers/ZeroPadding2D.cpp
    )
    target_link_libraries(LayerTests PRIVATE GTest::Main)
    target_link_libraries(LayerTests PRIVATE CompiledNN)
    gtest_discover_tests(LayerTests)
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# install library
install(TARGETS CompiledNN
    EXPORT CompiledNNTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/CompiledNN"
)
# install binaries
if(WITH_APPLICATIONS)
    install(TARGETS Benchmark
        EXPORT CompiledNNTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
    install(TARGETS Check
        EXPORT CompiledNNTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endif()

# install 3rd-party headers (because PUBLIC_HEADER does not support nested directories)
install(FILES
    Src/CompiledNN/CompiledNN/CompilationSettings.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/CompiledNN/CompiledNN"
)
install(FILES
    3rdParty/B-Human/Platform/BHAssert.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/CompiledNN/Platform"
)
install(FILES
    3rdParty/B-Human/Tools/Math/BHMath.h
    3rdParty/B-Human/Tools/Math/NeumaierSum.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/CompiledNN/Tools/Math"
)

# install targets, configuration and version files
install(EXPORT CompiledNNTargets
    NAMESPACE CompiledNN::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/CompiledNN"
)
configure_file(CompiledNNConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/CompiledNNConfig.cmake"
    COPYONLY
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CompiledNNConfigVersion.cmake"
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CompiledNNConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CompiledNNConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/CompiledNN"
)
