/**
 * @file Benchmark.cpp
 *
 * This file contains a program to benchmark the performance of CompiledNN on a model.
 *
 * @author Arne Hasselbring
 */

#include "CompiledNN/Model.h"
#include "CompiledNN/CompiledNN.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifdef __linux__

#include <time.h>

#define TIMESTAMP_DECLARE(t) timespec t
#define TIMESTAMP_GET(t) clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)
#define TIMESTAMP_DIFF(t1, t2) (static_cast<std::int64_t>(t2.tv_sec - t1.tv_sec) * 1000000000ll + (t2.tv_nsec - t1.tv_nsec))

#else

#error "Time measurement not implemented yet for Windows/macOS."

#endif

int main(int argc, char* argv[])
{
  if(argc != 3)
  {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "Benchmark") << " <path to model> <number of iterations>\n";
    return EXIT_FAILURE;
  }

  TIMESTAMP_DECLARE(t1);
  TIMESTAMP_DECLARE(t2);

  TIMESTAMP_GET(t1);
  NeuralNetwork::Model model(argv[1]);
  NeuralNetwork::CompiledNN nn;
  nn.compile(model);
  TIMESTAMP_GET(t2);

  std::cout << "Loading and compilation time: " << TIMESTAMP_DIFF(t1, t2) << "ns\n";

  const unsigned int iterations = std::atoi(argv[2]);

  nn.apply();
  nn.apply();
  nn.apply();
  nn.apply();
  nn.apply();
  nn.apply();


  TIMESTAMP_GET(t1);
  for(unsigned int i = 0; i < iterations; ++i)
    nn.apply();
  TIMESTAMP_GET(t2);

  std::cout << "Average execution time over " << iterations << " runs: " << TIMESTAMP_DIFF(t1, t2) / iterations << "ns\n";

  return EXIT_SUCCESS;
}

