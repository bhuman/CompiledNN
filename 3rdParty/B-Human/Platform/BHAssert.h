/**
 * @file BHAssert.h
 * This file contains macros for low level debugging
 * @author Thomas Röfer
 * @author Colin Graf
 */

#pragma once

#undef ASSERT
#undef FAIL
#undef VERIFY

#include <string>
#include <sstream>

/**
 * Some tools for low level debugging
 */
class Assert
{
public:
  /**
   * Prints a formated message to stdout
   * @param file The name of the current file (__FILE__)
   * @param line The current file line (__LINE__)
   * @param format The format of the message to be printed
   */
  static void print(const char* file, int line, const char* format, ...);
  static void print(const char* file, int line, const std::string& message);

  /**
   * Aborts the execution of the program
   */
  static void abort();
};

/**
 * ASSERT prints a message if \c cond is \c false and NDEBUG is not defined.
 * ASSERT does not evaluate \c cond if NDEBUG is defined.
 * @param c The condition to be checked.
 */
#ifdef NDEBUG
#define ASSERT(cond) static_cast<void>(0)
#else
#define ASSERT(cond) static_cast<void>((cond) ? 0 : (Assert::print(__FILE__, __LINE__, "ASSERT(%s) failed", #cond), Assert::abort(), 0))
#endif

/**
 * FAIL is equivalent to ASSERT(false) and additionally prints a text.
 * This text is passed into a std::stringstream, thus FAIL("error " << 1) is a valid expression.
 */
#ifdef NDEBUG
#define FAIL(text) static_cast<void>(0)
#else
#define FAIL(text) \
  do \
  { \
    std::stringstream sstream; \
    sstream << "FAIL: " << text; \
    Assert::print(__FILE__, __LINE__, sstream.str()); \
    Assert::abort(); \
  } \
  while(false)
#endif

/**
 * VERIFY prints a message if \cond cond is \c false and NDEBUG is not defined.
 * VERIFY does evaluate \c cond even if NDEBUG is defined.
 * @param c The condition to be checked.
 */
#ifdef NDEBUG
#define VERIFY(cond) static_cast<void>(cond)
#else
#define VERIFY(cond) static_cast<void>((cond) ? 0 : (Assert::print(__FILE__, __LINE__, "VERIFY(%s) failed", #cond), Assert::abort(), 0))
#endif
