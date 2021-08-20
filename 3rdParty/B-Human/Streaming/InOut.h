/**
 * @file InOut.h
 *
 * Definition of the abstract base classes In and Out for streams.
 * Include this header file for declaring streaming operators.
 *
 * @author Thomas Röfer
 * @author Martin Lötzsch
 */

#pragma once

#include <string>

/**
 * The class In is the abstract base class for all classes
 * that implement reading from streams.
 */
class In
{
protected:
  /**
   * Virtual redirection for operator>>(bool& value).
   */
  virtual void inBool(bool&) = 0;

  /**
   * Virtual redirection for operator>>(char& value).
   */
  virtual void inChar(char&) = 0;

  /**
   * Virtual redirection for operator>>(signed char& value).
   */
  virtual void inSChar(signed char&) = 0;

  /**
   * Virtual redirection for operator>>(unsigned char& value).
   */
  virtual void inUChar(unsigned char&) = 0;

  /**
   * Virtual redirection for operator>>(short& value).
   */
  virtual void inShort(short&) = 0;

  /**
   * Virtual redirection for operator>>(unsigned short& value).
   */
  virtual void inUShort(unsigned short&) = 0;

  /**
   * Virtual redirection for operator>>(int& value).
   */
  virtual void inInt(int&) = 0;

  /**
   * Virtual redirection for operator>>(unsigned int& value).
   */
  virtual void inUInt(unsigned int&) = 0;

  /**
   * Virtual redirection for operator>>(float& value).
   */
  virtual void inFloat(float&) = 0;

  /**
   * Virtual redirection for operator>>(double& value).
   */
  virtual void inDouble(double&) = 0;

  /**
   * Virtual redirection for operator>>(std::string& value).
   */
  virtual void inString(std::string&) = 0;

  /**
   * Virtual redirection for operator>>(In& (*f)(In&)) that reads
   * the symbol "endl";
   */
  virtual void inEndL() = 0;

public:
  /** Virtual destructor for derived classes. */
  virtual ~In() = default;

  /**
   * The function reads a number of bytes from a stream.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   */
  virtual void read(void* p, size_t size) = 0;

  /**
   * The function skips a number of bytes in a stream.
   * @param size The number of bytes to be skipped.
   */
  virtual void skip(size_t size) = 0;

  /**
   * Determines whether the end of file has been reached.
   */
  virtual bool eof() const = 0;

  /**
   * The function returns whether this is a binary stream.
   * @return Does it output data in binary format?
   */
  virtual bool isBinary() const { return false; }

  friend In& operator>>(In& in, bool& value);
  friend In& operator>>(In& in, char& value);
  friend In& operator>>(In& in, signed char& value);
  friend In& operator>>(In& in, unsigned char& value);
  friend In& operator>>(In& in, short& value);
  friend In& operator>>(In& in, unsigned short& value);
  friend In& operator>>(In& in, int& value);
  friend In& operator>>(In& in, unsigned int& value);
  friend In& operator>>(In& in, float& value);
  friend In& operator>>(In& in, double& value);
  friend In& operator>>(In& in, std::string& value);
  friend In& operator>>(In& in, In& (*f)(In&));
  friend In& endl(In& stream);
};

/**
 * Operator that reads a Boolean from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, bool& value) {in.inBool(value); return in;}

/**
 * Operator that reads a char from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, char& value) {in.inChar(value); return in;}

/**
 * Operator that reads a signed char from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, signed char& value) {in.inSChar(value); return in;}

/**
 * Operator that reads an unsigned char from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, unsigned char& value) {in.inUChar(value); return in;}

/**
 * Operator that reads a short int from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, short& value) {in.inShort(value); return in;}

/**
 * Operator that reads an unsigned short int from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, unsigned short& value) {in.inUShort(value); return in;}

/**
 * Operator that reads an int from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, int& value) {in.inInt(value); return in;}

/**
 * Operator that reads an unsigned int from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, unsigned int& value) {in.inUInt(value); return in;}

/**
 * Operator that reads a float from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, float& value) {in.inFloat(value); return in;}

/**
 * Operator that reads a double from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, double& value) {in.inDouble(value); return in;}

/**
 * Operator that reads a string from a stream.
 * @param in The stream from which is read.
 * @param value The value that is read.
 * @return The stream.
 */
inline In& operator>>(In& in, std::string& value) {in.inString(value); return in;}

/**
 * Operator that reads the endl-symbol from a stream.
 * @param in The stream from which is read.
 * @param f A function that is normally endl.
 * @return The stream.
 */
inline In& operator>>(In& in, In& (*f)(In&)) {return f(in);}

/**
 * This function can be read from a stream to represent an end of line.
 * @param in The stream the endl-symbol is read from.
 * @return The stream.
 */
In& endl(In& in);
