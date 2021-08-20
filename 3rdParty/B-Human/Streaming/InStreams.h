/**
 * @file InStreams.h
 *
 * Declaration of in stream classes for different media and formats.
 *
 * @author Thomas Röfer
 * @author Martin Lötzsch
 */

#pragma once

#include "SimpleMap.h"

/**
 * @class PhysicalInStream
 *
 * The base class for physical in streams. Derivates of PhysicalInStream only handle the
 * reading of data from a medium, not of formatting data.
 */
class PhysicalInStream
{
public:
  /**
   * The function reads a number of bytes from a stream.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   */
  virtual void readFromStream(void* p, size_t size) = 0;

  /**
   * The function skips a number of bytes in a stream.
   * @param size The number of bytes to be read.
   */
  virtual void skipInStream(size_t size);

  /**
   * The function states whether this stream actually exists.
   * This function is relevant if the stream represents a file.
   * @return Does the stream exist?
   */
  virtual bool exists() const {return true;}

  /**
   * The function states whether the end of the stream has been reached.
   * @return End of stream reached?
   */
  virtual bool getEof() const = 0;
};

/**
 * @class StreamReader
 *
 * Generic class for formatted reading of data to be used in streams.
 * The physical reading is then done by PhysicalOutStream derivates.
 */
class StreamReader
{
protected:
  /**
   * reads a bool from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readBool(bool& d, PhysicalInStream& stream) = 0;

  /**
   * reads a character from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readChar(char& d, PhysicalInStream& stream) = 0;

  /**
   * reads a signed character from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readSChar(signed char& d, PhysicalInStream& stream) = 0;

  /**
   * reads an unsigned character from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readUChar(unsigned char& d, PhysicalInStream& stream) = 0;

  /**
   * reads a short from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readShort(short& d, PhysicalInStream& stream) = 0;

  /**
   * reads an unsigned short from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readUShort(unsigned short& d, PhysicalInStream& stream) = 0;

  /**
   * reads an int from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readInt(int& d, PhysicalInStream& stream) = 0;

  /**
   * reads an unsigned int from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readUInt(unsigned int& d, PhysicalInStream& stream) = 0;

  /**
   * reads a float from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readFloat(float& d, PhysicalInStream& stream) = 0;

  /**
   * reads a double from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readDouble(double& d, PhysicalInStream& stream) = 0;

  /**
   * reads a string from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  virtual void readString(std::string& d, PhysicalInStream& stream) = 0;

  /**
   * reads the 'end of line' from a stream
   * @param stream the stream to read from
   */
  virtual void readEndl(PhysicalInStream& stream) = 0;

  /**
   * The function reads a number of bytes from the file.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   * @param stream The stream to read from.
   */
  virtual void readData(void* p, size_t size, PhysicalInStream& stream) = 0;

  /**
   * The function skips a number of bytes in the file.
   * @param size The number of bytes to be skipped.
   * @param stream The stream to read from.
   */
  virtual void skipData(size_t size, PhysicalInStream& stream);

  /**
   * The function states whether the end of the stream has been reached.
   * @param stream The stream to be tested
   * @return End of stream reached?
   */
  virtual bool isEof(const PhysicalInStream& stream) const = 0;
};

/**
 * @class InStream
 *
 * Generic class for classes that do both formatted and physical reading of data from streams.
 */
template<typename S, typename R> class InStream : public S, public R, public In
{
public:
  /**
   * The function reads a number of bytes from a stream.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   */
  void read(void* p, size_t size) override
  {
    R::readData(p, size, *this);
  }

  /**
   * The function skips a number of bytes in the stream.
   * @param size The number of bytes to be skipped.
   */
  void skip(size_t size) override
  {
    R::skipData(size, *this);
  }

  /**
   * Determines whether the end of file has been reached.
   */
  bool eof() const override { return R::isEof(*this); }

protected:
  /**
   * Virtual redirection for operator>>(bool& value).
   */
  void inBool(bool& d) override
  {
    R::readBool(d, *this);
  }

  /**
   * Virtual redirection for operator>>(char& value).
   */
  void inChar(char& d) override
  {
    R::readChar(d, *this);
  }

  /**
   * Virtual redirection for operator>>(signed char& value).
   */
  void inSChar(signed char& d) override
  {
    R::readSChar(d, *this);
  }

  /**
   * Virtual redirection for operator>>(unsigned char& value).
   */
  void inUChar(unsigned char& d) override
  {
    R::readUChar(d, *this);
  }

  /**
   * Virtual redirection for operator>>(short& value).
   */
  void inShort(short& d) override
  {
    R::readShort(d, *this);
  }

  /**
   * Virtual redirection for operator>>(unsigned short& value).
   */
  void inUShort(unsigned short& d) override
  {
    R::readUShort(d, *this);
  }

  /**
   * Virtual redirection for operator>>(int& value).
   */
  void inInt(int& d) override
  {
    R::readInt(d, *this);
  }

  /**
   * Virtual redirection for operator>>(unsigned int& value).
   */
  void inUInt(unsigned int& d) override
  {
    R::readUInt(d, *this);
  }

  /**
   * Virtual redirection for operator>>(float& value).
   */
  void inFloat(float& d) override
  {
    R::readFloat(d, *this);
  }

  /**
   * Virtual redirection for operator>>(double& value).
   */
  void inDouble(double& d) override
  {
    R::readDouble(d, *this);
  }

  /**
   * Virtual redirection for operator>>(std::string& value).
   */
  void inString(std::string& d) override
  {
    R::readString(d, *this);
  }

  /**
   * Virtual redirection for operator>>(In& (*f)(In&)) that reads
   * the symbol "endl";
   */
  void inEndL() override
  {
    R::readEndl(*this);
  }
};

/**
 * @class InText
 *
 * Formatted reading of text data to be used in streams.
 * The physical reading is done by PhysicalInStream derivates.
 */
class InText : public StreamReader
{
protected:
  /** The last character read. */
  char theChar = ' ',
       theNextChar = ' ';
private:
  std::string buf; /**< A buffer to convert read strings. */
  bool eof = false,  /**< Stores whether the end of file was reached during the last call to nextChar. */
       nextEof = false;

public:
  InText() {buf.reserve(200);};

  /**
   * Resets theChar to be able to use the same instance of InText or InConfig
   * more than once.
   */
  void reset()
  {
    theChar = theNextChar = ' ';
    eof = nextEof = false;
  }

protected:
  /**
   * The function initializes the end-of-file flag.
   * It has to be called only once after the stream was initialized.
   * @param stream The stream.
   */
  virtual void initEof(PhysicalInStream& stream)
  {
    eof = nextEof = stream.getEof();
    if(stream.exists())
      nextChar(stream);
  }

  /**
   * The function returns whether the end of stream has been reached.
   * If this function returns false, "theChar" is valid, otherwise it is not.
   * @param stream The stream.
   * @return End of stream reached?
   */
  bool isEof(const PhysicalInStream& stream) const override;

  /**
   * reads a bool from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readBool(bool& d, PhysicalInStream& stream) override;

  /**
   * reads a character from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readChar(char& d, PhysicalInStream& stream) override;

  /**
   * reads a signed character from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readSChar(signed char& d, PhysicalInStream& stream) override;

  /**
   * reads an unsigned character from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readUChar(unsigned char& d, PhysicalInStream& stream) override;

  /**
   * reads a short from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readShort(short& d, PhysicalInStream& stream) override;

  /**
   * reads an unsigned short from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readUShort(unsigned short& d, PhysicalInStream& stream) override;

  /**
   * reads an int from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readInt(int& d, PhysicalInStream& stream) override;

  /**
   * reads an unsigned int from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readUInt(unsigned int& d, PhysicalInStream& stream) override;

  /**
   * reads a float from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readFloat(float& d, PhysicalInStream& stream) override;

  /**
   * reads a double from a stream
   * @param d the data to read from the stream
   * @param stream the stream to read from
   */
  void readDouble(double& d, PhysicalInStream& stream) override;

  /**
   * The function reads a string from a stream.
   * It skips all whitespace characters, and then reads
   * a sequence of non-whitespace characters to a buffer, until it
   * again recognizes a whitespace.
   * @param d The value that is read.
   * @param stream the stream to read from
   */
  void readString(std::string& d, PhysicalInStream& stream) override;

  /**
   * reads the 'end of line' from a stream
   * @param stream the stream to read from
   */
  void readEndl(PhysicalInStream& stream) override;

  /**
   * The function determines whether the current character is a whitespace.
   */
  virtual bool isWhitespace();

  /**
   * The function skips the whitespace.
   */
  virtual void skipWhitespace(PhysicalInStream& stream);

  /**
   * The function reads the next character from the stream.
   */
  virtual void nextChar(PhysicalInStream& stream)
  {
    if(!eof)
    {
      eof = nextEof;
      theChar = theNextChar;
      if(stream.getEof())
      {
        nextEof = true;
        theNextChar = ' ';
      }
      else
        stream.readFromStream(&theNextChar, 1);
    }
  }

  /**
   * The function reads a number of bytes from the file.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   * @param stream The stream to read from.
   */
  void readData(void* p, size_t size, PhysicalInStream& stream) override;

private:
  /**
   * Tries to read the given string from the stream.
   * @param str The string which is expected.
   * @param stream The stream to read from.
   * @return true if the string could be read from the stream.
   */
  bool expectString(const std::string& str, PhysicalInStream& stream);
};

inline bool InText::isEof(const PhysicalInStream&) const { return eof; }

inline void InText::readEndl(PhysicalInStream&) {}

/**
 * @class InBinary
 *
 * Formatted reading of binary data to be used in streams.
 * The physical reading is done by PhysicalInStream derivates.
 */
class InBinary : public StreamReader
{
protected:
  /**
   * The function returns whether the end of stream has been reached.
   * @return End of stream reached?
   */
  bool isEof(const PhysicalInStream& stream) const override { return stream.getEof(); }

  /**
   * The function reads a bool from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readBool(bool& d, PhysicalInStream& stream) override
  {
    char c;
    stream.readFromStream(&c, sizeof(d));
    d = c != 0;
  }

  /**
   * The function reads a char from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readChar(char& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads an signed char from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readSChar(signed char& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads an unsigned char from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readUChar(unsigned char& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads a short int from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readShort(short& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads an unsigned short int from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readUShort(unsigned short& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads an int from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readInt(int& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads an unsigned int from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readUInt(unsigned int& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads a float from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readFloat(float& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads a double from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readDouble(double& d, PhysicalInStream& stream) override
  {
    stream.readFromStream(&d, sizeof(d));
  }

  /**
   * The function reads a string from the stream.
   * @param d The value that is read.
   * @param stream A stream to read from.
   */
  void readString(std::string& d, PhysicalInStream& stream) override
  {
    size_t size = 0;
    stream.readFromStream(&size, sizeof(unsigned));
    d.resize(size);
    if(size)
      stream.readFromStream(&d[0], size);
  }

  /**
   * The function is intended to read an endl-symbol from the stream.
   * In fact, the function does nothing.
   * @param stream A stream to read from.
   */
  void readEndl(PhysicalInStream& stream) override;

  /**
   * The function reads a number of bytes from a stream.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   * @param stream A stream to read from.
   */
  void readData(void* p, size_t size, PhysicalInStream& stream) override
  {
    stream.readFromStream(p, size);
  }

  /**
   * The function skips a number of bytes in the file.
   * @param size The number of bytes to be skipped.
   * @param stream The stream to read from.
   */
  void skipData(size_t size, PhysicalInStream& stream) override
  {
    stream.skipInStream(size);
  }
};

inline void InBinary::readEndl(PhysicalInStream&) {}

/**
 * @class InMemory
 *
 * An PhysicalInStream that reads the data from a memory region.
 */
class InMemory : public PhysicalInStream
{
private:
  const char* memory = nullptr, /**< Points to the next byte to read from memory. */
            * end = nullptr; /**< Points to the end of the memory block. */

public:
  /**
   * The function states whether the stream actually exists.
   * @return Does the stream exist? This is always true for memory streams.
   */
  bool exists() const override {return (memory != nullptr);}

  /**
   * The function states whether the end of the file has been reached.
   * It will only work if the correct size of the memory block was
   * specified during the construction of the stream.
   * @return End of file reached?
   */
  bool getEof() const override
  {
    return memory != nullptr && memory >= end;
  }

protected:
  /**
   * Opens the stream.
   * @param mem The address of the memory block from which is read.
   * @param size The size of the memory block. It is only used to
   *             implement the function eof(). If the size is not
   *             specified, eof() will always return true, but reading
   *             from the stream is still possible.
   */
  void open(const void* mem, size_t size = 0)
  {
    if(memory == nullptr)
    {
      memory = reinterpret_cast<const char*>(mem);
      end = memory + size;
    }
  }

  /**
   * The function reads a number of bytes from memory.
   * @param p The address the data is written to. Note that p
   *          must point to a memory area that is at least
   *          "size" bytes large.
   * @param size The number of bytes to be read.
   */
  void readFromStream(void* p, size_t size) override;

  /**
   * The function skips a number of bytes.
   * @param size The number of bytes to be skipped.
   */
  void skipInStream(size_t size) override {memory += size;}
};

/**
 * @class InBinaryMemory
 *
 * A Binary Stream from a memory region.
 */
class InBinaryMemory : public InStream<InMemory, InBinary>
{
public:
  /**
   * Constructor.
   * @param mem The address of the memory block from which is read.
   * @param size The size of the memory block. It is only used to
   *             implement the function eof(). If the size is not
   *             specified, eof() will always return true, but reading
   *             from the stream is still possible.
   */
  InBinaryMemory(const void* mem, size_t size = 0) {open(mem, size);}

  /**
   * The function returns whether this is a binary stream.
   * @return Does it output data in binary format?
   */
  bool isBinary() const override {return true;}
};

/**
 * @class InTextMemory
 *
 * A Binary Stream from a memory region.
 */
class InTextMemory : public InStream<InMemory, InText>
{
public:
  /**
   * Constructor.
   * @param mem The address of the memory block from which is read.
   * @param size The size of the memory block. It is only used to
   *             implement the function eof().
   */
  InTextMemory(const void* mem, size_t size)
  {
    open(mem, size);
    initEof(*this);
  }
};
