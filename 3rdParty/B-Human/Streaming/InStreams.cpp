/**
 * @file InStreams.cpp
 *
 * Implementation of in stream classes.
 *
 * @author Thomas Röfer
 * @author Martin Lötzsch
 */


#include "InStreams.h"
#include "Platform/BHAssert.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

void StreamReader::skipData(size_t size, PhysicalInStream& stream)
{
  // default implementation
  char* dummy = new char[size];
  readData(dummy, size, stream);
  delete[] dummy;
}

void PhysicalInStream::skipInStream(size_t size)
{
  // default implementation
  char* dummy = new char[size];
  readFromStream(dummy, size);
  delete[] dummy;
}

void InText::readString(std::string& value, PhysicalInStream& stream)
{
  value = "";
  skipWhitespace(stream);
  bool containsSpaces = theChar == '"';
  if(containsSpaces && !isEof(stream))
    nextChar(stream);
  while(!isEof(stream) && (containsSpaces || !isWhitespace()) && (!containsSpaces || theChar != '"'))
  {
    if(theChar == '\\')
    {
      nextChar(stream);
      if(theChar == 'n')
        theChar = '\n';
      else if(theChar == 'r')
        theChar = '\r';
      else if(theChar == 't')
        theChar = '\t';
    }
    value += theChar;
    if(!isEof(stream))
      nextChar(stream);
  }
  if(containsSpaces && !isEof(stream))
    nextChar(stream);
  skipWhitespace(stream);
}

void InText::readData(void* p, size_t size, PhysicalInStream& stream)
{
  for(size_t i = 0; i < size; ++i)
    readChar(*reinterpret_cast<char*&>(p)++, stream);
}

bool InText::isWhitespace()
{
  return theChar == ' ' || theChar == '\n' || theChar == '\r' || theChar == '\t';
}

void InText::skipWhitespace(PhysicalInStream& stream)
{
  while(!isEof(stream) && isWhitespace())
    nextChar(stream);
}

void InText::readBool(bool& value, PhysicalInStream& stream)
{
  skipWhitespace(stream);
  if(!isEof(stream))
  {
    if(theChar == '0' || theChar == '1')
    {
      value = theChar != '0';
      nextChar(stream);
    }
    else
    {
      value = theChar != 'f';
      static const char* falseString = "false";
      static const char* trueString = "true";
      const char* p = value ? trueString : falseString;
      while(!isEof(stream) && *p && theChar == *p)
      {
        ++p;
        nextChar(stream);
      }
    }
  }
}

void InText::readChar(char& d, PhysicalInStream& stream)
{
  int i;
  readInt(i, stream);
  d = static_cast<char>(i);
}

void InText::readSChar(signed char& d, PhysicalInStream& stream)
{
  int i;
  readInt(i, stream);
  d = static_cast<signed char>(i);
}

void InText::readUChar(unsigned char& d, PhysicalInStream& stream)
{
  unsigned u;
  readUInt(u, stream);
  d = static_cast<unsigned char>(u);
}

void InText::readShort(short& d, PhysicalInStream& stream)
{
  int i;
  readInt(i, stream);
  d = static_cast<short>(i);
}

void InText::readUShort(unsigned short& d, PhysicalInStream& stream)
{
  unsigned u;
  readUInt(u, stream);
  d = static_cast<unsigned short>(u);
}

void InText::readInt(int& d, PhysicalInStream& stream)
{
  skipWhitespace(stream);
  int sign = 1;
  if(!isEof(stream) && theChar == '-')
  {
    sign = -1;
    nextChar(stream);
  }
  else if(!isEof(stream) && theChar == '+')
    nextChar(stream);
  unsigned u;
  readUInt(u, stream);
  d = sign * static_cast<int>(u);
}

void InText::readUInt(unsigned int& d, PhysicalInStream& stream)
{
  buf = "";
  skipWhitespace(stream);
  while(!isEof(stream) && isdigit(theChar))
  {
    buf += theChar;
    nextChar(stream);
  }
  d = static_cast<unsigned>(strtoul(buf.c_str(), nullptr, 0));
  skipWhitespace(stream);
}

void InText::readFloat(float& d, PhysicalInStream& stream)
{
  double f;
  readDouble(f, stream);
  d = static_cast<float>(f);
}

void InText::readDouble(double& d, PhysicalInStream& stream)
{
  buf = "";
  skipWhitespace(stream);
  if(!isEof(stream) && (theChar == '-' || theChar == '+'))
  {
    buf += theChar;
    nextChar(stream);
  }
  while(!isEof(stream) && isdigit(theChar))
  {
    buf += theChar;
    nextChar(stream);
  }
  if(!isEof(stream) && theChar == '.')
  {
    buf += theChar;
    nextChar(stream);
  }
  while(!isEof(stream) && isdigit(theChar))
  {
    buf += theChar;
    nextChar(stream);
  }
  if(!isEof(stream) && (theChar == 'e' || theChar == 'E'))
  {
    buf += theChar;
    nextChar(stream);
  }
  if(!isEof(stream) && (theChar == '-' || theChar == '+'))
  {
    buf += theChar;
    nextChar(stream);
  }
  while(!isEof(stream) && isdigit(theChar))
  {
    buf += theChar;
    nextChar(stream);
  }
  d = atof(buf.c_str());
  skipWhitespace(stream);
}

bool InText::expectString(const std::string& str, PhysicalInStream& stream)
{
  const char* p = str.c_str();
  if(str.length())
  {
    while(*p)
    {
      if(isEof(stream) || theChar != *p)
        return false;
      ++p;
      nextChar(stream);
    }
  }
  return true;
}

void InMemory::readFromStream(void* p, size_t size)
{
  if(memory)
  {
    std::memcpy(p, memory, size);
    memory += size;
  }
}
