/**
 * @file InOut.cpp
 *
 * Implementation of the streamable function endl.
 *
 * @author <a href="mailto:oberlies@sim.tu-darmstadt.de">Tobias Oberlies</a>
 */

#include "InOut.h"

In& endl(In& in)
{
  in.inEndL();
  return in;
}
