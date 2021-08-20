/**
 * @file BHMath.h
 *
 * This contains some often used mathematical definitions and functions.
 *
 * @author <a href="mailto:alexists@tzi.de">Alexis Tsogias</a>
 */

#pragma once

/**
 * Calculates the square of a value.
 * @param a The value.
 * @return The square of \c a.
 */
template<typename V>
constexpr V sqr(const V& a) { return a * a; }
