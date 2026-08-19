#ifndef SCALE_H
#define SCALE_H

#include "../tensorgrid.h" // Real

/*
 * Maps stored entries onto the real numbers, so they can be kept in a narrower
 * datatype. Entries are divided into groups, and the entry at offset i reads as
 * values[g] * (entry - zeros[g]) for the group g covering i.
 */
typedef struct {
    Real *values; // group scales, NULL if unscaled
    Real *zeros;  // group zero points, in stored units, NULL if symmetric
    int   groups; // number of groups
    int   stride; // distance between groups, 0 for a single group
} Scale;

/*
 * Allocates a Scale with the given number of groups, stride and zero points.
 */
void ScaleNew(Scale *s, int groups, int stride, int zeros);

/*
 * Frees a scale and resets it to unscaled.
 */
void ScaleFree(Scale *s);

/*
 * A group's scale and zero point. An unscaled scale reads as 1 and a symmetric one
 * as a zero point of 0.
 */
Real ScaleValue(Scale s, int group);
Real ScaleZero(Scale s, int group);

/*
 * The same by entry position rather than by group.
 */
Real ScaleAt(Scale s, int position);
Real ScaleZeroAt(Scale s, int position);

#endif
