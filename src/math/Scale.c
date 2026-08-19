// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include "Scale.h"

// The group covering an entry position
static int ScaleGroup(Scale s, int position) { return s.stride ? position / s.stride : 0; }
Real ScaleValue(Scale s, int group)          { return s.values ? s.values[group] : (Real)1.0; }
Real ScaleZero(Scale s, int group)           { return s.zeros  ? s.zeros[group]  : (Real)0.0; }

Real ScaleAt(Scale s, int position)     { return ScaleValue(s, ScaleGroup(s, position)); }
Real ScaleZeroAt(Scale s, int position) { return ScaleZero(s, ScaleGroup(s, position));  }

void ScaleNew(Scale *s, int groups, int stride, int zeros)
{
    s->values = calloc((size_t)groups * (zeros ? 2 : 1), sizeof(Real));
    if (!s->values) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    s->zeros  = zeros ? s->values + groups : NULL;
    s->groups = groups;
    s->stride = groups >= 2 ? stride : 0; // single group uses one scale
}

void ScaleFree(Scale *s)
{
    free(s->values);
    s->values = NULL;
    s->zeros  = NULL;
    s->groups = 0;
    s->stride = 0;
}
