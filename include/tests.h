#define LENGTH 5
#define TOLERANCE 0.0001f
#define BATCH_SIZE 3

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FLOAT_EQ(a, b) (fabsf((a) - (b)) < EPSILON)