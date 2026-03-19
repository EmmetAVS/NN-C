#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "model2d.h"

bool write_model2d_params(Model2D *model, const char *filename);
bool load_model2d_params(Model2D *model, const char *filename);

#endif