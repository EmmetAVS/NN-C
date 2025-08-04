#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include "model.h"

bool write_model_params(Model *model, const char *filename);
bool load_model_params(Model *model, const char *filename);

#endif