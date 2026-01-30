#ifndef RF_INFERENCE_H
#define RF_INFERENCE_H

#include <ap_int.h>

#define N_FEATURES 6
#define N_TREES 80
#define VOTE_REQUIRED 56

// input: quantized int features
int rf_predict(ap_int<16> x[N_FEATURES]);

#endif
