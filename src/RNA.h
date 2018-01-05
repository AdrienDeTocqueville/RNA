#pragma once

#include "Network.h"
#include "SGD.h"

#include "Losses/MSE.h"
#include "Losses/NLL.h"

#include "Layers/Linear.h"
#include "Layers/Reshape.h"
#include "Layers/MaxPooling.h"
#include "Layers/LogSoftMax.h"
#include "Layers/Convolutional.h"

#include "Layers/activations.h"

#include "Optimizers/Momentum.h"

// TODO: Call clSetKernelArg once for const args ?
// TODO: remove all readBuffers
// TODO: add dropout layer and rectifiers
