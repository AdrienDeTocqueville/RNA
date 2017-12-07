#pragma once

// TODO
// Network::feedforward return ref
// inertia

// use clCreateImage2D for image tensor

#include "Network.h"

#include "LossFunction.h"

#include "Layers/Layer.h"
#include "Layers/Linear.h"
#include "Layers/Reshape.h"
#include "Layers/MaxPooling.h"
#include "Layers/LogSoftMax.h"
#include "Layers/Convolutional.h"
