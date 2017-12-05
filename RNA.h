#pragma once

// TODO
// feedforward return ref
// better backprop gradinput return
// network.addlayer template
// inertia

#include "Network.h"

#include "LossFunction.h"

#include "Layers/Layer.h"
#include "Layers/Linear.h"
#include "Layers/Reshape.h"
#include "Layers/MaxPooling.h"
#include "Layers/LogSoftMax.h"
#include "Layers/Convolutional.h"
