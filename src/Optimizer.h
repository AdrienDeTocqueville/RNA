#pragma once

#include "Losses/LossFunction.h"

namespace rna
{

class Network;

template <typename L>
class Optimizer
{
    friend class Network;

    public:
        Optimizer(Tensor::value_type _learningRate, Tensor::value_type _inertia):
            learningRate(std::max(_learningRate, Tensor::value_type(0.0))), // must be positive
            inertia(std::min(std::max(Tensor::value_type(0.0), _inertia), Tensor::value_type(1.0))) // clamp between 0 and 1
        { }

    private:
        L loss;

        Tensor::value_type learningRate;
        Tensor::value_type inertia;
};

}
