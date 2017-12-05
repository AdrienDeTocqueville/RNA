#pragma once

#include "Layer.h"

namespace rna
{

class MaxPooling: public Layer
{
    public:
        MaxPooling(): Layer("MaxPooling") {}

        // _input size must be divisible by 2
        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);

    private:
        Tensor indices;
};

}
