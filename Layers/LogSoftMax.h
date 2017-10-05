#pragma once

#include "Layers.h"

namespace rna
{

class LogSoftMax: public Layer
{
    public:
        LogSoftMax(): Layer("LogSoftMax") {}

        virtual Tensor feedForward(const Tensor& _input);
        virtual Tensor backprop(const Tensor& _input, const Tensor& _gradOutput);
};

}
