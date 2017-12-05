#pragma once

#include "Layer.h"

namespace rna
{

class LogSoftMax: public Layer
{
    public:
        LogSoftMax(): Layer("LogSoftMax") {}

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);
};

}
