#pragma once

#include "Layer.h"

namespace rna
{

class LogSoftMax: public Layer
{
    public:
        LogSoftMax(): Layer("LogSoftMax") {}

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);

        virtual void feedForward(cl::CommandQueue&, const Tensor& _inputBatch);
        virtual void backprop(cl::CommandQueue&, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        #else
        virtual void feedForward(const Tensor& _input);
        virtual void backprop(const Tensor& _input, const Tensor& _outputGrad);
        #endif // USE_OPENCL
};

}
