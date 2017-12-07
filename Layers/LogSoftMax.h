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

        virtual void GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch);

    private:
        virtual void toGPU(cl_context _context, cl_device_id _device);
};

}
