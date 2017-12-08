#pragma once

#include "Layer.h"

namespace rna
{

class LogSoftMax: public Layer
{
    public:
        LogSoftMax(): Layer("LogSoftMax") {}

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

    private:
        virtual void toGPU(const cl_context& _context, const cl_device_id& _deviceId) override;
};

}
