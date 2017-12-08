#pragma once

#include "Layer.h"

namespace rna
{

class MaxPooling: public Layer
{
    public:
        MaxPooling(): Layer("MaxPooling") {}

        virtual void feedForwardCPU(const Tensor& _input); // _input size must be divisible by 2
//        virtual void feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
//        virtual void backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

    private:
//        virtual void toGPU(const cl_context& _context, const cl_device_id& _deviceId) override;

        Tensor indices;
};

}
