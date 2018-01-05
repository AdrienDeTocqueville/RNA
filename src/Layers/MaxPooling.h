#pragma once

#include "Layer.h"

namespace rna
{

class MaxPooling: public Layer
{
    public:
        MaxPooling(): Layer("MaxPooling") {}

        virtual void feedForwardCPU(const Tensor& _input); // _input size must be divisible by 2
        virtual void feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

    private:
        virtual void openCL(cl::Context& _context) override;

        Tensor indices; // TODO: indices should be array of size_t
};

}
