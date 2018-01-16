#pragma once

#include "Layer.h"

namespace rna
{

class MaxPooling: public Layer
{
    public:
        MaxPooling(size_t _poolWidth = 2, size_t _poolHeight = 2);
        MaxPooling(std::ifstream& _file);

        virtual void feedForwardCPU(const Tensor& _input); // NOTE: _input size must be divisible by 2
        virtual void feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _outputGrad);
        virtual void backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        virtual void openCL(cl::Context& _context) override;

        size_t poolWidth, poolHeight;
        Tensor indices; // TODO: indices should be array of size_t
};

}
