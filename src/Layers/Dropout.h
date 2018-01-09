#pragma once

#include "Layer.h"

namespace rna
{

class Dropout: public Layer
{
    public:
        Dropout(Tensor::value_type _rate = 0.5): Layer("Dropout"), rate(_rate) {}
        Dropout(std::ifstream& _file);

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _outputGrad);
        virtual void backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        virtual void openCL(cl::Context& _context) override;

        Tensor::value_type rate;
        Tensor rands;
};

}
