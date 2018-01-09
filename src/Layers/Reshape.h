#pragma once

#include "Layer.h"

namespace rna
{

class Reshape: public Layer
{
    public:
        Reshape(coords_t _dimensions = {}, bool _useMinibatch = false);
        Reshape(std::ifstream& _file);

        void setBatchMode(bool _useMinibatch);

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _outputGrad);
        virtual void backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        virtual void openCL(cl::Context& _context) override;

        coords_t outputSize;
        bool useMinibatch;
};

}
