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
        virtual void feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        coords_t outputSize;
        bool useMinibatch;
};

}
