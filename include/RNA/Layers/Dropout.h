#pragma once

#include "Layer.h"

namespace rna
{

class Dropout: public Layer
{
    public:
        Dropout(Tensor::value_type _rate = 0.5);
        Dropout(std::ifstream& _file);

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);

        virtual void feedForward(cl::CommandQueue&, const Tensor& _inputBatch);
        virtual void backprop(cl::CommandQueue&, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        #else
        virtual void feedForward(const Tensor& _input);
        virtual void backprop(const Tensor& _input, const Tensor& _outputGrad);
        #endif // USE_OPENCL


        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        Tensor::value_type rate;
        Tensor rands;
};

}
