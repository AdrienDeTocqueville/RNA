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
        coords_t outputSize;
        bool useMinibatch;
};

}
