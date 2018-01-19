#pragma once

#include "Layer.h"

namespace rna
{

class MaxPooling: public Layer
{
    public:
        MaxPooling(size_t _poolWidth = 2, size_t _poolHeight = 2);
        MaxPooling(std::ifstream& _file);

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
        size_t poolWidth, poolHeight;

        Tensor indices; // TODO: indices should be array of size_t
};

}
