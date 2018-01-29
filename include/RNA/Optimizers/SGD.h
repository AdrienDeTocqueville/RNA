#pragma once

#include "Optimizer.h"

namespace rna
{

class SGD: public Optimizer
{
    public:
        SGD(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _inertia = 0.0f);

        #ifdef USE_OPENCL
        void updateParams(cl::CommandQueue& _commandQueue);
        void openCL(cl::Context& _context);
        #else
        void updateParams();
        #endif // USE_OPENCL

    protected:
        Tensor::value_type learningRate, inertia;

        std::vector<Tensor> paramsDelta;
};

}
