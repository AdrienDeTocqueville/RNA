#pragma once

#include "Optimizer.h"

namespace rna
{

class RMSProp: public Optimizer
{
    public:
        RMSProp(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _rho = 0.9, Tensor::value_type _learningRateDecay = 0.0, Tensor::value_type _delta = 10e-6);

        #ifdef USE_OPENCL
        void openCL(cl::Context& _context);

    protected:
        void updateParams(cl::CommandQueue& _commandQueue);
        #else
    protected:
        void updateParams();
        #endif // USE_OPENCL

        Tensor::value_type learningRate, learningRateDecay, rho, delta;

        std::vector<Tensor> r;
};

}
