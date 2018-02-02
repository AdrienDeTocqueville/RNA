#pragma once

#include "Optimizer.h"

namespace rna
{

class Adam: public Optimizer
{
    public:
        Adam(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _rho1 = 0.9, Tensor::value_type _rho2 = 0.999, Tensor::value_type _learningRateDecay = 0.0, Tensor::value_type _delta = 10e-8);

        #ifdef USE_OPENCL
        void openCL(cl::Context& _context);

    protected:
        void updateParams(cl::CommandQueue& _commandQueue);
        #else
    protected:
        void updateParams();
        #endif // USE_OPENCL

        Tensor::value_type learningRate, learningRateDecay, rho1, rho2, delta;

        std::vector<Tensor> s, r;
};

}
