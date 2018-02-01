#pragma once

#include "Utility/Tensor.h"

namespace rna
{

class Optimizer
{
    public:
        Optimizer(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad):
            params(&_params), paramsGrad(&_paramsGrad), iteration(0)
        { }

        virtual ~Optimizer() {}

        #ifdef USE_OPENCL
        virtual void updateParams(cl::CommandQueue& _commandQueue) = 0;
        virtual void openCL(cl::Context& _context) = 0;
        #else
        virtual void updateParams() = 0;
        #endif // USE_OPENCL

    protected:
        std::vector<Tensor*>* params;
        std::vector<Tensor*>* paramsGrad;

        #ifdef USE_OPENCL
        cl::Kernel updateKernel;
        #endif // USE_OPENCL

        int iteration; // TODO: auto increment ?
};

}
