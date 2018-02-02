#pragma once

#include "Utility/Tensor.h"

namespace rna
{

class Optimizer
{
    public:
        Optimizer(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad);
        virtual ~Optimizer() {}

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);

        void updateParams(cl::CommandQueue& _commandQueue, size_t _batchSize);
        #else
        void updateParams(size_t _batchSize);
        #endif // USE_OPENCL

    protected:
        std::vector<Tensor*>* params;
        std::vector<Tensor*>* paramsGrad;

        #ifdef USE_OPENCL
        cl::Kernel averageKernel, updateKernel;

        virtual void updateParams(cl::CommandQueue& _commandQueue) = 0;
        #else
        virtual void updateParams() = 0;
        #endif // USE_OPENCL

        int iteration; // TODO: auto increment ?
};

}
