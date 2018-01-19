#pragma once

#include "Utility/clWrapper.h"
#include "Utility/Tensor.h"

namespace rna
{

class Loss
{
    public:
        Loss();
        virtual ~Loss();

        virtual Tensor::value_type getLoss(const Tensor& _estimation, const Tensor& _target) const = 0;

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context) = 0;
        virtual void releaseCL();

        virtual const Tensor& getGradient(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) = 0;
        #else
        virtual const Tensor& getGradient(const Tensor& _estimation, const Tensor& _target) = 0;
        #endif // USE_OPENCL

    protected:
        Tensor gradient;

        #ifdef USE_OPENCL
        cl::Kernel gradientKernel;
        #endif // USE_OPENCL
};

}
