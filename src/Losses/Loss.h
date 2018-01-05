#pragma once

#include "../clWrapper.h"
#include "../Utility/Tensor.h"

namespace rna
{

class Loss
{
    public:
        Loss();
        virtual ~Loss();

        virtual Tensor::value_type getLoss(const Tensor& _estimation, const Tensor& _target) const = 0;

        virtual const Tensor& getGradient(const Tensor& _estimation, const Tensor& _target) = 0;
        virtual const Tensor& getGradientCL(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) = 0;

        virtual void openCL(cl::Context& _context) = 0;
        virtual void releaseCL();

    protected:
        Tensor gradient;

        cl::Kernel lossKernel, gradientKernel;
};

}
