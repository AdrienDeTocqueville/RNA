#pragma once

#include "../clWrapper.h"

namespace rna
{

class LossFunction
{
    friend class Network;

    public:
        LossFunction();
        virtual ~LossFunction();

        virtual Tensor::value_type getLoss(const Tensor& _estimation, const Tensor& _target) const = 0;

        virtual Tensor getGradient(const Tensor& _estimation, const Tensor& _target) const = 0;
        virtual void   getGradientGPU(const cl_command_queue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) const = 0;

    protected:
        virtual void openCL(cl::ContextWrapper& _context) = 0;
        virtual void releaseCL();

        cl::KernelWrapper lossKernel, gradientKernel;
};

}