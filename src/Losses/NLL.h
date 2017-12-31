#pragma once

#include "LossFunction.h"

namespace rna
{

class NLL: public LossFunction
{
    public:
        virtual Tensor::value_type getLoss(const Tensor& _estimation, const Tensor& _target) const;

        virtual Tensor getGradient(const Tensor& _estimation, const Tensor& _target) const;
        virtual void   getGradientGPU(const cl_command_queue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) const;

    private:
        virtual void openCL(cl::ContextWrapper& _context);
};

}
