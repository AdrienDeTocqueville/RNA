#pragma once

#include "Loss.h"

namespace rna
{

class MSE: public Loss
{
    friend class Network;

    public:
        virtual Tensor::value_type getLoss(const Tensor& _estimation, const Tensor& _target) const;

        virtual const Tensor& getGradient(const Tensor& _estimation, const Tensor& _target) override;
        virtual const Tensor& getGradientCL(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) override;

        virtual void openCL(cl::Context& _context) override;
};

}
