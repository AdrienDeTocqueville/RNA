#pragma once

#include "Loss.h"

namespace rna
{

class NLL: public Loss
{
    friend class Network;

    public:
        virtual Tensor::value_type getLoss(const Tensor& _estimation, const Tensor& _target) const;

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);

        virtual const Tensor& getGradient(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch);
        #else
        virtual const Tensor& getGradient(const Tensor& _estimation, const Tensor& _target);
        #endif // USE_OPENCL
};

}
