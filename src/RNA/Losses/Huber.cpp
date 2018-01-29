#include "RNA/Losses/Huber.h"

#include <cmath>
#include <iostream>

namespace rna
{

Tensor::value_type Huber::getLoss(const Tensor& _estimation, const Tensor& _target) const
{
    Tensor::value_type loss = 0.0;

    for (unsigned i(0) ; i < _estimation.nElements() ; i++)
    {
        Tensor::value_type x = _estimation[i] - _target[i];

        if ( std::abs(x) < 1.0 )
            loss += 0.5 * x*x;

        else
            loss += std::abs(x) - 0.5;
    }

    return loss;
}

#ifdef USE_OPENCL
void Huber::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/losses.cl");

    gradientKernel.create(p, "gradientHuber");
}

const Tensor& Huber::getGradient(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch)
{
    gradient.resize(_estimationBatch.size());
    gradient.openCL(_commandQueue.getContext());

    _estimationBatch.openCL(_commandQueue.getContext());
    _targetBatch.openCL(_commandQueue.getContext());

    gradientKernel.setArg(0, gradient);
    gradientKernel.setArg(1,_estimationBatch);
    gradientKernel.setArg(2,_targetBatch);

    _commandQueue.enqueueKernel(gradientKernel, _estimationBatch.size());

    return gradient;
}

#else
const Tensor& Huber::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    gradient.resize({_estimation.nElements()});

    for (unsigned i(0) ; i < _estimation.nElements() ; i++)
    {
        Tensor::value_type x = _estimation[i] - _target[i];

        gradient[i] = std::max(-1.0f, std::min(x, 1.0f));
    }

    return gradient;
}
#endif // USE_OPENCL

}
