#include "RNA/Losses/NLL.h"

namespace rna
{

Tensor::value_type NLL::getLoss(const Tensor& _estimation, const Tensor& _target) const
{
    if (_estimation.nDimensions() == 1)
        return - _estimation(_target(0));

    else
    {
        Tensor::value_type loss = 0.0;
        for (unsigned i(0) ; i < _estimation.size(0) ; i++)
            loss -= _estimation(i, _target(i, 0));

        return loss;
    }
}

#ifdef USE_OPENCL
void NLL::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/losses.cl");

    gradientKernel.create(p, "gradientNLL");
}

const Tensor& NLL::getGradient(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch)
{
    gradient.resize(_estimationBatch.size());
    gradient.openCL(_commandQueue.getContext());

    _targetBatch.openCL(_commandQueue.getContext());

    gradientKernel.setArg(0, gradient);
    gradientKernel.setArg(1,_targetBatch);
    gradientKernel.setArg(2,_estimationBatch.size(1));

    _commandQueue.enqueueKernel(gradientKernel, { _estimationBatch.size(0) });

    return gradient;
}

#else
const Tensor& NLL::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    gradient.resize(_estimation.size());
    gradient.fill(0.0);

    if (_estimation.nDimensions() == 1)
        gradient(_target(0)) = -1.0;

    else
    {
        for (unsigned i(0) ; i < _estimation.size(0) ; i++)
            gradient(i, _target(i, 0)) = -1.0;
    }

    return gradient;
}
#endif // USE_OPENCL

}
