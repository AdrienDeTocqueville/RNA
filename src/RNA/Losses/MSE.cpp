#include "RNA/Losses/MSE.h"

namespace rna
{

Tensor::value_type MSE::getLoss(const Tensor& _estimation, const Tensor& _target) const
{
    return (_estimation - _target).length2();
}

#ifdef USE_OPENCL
void MSE::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/losses.cl");

    gradientKernel.create(p, "gradientMSE");
}

const Tensor& MSE::getGradient(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch)
{
    gradient.resize(_estimationBatch.size());
    gradient.openCL(_commandQueue.getContext());

    _targetBatch.openCL(_commandQueue.getContext());

    gradientKernel.setArg(0, gradient);
    gradientKernel.setArg(1,_estimationBatch);
    gradientKernel.setArg(2,_targetBatch);

    _commandQueue.enqueueKernel(gradientKernel, _estimationBatch.size());

    return gradient;
}

#else
const Tensor& MSE::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    gradient = Tensor::value_type(2.0) * (_estimation - _target);

    return gradient;
}
#endif // USE_OPENCL

}
