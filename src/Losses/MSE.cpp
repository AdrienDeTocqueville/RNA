#include "MSE.h"

namespace rna
{

void MSE::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/losses.cl");

    gradientKernel.create(p, "gradientMSE");
}

Tensor::value_type MSE::getLoss(const Tensor& _estimation, const Tensor& _target) const
{
    return (_estimation - _target).length2();
}

const Tensor& MSE::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    gradient = Tensor::value_type(2.0) * (_estimation - _target);

    return gradient;
}

const Tensor& MSE::getGradientCL(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch)
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

}
