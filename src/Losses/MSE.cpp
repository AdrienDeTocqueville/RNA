#include "MSE.h"

namespace rna
{

// TODO: Finish this class

void MSE::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/losses.cl");

    lossKernel.create(p, "lossMSE");
    gradientKernel.create(p, "gradientMSE");
}

Tensor::value_type MSE::getLoss(const Tensor& _estimation, const Tensor& _target) const
{
    if (_estimation.nDimensions() == 1)
        return (_estimation - _target).length2();

    else // doesn't work
        return (_estimation - _target).length2();
}

const Tensor& MSE::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    gradient = Tensor::value_type(2.0) * (_estimation - _target);

    return gradient;
}

const Tensor& MSE::getGradientCL(cl::CommandQueue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch)
{

    return gradient;
}

}
