#include "MSE.h"

namespace rna
{

// TODO: Finish this class

void MSE::openCL(cl::ContextWrapper& _context)
{
    auto& p = _context.getProgram("res/OpenCL/losses.cl");

    lossKernel.create(p, "mseLoss");
    gradientKernel.create(p, "mseGradient");
}

Tensor::value_type MSE::getLoss(const Tensor& _estimation, const Tensor& _target) const
{
    if (_estimation.nDimensions() == 1)
        return (_estimation - _target).length2();

    else // doesn't work
        return (_estimation - _target).length2();
}

Tensor MSE::getGradient(const Tensor& _estimation, const Tensor& _target) const
{
    return Tensor::value_type(2.0) * (_estimation - _target);
}

void MSE::getGradientGPU(const cl_command_queue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) const
{
}

}
