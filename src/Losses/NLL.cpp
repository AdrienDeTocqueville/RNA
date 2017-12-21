#include "NLL.h"

namespace rna
{

// TODO: Finish this class

void NLL::openCL(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!lossKernel)
        lossKernel = loadKernel(_context, _deviceId, "src/OpenCL/losses.cl", "mseGradient");

    if (!gradientKernel)
        gradientKernel = loadKernel(_context, _deviceId, "src/OpenCL/losses.cl", "mseGradient");
}

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

Tensor NLL::getGradient(const Tensor& _estimation, const Tensor& _target) const
{
    if (_estimation.nDimensions() == 1)
    {
        Tensor output(_estimation.size(), 0.0);
        output(_target(0)) = -1.0;

        return output;
    }
    else
    {
        Tensor output(_estimation.size(), 0.0);

        for (unsigned i(0) ; i < _estimation.size(0) ; i++)
            output(i, _target(i, 0)) = -1.0;

        return output;
    }
}

void NLL::getGradientGPU(const cl_command_queue& _commandQueue, const Tensor& _estimationBatch, const Tensor& _targetBatch) const
{
}

}
