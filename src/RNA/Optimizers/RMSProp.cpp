#include "RNA/Optimizers/RMSProp.h"

#include <cmath>

namespace rna
{

RMSProp::RMSProp(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _rho, Tensor::value_type _learningRateDecay, Tensor::value_type _delta):
    Optimizer(_params, _paramsGrad),
    learningRate(_learningRate), learningRateDecay(_learningRateDecay),
    rho(_rho), delta(_delta)
{
    r.reserve(params->size());

    for (size_t i(0); i < params->size(); i++)
        r.emplace_back((*params)[i]->size(), 0.0);
}

#ifdef USE_OPENCL
void RMSProp::updateParams(cl::CommandQueue& _commandQueue)
{
//    learningRate *= (1.0 / (1.0 + learningRateDecay * iteration));
//    updateKernel.setArg(3, learningRate);

    for (size_t i(0); i < params->size(); i++)
    {
        updateKernel.setArg(0, *(*params)[i]);
        updateKernel.setArg(1, *(*paramsGrad)[i]);
        updateKernel.setArg(2, r[i]);

        _commandQueue.enqueueKernel(updateKernel, { r[i].nElements() });
    }
}

void RMSProp::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/rmsprop.cl");
    updateKernel.create(p, "updateParam");

    updateKernel.setArg(3, learningRate);
    updateKernel.setArg(4, rho);
    updateKernel.setArg(5, delta);

    for (size_t i(0); i < r.size(); i++)
        r[i].openCL(_context);
}

#else
void RMSProp::updateParams()
{
//    learningRate *= (1.0 / (1.0 + learningRateDecay * iteration));

    for (size_t i(0); i < params->size(); i++)
    {
        Tensor& param = *(*params)[i];
        Tensor& paramGrad = *(*paramsGrad)[i];

        for (unsigned j(0); j < param.nElements(); j++)
        {
            r[i][j] = rho * r[i][j] + (1.0f-rho) * paramGrad[j]*paramGrad[j];

            Tensor::value_type paramDelta = -(learningRate * paramGrad[j]) / sqrt(delta+r[i][j]);
            param[j] += paramDelta;

            paramGrad[j] = 0.0f;
        }
    }
}
#endif // USE_OPENCL

}