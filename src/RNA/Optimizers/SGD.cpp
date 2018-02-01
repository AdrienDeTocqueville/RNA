#include "RNA/Optimizers/SGD.h"

#include <cmath>

namespace rna
{

SGD::SGD(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _inertia):
    Optimizer(_params, _paramsGrad),
    learningRate(_learningRate), inertia(_inertia)
{
    paramsDelta.reserve(params->size());

    for (size_t i(0); i < params->size(); i++)
        paramsDelta.emplace_back((*params)[i]->size(), 0.0);
}


#ifdef USE_OPENCL
void SGD::updateParams(cl::CommandQueue& _commandQueue)
{
    for (size_t i(0); i < params->size(); i++)
    {
        updateKernel.setArg(0, *(*params)[i]);
        updateKernel.setArg(1, *(*paramsGrad)[i]);
        updateKernel.setArg(2, paramsDelta[i]);

        _commandQueue.enqueueKernel(updateKernel, { paramsDelta[i].nElements() });
    }
}

void SGD::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/sgd.cl");
    updateKernel.create(p, "updateParam");

    updateKernel.setArg(3, learningRate);
    updateKernel.setArg(4, inertia);

    for (size_t i(0); i < paramsDelta.size(); i++)
        paramsDelta[i].openCL(_context);
}

#else
void SGD::updateParams()
{
    for (size_t i(0); i < params->size(); i++)
    {
        Tensor& param = *(*params)[i];
        Tensor& paramGrad = *(*paramsGrad)[i];

        paramsDelta[i] = inertia * paramsDelta[i] - 0.01f * learningRate * paramGrad;
        param += paramsDelta[i];

        paramGrad.fill(0.0f);
    }
}
#endif // USE_OPENCL

}