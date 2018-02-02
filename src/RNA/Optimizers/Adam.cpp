#include "RNA/Optimizers/Adam.h"

#include <cmath>

namespace rna
{

Adam::Adam(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _rho1, Tensor::value_type _rho2, Tensor::value_type _learningRateDecay, Tensor::value_type _delta):
    Optimizer(_params, _paramsGrad),
    learningRate(_learningRate), learningRateDecay(_learningRateDecay),
    rho1(_rho1), rho2(_rho2), delta(_delta)
{
    s.reserve(params->size());
    r.reserve(params->size());

    for (size_t i(0); i < params->size(); i++)
    {
        s.emplace_back((*params)[i]->size(), 0.0);
        r.emplace_back((*params)[i]->size(), 0.0);
    }
}

#ifdef USE_OPENCL
void Adam::updateParams(cl::CommandQueue& _commandQueue)
{ // TODO: implement Adam on GPU
}

void Adam::openCL(cl::Context& _context)
{
    Optimizer::openCL(_context);

    auto& p = _context.getProgram("Kernels/Adam.cl");
    updateKernel.create(p, "updateParam");

    updateKernel.setArg(3, learningRate);
    updateKernel.setArg(4, rho1);
    updateKernel.setArg(5, rho2);
    updateKernel.setArg(6, delta);

    for (size_t i(0); i < r.size(); i++)
        r[i].openCL(_context);
}

#else
void Adam::updateParams()
{
    iteration++;

    for (size_t i(0); i < params->size(); i++)
    {
        Tensor& param = *(*params)[i];
        Tensor& paramGrad = *(*paramsGrad)[i];

        for (unsigned j(0); j < param.nElements(); j++)
        {
            s[i][j] = rho1 * s[i][j] + (1.0f-rho1) * paramGrad[j];
            r[i][j] = rho2 * r[i][j] + (1.0f-rho2) * paramGrad[j]*paramGrad[j];

            Tensor::value_type biasCorrectedS = s[i][j] / (1.0f - pow(rho1, iteration));
            Tensor::value_type biasCorrectedR = r[i][j] / (1.0f - pow(rho2, iteration));

            Tensor::value_type paramDelta = -(learningRate * biasCorrectedS) / (delta + sqrt(biasCorrectedR));

            param[j] += paramDelta;

            paramGrad[j] = 0.0f;
        }
    }
}
#endif // USE_OPENCL

}