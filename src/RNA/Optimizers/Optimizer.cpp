#include "RNA/Optimizers/SGD.h"

#include <cmath>

namespace rna
{

Optimizer::Optimizer(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad):
    params(&_params), paramsGrad(&_paramsGrad), iteration(0)
{ }

#ifdef USE_OPENCL
void Optimizer::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/maths.cl");
    averageKernel.create(p, "mul");
}

void Optimizer::updateParams(cl::CommandQueue& _commandQueue, size_t _batchSize)
{
//    Tensor::value_type averageFactor = 1.0 / _batchSize;
//
//    for (Tensor* paramGrad: *paramsGrad)
//    {
//        averageKernel.setArg(0, *paramGrad);
//        averageKernel.setArg(1, averageFactor);
//        averageKernel.setArg(2, paramGrad->getStride(0));
//
//        _commandQueue.enqueueKernel(averageKernel, {paramGrad->size(0)});
//    }
//
//    _commandQueue.enqueueBarrier();

    updateParams(_commandQueue);
}

#else
void Optimizer::updateParams(size_t _batchSize)
{
    Tensor::value_type averageFactor = 1.0 / _batchSize;

    for (Tensor* paramGrad: *paramsGrad)
        *paramGrad *= averageFactor;

    updateParams();
}
#endif // USE_OPENCL

}