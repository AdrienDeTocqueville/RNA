#pragma once

#include "Optimizer.h"

namespace rna
{

class RMSProp: public Optimizer
{
    public:
        RMSProp(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _rho = 0.9, Tensor::value_type _learningRateDecay = 0.0, Tensor::value_type _delta = 10e-6):
            Optimizer(_params, _paramsGrad),
            learningRate(_learningRate), learningRateDecay(_learningRateDecay),
            rho(_rho), delta(_delta)
        {
            r.reserve(params->size());

            for (size_t i(0); i < params->size(); i++)
                r.emplace_back((*params)[i]->size(), 0.0);
        }

        void updateParams()
        {
//            learningRate *= (1.0 / (1.0 + learningRateDecay * iteration));
//            updateKernel.setArg(3, learningRate);

            for (size_t i(0); i < params->size(); i++)
            {
                updateKernel.setArg(0, *(*params)[i]);
                updateKernel.setArg(1, *(*paramsGrad)[i]);
                updateKernel.setArg(2, r[i]);

                commandQueue.enqueueKernel(updateKernel, { r[i].nElements() });
            }

            commandQueue.join();
        }

        void updateParams(cl::CommandQueue& _commandQueue)
        {
//            learningRate *= (1.0 / (1.0 + learningRateDecay * iteration));
//            updateKernel.setArg(3, learningRate);

            for (size_t i(0); i < params->size(); i++)
            {
                updateKernel.setArg(0, *(*params)[i]);
                updateKernel.setArg(1, *(*paramsGrad)[i]);
                updateKernel.setArg(2, r[i]);

                _commandQueue.enqueueKernel(updateKernel, { r[i].nElements() });
            }
        }

        void openCL(cl::Context& _context)
        {
            commandQueue.create(_context, false);

            auto& p = _context.getProgram("Kernels/rmsprop.cl");
            updateKernel.create(p, "updateParam");

            updateKernel.setArg(3, learningRate);
            updateKernel.setArg(4, rho);
            updateKernel.setArg(5, delta);

            for (size_t i(0); i < r.size(); i++)
                r[i].openCL(_context);
        }

    protected:
        Tensor::value_type learningRate, learningRateDecay, rho, delta;

        std::vector<Tensor> r;

        cl::CommandQueue commandQueue;
        cl::Kernel updateKernel;
};

}
