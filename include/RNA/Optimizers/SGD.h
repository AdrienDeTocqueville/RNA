#pragma once

#include "Optimizer.h"

namespace rna
{

class SGD: public Optimizer
{
    public:
        SGD(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad, Tensor::value_type _learningRate, Tensor::value_type _inertia = 0.0f):
            Optimizer(_params, _paramsGrad),
            learningRate(_learningRate), inertia(_inertia)
        {
            paramsDelta.reserve(params->size());

            for (size_t i(0); i < params->size(); i++)
                paramsDelta.emplace_back((*params)[i]->size(), 0.0);
        }

        void updateParams()
        {
            for (size_t i(0); i < params->size(); i++)
            {
                updateKernel.setArg(0, *(*params)[i]);
                updateKernel.setArg(1, *(*paramsGrad)[i]);
                updateKernel.setArg(2, paramsDelta[i]);

                commandQueue.enqueueKernel(updateKernel, { paramsDelta[i].nElements() });
            }

            commandQueue.join();
        }

        void updateParams(cl::CommandQueue& _commandQueue)
        {
            for (size_t i(0); i < params->size(); i++)
            {
                updateKernel.setArg(0, *(*params)[i]);
                updateKernel.setArg(1, *(*paramsGrad)[i]);
                updateKernel.setArg(2, paramsDelta[i]);

                _commandQueue.enqueueKernel(updateKernel, { paramsDelta[i].nElements() });
            }
        }

        void openCL(cl::Context& _context)
        {
            commandQueue.create(_context, false);

            auto& p = _context.getProgram("Kernels/sgd.cl");
            updateKernel.create(p, "updateParam");

            updateKernel.setArg(3, learningRate);
            updateKernel.setArg(4, inertia);

            for (size_t i(0); i < paramsDelta.size(); i++)
                paramsDelta[i].openCL(_context);
        }

    protected:
        Tensor::value_type learningRate, inertia;

        std::vector<Tensor> paramsDelta;

        cl::CommandQueue commandQueue;
        cl::Kernel updateKernel;
};

}
