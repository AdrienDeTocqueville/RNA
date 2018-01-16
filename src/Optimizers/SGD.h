#pragma once

#include "Optimizer.h"

namespace rna
{

class SGD: public Optimizer
{
    public:
        SGD(Tensor::value_type _learningRate, Tensor::value_type _inertia = 0.0f):
            learningRate(_learningRate), inertia(_inertia)
        { }

        void init(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
            paramsDelta.clear();
            paramsDelta.reserve(_params.size());

            for (size_t i(0); i < _params.size(); i++)
                paramsDelta.emplace_back(_params[i]->size(), 0.0);
        }

        void updateParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
            for (size_t i(0); i < _params.size(); i++)
            {
                updateKernel.setArg(0, *_params[i]);
                updateKernel.setArg(1, *_paramsGrad[i]);
                updateKernel.setArg(2, paramsDelta[i]);

                commandQueue.enqueue(updateKernel, { paramsDelta[i].nElements() });
            }

            commandQueue.join();
        }

        void openCL(cl::Context& _context)
        {
            commandQueue.create(_context, false);

            auto& p = _context.getProgram("res/OpenCL/sgd.cl");
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
