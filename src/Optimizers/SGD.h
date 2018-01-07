#pragma once

#include "Optimizer.h"

namespace rna
{

class SGD: public Optimizer
{
    public:
        SGD(Tensor::value_type _learningRate, Tensor::value_type _inertia):
            learningRate(_learningRate), inertia(_inertia)
        { }

        void init(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
            paramsDelta.clear();
            paramsDelta.reserve(_params.size());

            for (size_t i(0); i < paramsDelta.size(); i++)
                paramsDelta.emplace_back(_params[i]->size(), 0.0);
        }

        void updateParams(cl::CommandQueue& _commandQueue, std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
            for (size_t i(0); i < paramsDelta.size(); i++)
            {
                updater.setArg(0, *_params[i]);
                updater.setArg(1, *_paramsGrad[i]);
                updater.setArg(2, paramsDelta[i]);

                _commandQueue.enqueue(updater, { paramsDelta[i].nElements() });
            }
        }

        void openCL(cl::Context& _context)
        {
            auto& p = _context.getProgram("res/OpenCL/sgd.cl");
            updater.create(p, "updateParam");

            updater.setArg(3, learningRate);
            updater.setArg(4, inertia);

            for (size_t i(0); i < paramsDelta.size(); i++)
                paramsDelta[i].openCL(_context);
        }

    protected:
        Tensor::value_type learningRate, inertia;

        std::vector<Tensor> paramsDelta;

        cl::Kernel updater;
};

}
