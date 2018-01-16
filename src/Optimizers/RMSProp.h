#pragma once

#include "Optimizer.h"

namespace rna
{

class RMSProp: public Optimizer
{
    public:
        RMSProp(Tensor::value_type _learningRate, Tensor::value_type _rho = 0.9, Tensor::value_type _learningRateDecay = 0.0, Tensor::value_type _delta = 0.0001):
            learningRate(_learningRate), learningRateDecay(_learningRateDecay),
            rho(_rho), delta(_delta),
            iteration(0)
        { }

        void init(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
            r.clear();
            r.reserve(_params.size());

            for (size_t i(0); i < _params.size(); i++)
                r.emplace_back(_params[i]->size(), 0.0);
        }

        void updateParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
//            learningRate *= (1.0 / (1.0 + learningRateDecay * iteration++));
//            updateKernel.setArg(3, learningRate);

            for (size_t i(0); i < _params.size(); i++)
            {
                updateKernel.setArg(0, *_params[i]);
                updateKernel.setArg(1, *_paramsGrad[i]);
                updateKernel.setArg(2, r[i]);

                commandQueue.enqueue(updateKernel, { r[i].nElements() });
            }

            commandQueue.join();
        }

        void openCL(cl::Context& _context)
        {
            commandQueue.create(_context, false);

            auto& p = _context.getProgram("res/OpenCL/rmsprop.cl");
            updateKernel.create(p, "updateParam");

            updateKernel.setArg(3, learningRate);
            updateKernel.setArg(4, rho);
            updateKernel.setArg(5, delta);

            for (size_t i(0); i < r.size(); i++)
                r[i].openCL(_context);
        }

    protected:
        Tensor::value_type learningRate, learningRateDecay, rho, delta;
        int iteration;

        std::vector<Tensor> r;

        cl::CommandQueue commandQueue;
        cl::Kernel updateKernel;
};

}
