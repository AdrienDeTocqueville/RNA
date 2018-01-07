#pragma once

#include "Optimizer.h"

namespace rna
{

class RMSProp: public Optimizer
{
    public:
        RMSProp(Tensor::value_type _learningRate, Tensor::value_type _rho, Tensor::value_type _rateDecay):
            learningRate(_learningRate), rho(_rho), rateDecay(_rateDecay), epsilon(0.01)
        { }

        void init(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
        }

        void updateParams(cl::CommandQueue& _commandQueue, std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
        {
        }

        void openCL(cl::Context& _context)
        {
            auto& p = _context.getProgram("res/OpenCL/rmsprop.cl");
            updater.create(p, "updateParam");
        }

    protected:
        Tensor::value_type learningRate, rho, rateDecay, epsilon;

        cl::Kernel updater;
};

}
