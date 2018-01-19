#pragma once

#include "Utility/Tensor.h"

namespace rna
{

class Optimizer
{
    public:
        Optimizer(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad):
            params(&_params), paramsGrad(&_paramsGrad), iteration(0)
        { }

        virtual ~Optimizer() {}

        virtual void updateParams() = 0;
        virtual void updateParams(cl::CommandQueue& _commandQueue) = 0;
        virtual void openCL(cl::Context& _context) = 0;

    protected:
        std::vector<Tensor*>* params;
        std::vector<Tensor*>* paramsGrad;

        int iteration; // TODO: actually increment it
};

}
