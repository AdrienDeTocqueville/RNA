#pragma once

#include "../Utility/Tensor.h"

namespace rna
{

class Optimizer
{
    public:
        Optimizer() { }
        virtual ~Optimizer() {}

        virtual void init(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) = 0;
        virtual void updateParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) = 0;

        virtual void openCL(cl::Context& _context) = 0;
};

}
