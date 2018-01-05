#pragma once

#include "Losses/LossFunction.h"

namespace rna
{

// TODO: add rectifier
template <typename L>
class Optimizer
{
    public:
        virtual void updateParams();

    protected:
};

}
