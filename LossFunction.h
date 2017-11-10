#pragma once

#include "Tensor.h"

namespace rna
{

struct LossFunction
{
    virtual ~LossFunction() {}

    virtual double getLoss(const Tensor& _estimation, const Tensor& _target) = 0;
    virtual Tensor getGradient(const Tensor& _estimation, const Tensor& _target) = 0;
};

struct MSE: public LossFunction
{
    virtual double getLoss(const Tensor& _estimation, const Tensor& _target);
    virtual Tensor getGradient(const Tensor& _estimation, const Tensor& _target);
};

struct NLL: public LossFunction
{
    virtual double getLoss(const Tensor& _estimation, const Tensor& _target);
    virtual Tensor getGradient(const Tensor& _estimation, const Tensor& _target);
};

}
