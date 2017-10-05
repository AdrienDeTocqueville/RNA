#include "LossFunction.h"

namespace rna
{

double MSE::getLoss(const Tensor& _estimation, const Tensor& _target)
{
    return 0.5 * (_estimation - _target).length2();
}

Tensor MSE::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    return _estimation - _target;
}


double NLL::getLoss(const Tensor& _estimation, const Tensor& _target)
{
    return - _estimation(_target(0));
}

Tensor NLL::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    Tensor output(_estimation.size(), 0.0);
    output(_target(0)) = -1.0;

    return output;
}

};
