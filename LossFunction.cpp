#include "LossFunction.h"
#include "Network.h"

namespace rna
{

double MSE::getLoss(const Tensor& _estimation, const Tensor& _target)
{
    if (_estimation.nDimensions() == 1)
        return (_estimation - _target).length2();

    else
        return (_estimation - _target).length2();
}

Tensor MSE::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    return 2.0 * (_estimation - _target);
}

double NLL::getLoss(const Tensor& _estimation, const Tensor& _target)
{
    if (_estimation.nDimensions() == 1)
        return - _estimation(_target(0));

    else
    {
        double loss = 0.0;
        for (unsigned i(0) ; i < _estimation.size(0) ; i++)
            loss -= _estimation(i, _target(i, 0));

        return loss;
    }
}

Tensor NLL::getGradient(const Tensor& _estimation, const Tensor& _target)
{
    if (_estimation.nDimensions() == 1)
    {
        Tensor output(_estimation.size(), 0.0);
        output(_target(0)) = -1.0;

        return output;
    }
    else
    {
        Tensor output(_estimation.size(), 0.0);

        for (unsigned i(0) ; i < _estimation.size(0) ; i++)
            output(i, _target(i, 0)) = -1.0;

        return output;
    }
}

}
