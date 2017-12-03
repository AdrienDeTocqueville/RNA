#include "Layer.h"

#include <cmath>


double sigmoid(double _x)
{
    return 1.0 / ( 1.0 + exp(-_x) );
}

double dSigmoid(double _x)
{
    double s = sigmoid(_x);
    return s*(1.0 - s);
}

double dtanh(double _x)
{
    float t = tanh(_x);
    return 1.0 - t*t;
}

namespace rna
{

Tensor Tanh::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < output.nElements() ; i++)
        output[i] = tanh(_input[i]);

    return output;
}

Tensor Tanh::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    for (unsigned i(0) ; i < gradInput.nElements() ; i++)
        gradInput[i] = dtanh(_input[i]) * _gradOutput[i];

    return gradInput;
}

Tensor ReLU::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < output.nElements() ; i++)
		output[i] = std::max(_input[i], 0.0);

    return output;
}

Tensor ReLU::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    for (unsigned i(0) ; i < gradInput.nElements() ; i++)
	{
		if (_input[i] < 0.0)
			gradInput[i] = 0.0;
		else
			gradInput[i] =  _gradOutput[i];
	}

    return gradInput;
}

}
