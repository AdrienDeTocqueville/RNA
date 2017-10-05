#include "Linear.h"

#include <fstream>
#include <cmath>

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
