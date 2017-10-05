#include "LogSoftMax.h"

#include <cmath>

namespace rna
{

Tensor LogSoftMax::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

	double logSum = 0.0;
	double maxInput = _input.max();

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        logSum += exp(_input[i] - maxInput);

	logSum = maxInput + log(logSum);

    for (unsigned i(0) ; i < output.nElements() ; i++)
		output[i] = _input[i] - logSum;

    return output;
}

Tensor LogSoftMax::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    double sum = 0.0;
    for (unsigned i(0) ; i < _gradOutput.nElements() ; i++)
        sum += _gradOutput[i];

    for (unsigned i(0) ; i < gradInput.nElements() ; i++)
        gradInput[i] = _gradOutput[i] - exp(output[i])*sum;

    return gradInput;
}

}
