#include "LogSoftMax.h"

#include <cmath>

namespace rna
{

const Tensor& LogSoftMax::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

    if (_input.nDimensions() == 1)
    {
        double logSum = 0.0;
        double maxInput = _input.max();

        for (unsigned i(0) ; i < _input.nElements() ; i++)
            logSum += exp(_input[i] - maxInput);

        logSum = maxInput + log(logSum);

        for (unsigned i(0) ; i < output.nElements() ; i++)
            output[i] = _input[i] - logSum;
    }
    else if (_input.nDimensions() == 2)
    {
        for (unsigned i(0); i < _input.size(0); i++)
        {
            double logSum = 0.0;
            double maxInput = _input(i, 0);

            for (unsigned j(1); j < _input.size(1); j++)
                maxInput = std::max(_input(i, j), maxInput);

            for (unsigned j(0); j < _input.size(1); j++)
                logSum += exp(_input(i, j) - maxInput);

            logSum = maxInput + log(logSum);

            for (unsigned j(0); j < _input.size(1); j++)
                output(i, j) = _input(i, j) - logSum;
        }
    }

    return output;
}

const Tensor& LogSoftMax::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    if (_input.nDimensions() == 1)
    {
        double sum = 0.0;
        for (unsigned i(0) ; i < _gradOutput.nElements() ; i++)
            sum += _gradOutput[i];

        for (unsigned i(0) ; i < gradInput.nElements() ; i++)
            gradInput[i] = _gradOutput[i] - exp(output[i])*sum;
    }
    else if (_input.nDimensions() == 2)
    {
        for (unsigned i(0); i < _gradOutput.size(0); i++)
        {
            double sum = 0.0;
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                sum += _gradOutput(i, j);

            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                gradInput(i, j) = _gradOutput(i, j) - exp(output(i, j))*sum;
        }
    }

    return gradInput;
}

}
