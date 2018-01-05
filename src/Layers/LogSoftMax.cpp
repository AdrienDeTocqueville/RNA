#include "LogSoftMax.h"

#include <cmath>

namespace rna
{

void LogSoftMax::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/logSoftMax.cl");

    forwardKernel.create(p, "feedForwardLogSoftMax");
    backwardKernel.create(p, "backpropLogSoftMax");
}

void LogSoftMax::feedForwardCPU(const Tensor& _input)
{
    output.resizeAs(_input);

    if (_input.nDimensions() == 1)
    {
        Tensor::value_type logSum = 0.0;
        Tensor::value_type maxInput = _input.max();

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
            Tensor::value_type logSum = 0.0;
            Tensor::value_type maxInput = _input(i, 0);

            for (unsigned j(1); j < _input.size(1); j++)
                maxInput = std::max(_input(i, j), Tensor::value_type(maxInput));

            for (unsigned j(0); j < _input.size(1); j++)
                logSum += exp(_input(i, j) - maxInput);

            logSum = maxInput + log(logSum);

            for (unsigned j(0); j < _input.size(1); j++)
                output(i, j) = _input(i, j) - logSum;
        }
    }
}

void LogSoftMax::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);
    output.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2,_inputBatch.size(1));

    forwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
}

void LogSoftMax::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    inputGrad.resizeAs(_input);

    if (_input.nDimensions() == 1)
    {
        Tensor::value_type sum = 0.0;
        for (unsigned i(0) ; i < _gradOutput.nElements() ; i++)
            sum += _gradOutput[i];

        for (unsigned i(0) ; i < inputGrad.nElements() ; i++)
            inputGrad[i] = _gradOutput[i] - exp(output[i])*sum;
    }
    else if (_input.nDimensions() == 2)
    {
        for (unsigned i(0); i < _gradOutput.size(0); i++)
        {
            Tensor::value_type sum = 0.0;
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                sum += _gradOutput(i, j);

            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                inputGrad(i, j) = _gradOutput(i, j) - exp(output(i, j))*sum;
        }
    }
}

void LogSoftMax::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_inputBatch);
    backwardKernel.setArg(2,_gradOutputBatch);
    backwardKernel.setArg(3, output);
    backwardKernel.setArg(4,_gradOutputBatch.size(1));

    backwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
	inputGrad.readBuffer(_commandQueue);
}

}
