#include "RNA/Layers/LogSoftMax.h"

#include <cmath>

namespace rna
{

#ifdef USE_OPENCL
void LogSoftMax::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/logSoftMax.cl");

    forwardKernel.create(p, "feedForwardLogSoftMax");
    backwardKernel.create(p, "backpropLogSoftMax");
}

void LogSoftMax::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);
    output.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2,_inputBatch.size(1));

    _commandQueue.enqueueKernel(forwardKernel, { _inputBatch.size(0) });
}

void LogSoftMax::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_inputBatch);
    backwardKernel.setArg(2,_outputGradBatch);
    backwardKernel.setArg(3, output);
    backwardKernel.setArg(4,_outputGradBatch.size(1));

    cl_event event;
    _commandQueue.enqueueKernel(backwardKernel, {_inputBatch.size(0)}, &event);
    _commandQueue.enqueueBarrier({event});
}

#else
void LogSoftMax::feedForward(const Tensor& _input)
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

void LogSoftMax::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    inputGrad.resizeAs(_input);

    if (_input.nDimensions() == 1)
    {
        Tensor::value_type sum = 0.0;
        for (unsigned i(0) ; i < _outputGrad.nElements() ; i++)
            sum += _outputGrad[i];

        for (unsigned i(0) ; i < inputGrad.nElements() ; i++)
            inputGrad[i] = _outputGrad[i] - exp(output[i])*sum;
    }
    else if (_input.nDimensions() == 2)
    {
        for (unsigned i(0); i < _outputGrad.size(0); i++)
        {
            Tensor::value_type sum = 0.0;
            for (unsigned j(0) ; j < _outputGrad.size(1) ; j++)
                sum += _outputGrad(i, j);

            for (unsigned j(0) ; j < _outputGrad.size(1) ; j++)
                inputGrad(i, j) = _outputGrad(i, j) - exp(output(i, j))*sum;
        }
    }
}
#endif // USE_OPENCL

}
