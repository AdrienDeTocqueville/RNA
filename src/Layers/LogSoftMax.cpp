#include "LogSoftMax.h"

#include <cmath>

namespace rna
{

void LogSoftMax::openCL(cl::ContextWrapper& _context)
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

void LogSoftMax::feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resizeAs(_inputBatch);
    output.openCL(context);

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2,_inputBatch.size(1));

    forwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
	output.readBuffer(_commandQueue);
}

void LogSoftMax::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    if (_input.nDimensions() == 1)
    {
        Tensor::value_type sum = 0.0;
        for (unsigned i(0) ; i < _gradOutput.nElements() ; i++)
            sum += _gradOutput[i];

        for (unsigned i(0) ; i < gradInput.nElements() ; i++)
            gradInput[i] = _gradOutput[i] - exp(output[i])*sum;
    }
    else if (_input.nDimensions() == 2)
    {
        for (unsigned i(0); i < _gradOutput.size(0); i++)
        {
            Tensor::value_type sum = 0.0;
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                sum += _gradOutput(i, j);

            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                gradInput(i, j) = _gradOutput(i, j) - exp(output(i, j))*sum;
        }
    }
}

void LogSoftMax::backpropCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    gradInput.resizeAs(_inputBatch);
    gradInput.openCL(context);

    backwardKernel.setArg(0, gradInput);
    backwardKernel.setArg(1,_inputBatch);
    backwardKernel.setArg(2,_gradOutputBatch);
    backwardKernel.setArg(3, output);
    backwardKernel.setArg(4,_gradOutputBatch.size(1));

    backwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
	gradInput.readBuffer(_commandQueue);
}

}
