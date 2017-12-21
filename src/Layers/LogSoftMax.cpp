#include "LogSoftMax.h"

#include <cmath>

namespace rna
{

void LogSoftMax::openCL(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!kernelForward)
        kernelForward = loadKernel(_context, _deviceId, "src/OpenCL/logSoftMax.cl", "logSoftMaxForward");

    if (!kernelBackward)
        kernelBackward = loadKernel(_context, _deviceId, "src/OpenCL/logSoftMax.cl", "logSoftMaxBackward");
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

    cl_int inputWidth = _inputBatch.size(1);

    clSetKernelArg(kernelForward, 0, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelForward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());
    clSetKernelArg(kernelForward, 2, sizeof(cl_int), &inputWidth);

    execKernel(_commandQueue, kernelForward, {_inputBatch.size(0)});
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

    cl_int gradOutputWidth = _gradOutputBatch.size(1);

    clSetKernelArg(kernelBackward, 0, sizeof(cl_mem), &gradInput.getBuffer());
    clSetKernelArg(kernelBackward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());
    clSetKernelArg(kernelBackward, 2, sizeof(cl_mem), &_gradOutputBatch.getBuffer());
    clSetKernelArg(kernelBackward, 3, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelBackward, 4, sizeof(cl_int), &gradOutputWidth);

    execKernel(_commandQueue, kernelBackward, { _inputBatch.size(0) });
	gradInput.readBuffer(_commandQueue);
}

}
