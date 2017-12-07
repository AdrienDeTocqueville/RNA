#include "LogSoftMax.h"

#include <cmath>

namespace rna
{

void LogSoftMax::toGPU(cl_context _context, cl_device_id _device)
{
    loadKernel(_context, _device, "OpenCL/logSoftMax.cl", "logSoftMax");
}

const Tensor& LogSoftMax::feedForward(const Tensor& _input)
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

    return output;
}

const Tensor& LogSoftMax::backprop(const Tensor& _input, const Tensor& _gradOutput)
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

    return gradInput;
}

void LogSoftMax::GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);

    cl_context context;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.toGPU(context, CL_MEM_WRITE_ONLY);

    cl_mem outputBuffer = output.getBuffer();
    cl_mem inputBuffer = _inputBatch.getBuffer();
    cl_int inputWidth = _inputBatch.size(1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &inputWidth);

	size_t global_work_size[] = { _inputBatch.size(0) };
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
	clEnqueueReadBuffer(commandQueue, output.getBuffer(), CL_FALSE, 0, output.nElements() * sizeof(float), output.data(), 0, nullptr, nullptr);
}

}
