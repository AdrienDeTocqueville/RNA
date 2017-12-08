#include "Linear.h"

#include <fstream>
#include <iostream>

#include "../Utility/Error.h"

namespace rna
{

Linear::Linear(size_t _inputSize, size_t _outputSize):
    Layer("Linear"),
    weights{_outputSize, _inputSize}, bias{_outputSize},
    kernelGradParam(0)
{
    randomize();

    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

Linear::Linear(std::ifstream& _file):
    Layer("Linear"),
    kernelGradParam(0)
{
    size_t inputSize, outputSize;
    _file >> inputSize >> outputSize;

    // Load weights
    weights.resize({outputSize, inputSize});
    for (unsigned i(0) ; i < outputSize ; i++)
        for (unsigned j(0) ; j < inputSize ; j++)
            _file >> weights(i, j);

    // Load bias
    bias.resize({outputSize});
    for (unsigned i(0) ; i < outputSize ; i++)
        _file >> bias(i);


    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

void Linear::randomize()
{
    weights.randomize(-1.0, 1.0);
    bias.randomize(-1.0, 1.0);
}

void Linear::toGPU(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!kernelForward)
        kernelForward = loadKernel(_context, _deviceId, "OpenCL/linear.cl", "linearForward");

    if (!kernelBackward)
        kernelBackward = loadKernel(_context, _deviceId, "OpenCL/linear.cl", "linearBackward");

    if (!kernelGradParam)
        kernelGradParam = loadKernel(_context, _deviceId, "OpenCL/linear.cl", "linearParametersGradients");

    weights.toGPU(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    bias.toGPU(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    gradWeight.toGPU(_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    gradBias.toGPU(_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
}

void Linear::leaveGPU()
{
	Layer::leaveGPU();

    clReleaseKernel(kernelGradParam); kernelGradParam = 0;
}

void Linear::feedForwardCPU(const Tensor& _input)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(output, weights, _input);
        output += bias;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmmt(output, _input, weights);

        for (unsigned i(0); i < output.size(0); i++)
            for (unsigned j(0); j < output.size(1); j++)
                output(i, j) += bias(j);
    }
}

void Linear::feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resize({_inputBatch.size(0), weights.size(0)});
    output.toGPU(context);

    cl_int inputWidth = _inputBatch.size(1);

    clSetKernelArg(kernelForward, 0, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelForward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());
    clSetKernelArg(kernelForward, 2, sizeof(cl_mem), &weights.getBuffer());
    clSetKernelArg(kernelForward, 3, sizeof(cl_mem), &bias.getBuffer());
    clSetKernelArg(kernelForward, 4, sizeof(cl_int), &inputWidth);

    execKernel(_commandQueue, kernelForward, { output.size(0), output.size(1) });
	output.readBuffer(_commandQueue);
}

void Linear::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(gradInput, weights.getTranspose(), _gradOutput);

        gradWeight.addOuterProduct(_gradOutput, _input);
        gradBias += _gradOutput;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmm(gradInput, _gradOutput, weights);

        Tensor temp; mulmtm(temp, _gradOutput, _input);
        gradWeight += temp;

        for (unsigned i(0) ; i < _gradOutput.size(0) ; i++)
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                gradBias(j) += _gradOutput(i, j);
    }
}

void Linear::backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    gradInput.resizeAs({_gradOutputBatch.size(0), weights.size(1)});
    gradInput.toGPU(context);

    cl_int gradOutputWidth = _gradOutputBatch.size(1);
    cl_int inputWidth = _inputBatch.size(1);

    // gradInput
    clSetKernelArg(kernelBackward, 0, sizeof(cl_mem), &gradInput.getBuffer());
    clSetKernelArg(kernelBackward, 1, sizeof(cl_mem), &_gradOutputBatch.getBuffer());
    clSetKernelArg(kernelBackward, 2, sizeof(cl_mem), &weights.getBuffer());
    clSetKernelArg(kernelBackward, 3, sizeof(cl_int), &gradOutputWidth);

    execKernel(_commandQueue, kernelBackward, { gradInput.size(0), gradInput.size(1) });
	gradInput.readBuffer(_commandQueue);

    // gradWeight, gradBias
    clSetKernelArg(kernelGradParam, 0, sizeof(cl_mem), &gradWeight.getBuffer());
    clSetKernelArg(kernelGradParam, 1, sizeof(cl_mem), &gradBias.getBuffer());
    clSetKernelArg(kernelGradParam, 2, sizeof(cl_mem), &_gradOutputBatch.getBuffer());
    clSetKernelArg(kernelGradParam, 3, sizeof(cl_mem), &_inputBatch.getBuffer());
    clSetKernelArg(kernelGradParam, 4, sizeof(cl_int), &inputWidth);

    execKernel(_commandQueue, kernelGradParam, { _gradOutputBatch.size(0), _gradOutputBatch.size(1) });
	gradWeight.readBuffer(_commandQueue);
	gradBias.readBuffer(_commandQueue);
}

void Linear::zeroParametersGradients()
{
    gradWeight.fill(0.0);
    gradBias.fill(0.0);
}

void Linear::updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia)
{
//    deltaWeight = _inertia * deltaWeight + _learningRate * gradWeight;
//    deltaBias = _inertia * deltaBias + _learningRate * gradBias;
//
//    weights -= deltaWeight;
//    bias    -= deltaBias;

    // TODO: Fix inertia
    deltaWeight = (1.0 - _inertia) * _learningRate * gradWeight + _inertia * deltaWeight;
    deltaBias   = (1.0 - _inertia) * _learningRate * gradBias   + _inertia * deltaBias;

    weights -= deltaWeight;
    bias    -= deltaBias;
}


void Linear::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << weights.size(1) << "   " << weights.size(0) << std::endl;

    // Save weights
    for (unsigned i(0) ; i < weights.size(0) ; i++)
    {
        for (unsigned j(0) ; j < weights.size(1) ; j++)
            _file << weights(i, j) << " ";

        _file << std::endl;
    }

    // Save bias
    for (unsigned i(0) ; i < bias.size(0) ; i++)
        _file << bias(i) << " ";
}

}
