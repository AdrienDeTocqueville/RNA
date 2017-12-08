#include "Layer.h"

#include <cmath>
#include <fstream>

#include "../Utility/Error.h"
#include "../Utility/util.h"

namespace rna
{

Tensor::value_type sigmoid(Tensor::value_type _x)
{
    return Tensor::value_type(1.0) / ( Tensor::value_type(1.0) + exp(-_x) );
}

Tensor::value_type dSigmoid(Tensor::value_type _x)
{
    Tensor::value_type s = sigmoid(_x);
    return s*(Tensor::value_type(1.0) - s);
}

Tensor::value_type dtanh(Tensor::value_type _x)
{
    Tensor::value_type t = tanh(_x);
    return Tensor::value_type(1.0) - t*t;
}


/// Layer
Layer::Layer(std::string _type):
    type(_type),
    kernelForward(0),
    kernelBackward(0)
{ }

Layer::~Layer()
{
    leaveGPU();
}

const Tensor& Layer::getOutput() const
{
    return output;
}

const Tensor& Layer::getGradInput() const
{
    return gradInput;
}

void Layer::saveToFile(std::ofstream& _file) const
{
    _file << type << std::endl;
}

void Layer::leaveGPU()
{
	clReleaseKernel(kernelForward); kernelForward = 0;
	clReleaseKernel(kernelBackward); kernelBackward = 0;
}

/// Tanh
void Tanh::toGPU(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!kernelForward)
        kernelForward = loadKernel(_context, _deviceId, "OpenCL/activations.cl", "tanhForward");

    if (!kernelBackward)
        kernelBackward = loadKernel(_context, _deviceId, "OpenCL/activations.cl", "tanhBackward");
}

void Tanh::feedForwardCPU(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        output[i] = tanh(_input[i]);
}

void Tanh::feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resizeAs(_inputBatch);
    output.toGPU(context);

    clSetKernelArg(kernelForward, 0, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelForward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());

    execKernel(_commandQueue, kernelForward, { _inputBatch.size(0), _inputBatch.size(1) });
	output.readBuffer(_commandQueue);
}

void Tanh::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    for (unsigned i(0) ; i < gradInput.nElements() ; i++)
        gradInput[i] = dtanh(_input[i]) * _gradOutput[i];
}

void Tanh::backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    gradInput.resizeAs(_inputBatch);
    gradInput.toGPU(context);

    clSetKernelArg(kernelBackward, 0, sizeof(cl_mem), &gradInput.getBuffer());
    clSetKernelArg(kernelBackward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());
    clSetKernelArg(kernelBackward, 2, sizeof(cl_mem), &_gradOutputBatch.getBuffer());

    execKernel(_commandQueue, kernelBackward, { _inputBatch.size(0), _inputBatch.size(1) });
	gradInput.readBuffer(_commandQueue);
}


/// ReLU
void ReLU::toGPU(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!kernelForward)
        kernelForward = loadKernel(_context, _deviceId, "OpenCL/activations.cl", "reluForward");

    if (!kernelBackward)
        kernelBackward = loadKernel(_context, _deviceId, "OpenCL/activations.cl", "reluBackward");
}

void ReLU::feedForwardCPU(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < output.nElements() ; i++)
		output[i] = std::max(_input[i], Tensor::value_type(0.0));
}

void ReLU::feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resizeAs(_inputBatch);
    output.toGPU(context);

    clSetKernelArg(kernelForward, 0, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelForward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());

    execKernel(_commandQueue, kernelForward, { _inputBatch.size(0), _inputBatch.size(1) });
	output.readBuffer(_commandQueue);
}

void ReLU::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    for (unsigned i(0) ; i < gradInput.nElements() ; i++)
	    gradInput[i] = (_input[i] < 0.0)? 0.0: _gradOutput[i];
}

void ReLU::backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    gradInput.resizeAs(_inputBatch);
    gradInput.toGPU(context);

    clSetKernelArg(kernelBackward, 0, sizeof(cl_mem), &gradInput.getBuffer());
    clSetKernelArg(kernelBackward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());
    clSetKernelArg(kernelBackward, 2, sizeof(cl_mem), &_gradOutputBatch.getBuffer());

    execKernel(_commandQueue, kernelBackward, { _inputBatch.size(0), _inputBatch.size(1) });
	gradInput.readBuffer(_commandQueue);
}

}
