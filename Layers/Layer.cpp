#include "Layer.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "../Utility/Error.h"
#include "../Utility/util.h"

namespace rna
{

const Tensor& Layer::getOutput() const
{
    return output;
}

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

std::string loadProgram(const std::string& path)
{
	std::ifstream in(path);
	if (!in)
    {
        Error::add(ErrorType::FILE_NOT_FOUND, path);
        return "";
    }

	return std::string( (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>() );
}

Layer::Layer(std::string _type):
    type(_type),
    program(0), kernel(0)
{ }

Layer::~Layer()
{
	clReleaseKernel(kernel);
	clReleaseProgram(program);
}

void Layer::loadKernel(cl_context _context, cl_device_id _device, std::string _program, std::string _kernel)
{
    cl_int error;
    std::string src = loadProgram(_program);
    const char* strings = src.c_str();
    const size_t length = src.size();


    program = clCreateProgramWithSource(_context, 1, &strings, &length, &error);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::UNKNOWN_ERROR, "Unable to create program" + toString(error));


    error = clBuildProgram(program, 1, &_device, nullptr, nullptr, nullptr);//"-cl-std=CL2.0"
    if (error != CL_SUCCESS)
    {
        size_t logLength;
        error = clGetProgramBuildInfo(program, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);

        char* log = new char[logLength];
        error = clGetProgramBuildInfo(program, _device, CL_PROGRAM_BUILD_LOG, logLength, log, nullptr);

        std::cout << log << std::endl;
        delete[] log;

        Error::add(ErrorType::USER_ERROR, "Build Error");
    }


    kernel = clCreateKernel(program, _kernel.c_str(), &error);
    if (error != CL_SUCCESS)
        Error::add(ErrorType::USER_ERROR, "Unable to create kernel " + toString(error));
}

void Tanh::toGPU(cl_context _context, cl_device_id _device)
{
    loadKernel(_context, _device, "OpenCL/nonLinear.cl", "tanhLayer");
}

const Tensor& Tanh::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        output[i] = tanh(_input[i]);

    return output;
}

const Tensor& Tanh::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);

    for (unsigned i(0) ; i < gradInput.nElements() ; i++)
        gradInput[i] = dtanh(_input[i]) * _gradOutput[i];

    return gradInput;
}

void Tanh::GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);

    cl_context context;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.toGPU(context, CL_MEM_WRITE_ONLY);

    cl_mem outputBuffer = output.getBuffer();
    cl_mem inputBuffer = _inputBatch.getBuffer();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBuffer);

	size_t global_work_size[] = { _inputBatch.nElements() };
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
	clEnqueueReadBuffer(commandQueue, output.getBuffer(), CL_FALSE, 0, output.nElements() * sizeof(float), output.data(), 0, nullptr, nullptr);
}

void ReLU::toGPU(cl_context _context, cl_device_id _device)
{
    loadKernel(_context, _device, "OpenCL/nonLinear.cl", "reluLayer");
}

const Tensor& ReLU::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < output.nElements() ; i++)
		output[i] = std::max(_input[i], Tensor::value_type(0.0));

    return output;
}

const Tensor& ReLU::backprop(const Tensor& _input, const Tensor& _gradOutput)
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

void ReLU::GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);

    cl_context context;
    clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.toGPU(context, CL_MEM_WRITE_ONLY);

    cl_mem outputBuffer = output.getBuffer();
    cl_mem inputBuffer = _inputBatch.getBuffer();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBuffer);

	size_t global_work_size[] = { _inputBatch.nElements() };
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
	clEnqueueReadBuffer(commandQueue, output.getBuffer(), CL_FALSE, 0, output.nElements() * sizeof(float), output.data(), 0, nullptr, nullptr);
}

}
