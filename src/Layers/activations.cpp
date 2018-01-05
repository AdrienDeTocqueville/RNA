#include "activations.h"

#include <cmath>

#include "../Utility/Error.h"

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


/// Activation
void Activation::feedForwardCPU(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        output[i] = f(_input[i]);
}

void Activation::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);
    output.openCL(_commandQueue.getContext());

    cl_int inputWidth = _inputBatch.nElements() / _inputBatch.size(0); // TODO: use strides

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2, inputWidth);

    forwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
}

void Activation::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    inputGrad.resizeAs(_input);

    for (unsigned i(0) ; i < inputGrad.nElements() ; i++)
        inputGrad[i] = df(_input[i]) * _gradOutput[i];
}

void Activation::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    cl_int inputWidth = _inputBatch.nElements() / _inputBatch.size(0); // TODO: use strides

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_inputBatch);
    backwardKernel.setArg(2,_gradOutputBatch);
    backwardKernel.setArg(3, sizeof(cl_int), &inputWidth);

    backwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
	inputGrad.readBuffer(_commandQueue);
}


/// Tanh
void Tanh::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/activations.cl");

    forwardKernel.create(p, "feedForwardTanh");
    backwardKernel.create(p, "backpropTanh");
}

Tensor::value_type Tanh::f(Tensor::value_type _value)
{
    return tanh(_value);
}

Tensor::value_type Tanh::df(Tensor::value_type _value)
{
    Tensor::value_type t = tanh(_value);

    return Tensor::value_type(1.0) - t*t;
}


/// ReLU
void ReLU::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/activations.cl");

    forwardKernel.create(p, "feedForwardReLU");
    backwardKernel.create(p, "backpropReLU");
}

Tensor::value_type ReLU::f(Tensor::value_type _value)
{
    return std::max(_value, Tensor::value_type(0.0));
}

Tensor::value_type ReLU::df(Tensor::value_type _value)
{
    return (_value < 0.0)? 0.0: 1.0;
}

}
