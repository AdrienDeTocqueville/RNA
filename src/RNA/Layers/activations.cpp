#include "RNA/Layers/activations.h"
#include "Utility/Error.h"

#include <cmath>
#include <fstream>

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
#ifdef USE_OPENCL
void Activation::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resizeAs(_inputBatch);
    output.openCL(_commandQueue.getContext());

    int inputWidth = _inputBatch.getStride(0);

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2, inputWidth);

    _commandQueue.enqueueKernel(forwardKernel, { _inputBatch.size(0) });
}

void Activation::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_inputBatch);
    backwardKernel.setArg(2, output);
    backwardKernel.setArg(3,_outputGradBatch);
    backwardKernel.setArg(4, _inputBatch.getStride(0));

    _commandQueue.enqueueKernel(backwardKernel, {_inputBatch.size(0)});
}

#else
void Activation::feedForward(const Tensor& _input)
{
    output.resizeAs(_input);

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        output[i] = f(_input[i]);
}

void Activation::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    inputGrad.resizeAs(_input);

    for (unsigned i(0) ; i < inputGrad.nElements() ; i++)
        inputGrad[i] = df(_input[i]) * _outputGrad[i];
}
#endif // USE_OPENCL

/// Tanh
#ifdef USE_OPENCL
void Tanh::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/activations.cl");

    forwardKernel.create(p, "feedForwardTanh");
    backwardKernel.create(p, "backpropTanh");
}
#endif // USE_OPENCL

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
#ifdef USE_OPENCL
void ReLU::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/activations.cl");

    forwardKernel.create(p, "feedForwardReLU");
    backwardKernel.create(p, "backpropReLU");
}
#endif // USE_OPENCL

Tensor::value_type ReLU::f(Tensor::value_type _value)
{
    return std::max(_value, Tensor::value_type(0.0));
}

Tensor::value_type ReLU::df(Tensor::value_type _value)
{
    return (_value < 0.0)? 0.0: 1.0;
}


/// ELU
ELU::ELU(std::ifstream& _file):
    Activation("ELU")
{
    _file >> alpha;
}

#ifdef USE_OPENCL
void ELU::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/activations.cl");

    forwardKernel.create(p, "feedForwardELU");
    backwardKernel.create(p, "backpropELU");

    forwardKernel.setArg(3, alpha);
    backwardKernel.setArg(5, alpha);
}
#endif // USE_OPENCL

Tensor::value_type ELU::f(Tensor::value_type _value)
{
    return _value < 0.0? alpha * (exp(_value)-1.0): _value;
}

Tensor::value_type ELU::df(Tensor::value_type _value)
{
    return _value < 0.0? alpha * exp(_value): 1.0;
}

void ELU::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << alpha << std::endl;
}

}
