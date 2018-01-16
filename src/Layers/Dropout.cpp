#include "Dropout.h"

#include <fstream>

#include "../Utility/Random.h"

namespace rna
{

Dropout::Dropout(Tensor::value_type _rate):
    Layer("Dropout"),
    rate(_rate)
{}

Dropout::Dropout(std::ifstream& _file):
    Layer("Dropout")
{
    _file >> rate;
}

void Dropout::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/dropout.cl");

    forwardKernel.create(p, "feedForwardDropout");
    backwardKernel.create(p, "backpropDropout");
}

void Dropout::feedForwardCPU(const Tensor& _input)
{
    output = _input;

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        if (Random::nextFloat() < rate)
            output[i] = 0.0;
}

void Dropout::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    rands.resizeAs(_inputBatch);
    output.resizeAs(_inputBatch);

    rands.openCL(_commandQueue.getContext());
    output.openCL(_commandQueue.getContext());

    rands.randomize(0.0f, 1.0f);
    rands.writeBuffer(_commandQueue);

    cl_int inputWidth = _inputBatch.getStride(0);

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2, inputWidth);
    forwardKernel.setArg(3, rands);
    forwardKernel.setArg(4, rate);

    _commandQueue.enqueue(forwardKernel,  { _inputBatch.size(0) });
}

void Dropout::backpropCPU(const Tensor& _input, const Tensor& _outputGrad)
{
    inputGrad = _outputGrad;

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        if (output[i] == 0.0f)
            inputGrad[i] = 0.0f;
}

void Dropout::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    cl_int inputWidth = _inputBatch.getStride(0);

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1, output);
    backwardKernel.setArg(2,_outputGradBatch);
    backwardKernel.setArg(3, sizeof(cl_int), &inputWidth);

    backwardKernel.enqueue(_commandQueue,  { _inputBatch.size(0) });
}

void Dropout::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << rate << std::endl;
}

}
