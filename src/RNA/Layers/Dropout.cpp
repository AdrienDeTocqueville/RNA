#include "RNA/Layers/Dropout.h"
#include "Utility/Random.h"

#include <fstream>

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

#ifdef USE_OPENCL
void Dropout::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/dropout.cl");

    forwardKernel.create(p, "feedForwardDropout");
    backwardKernel.create(p, "backpropDropout");
}

void Dropout::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    rands.resizeAs(_inputBatch);
    output.resizeAs(_inputBatch);

    rands.openCL(_commandQueue.getContext());
    output.openCL(_commandQueue.getContext());

    rands.randomize(0.0f, 1.0f);
    _commandQueue.enqueueWrite(rands);

    int inputWidth = _inputBatch.getStride(0);

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(2, inputWidth);
    forwardKernel.setArg(3, rands);
    forwardKernel.setArg(4, rate);

    _commandQueue.enqueueKernel(forwardKernel, { _inputBatch.size(0) });
}

void Dropout::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    int inputWidth = _inputBatch.getStride(0);

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1, output);
    backwardKernel.setArg(2,_outputGradBatch);
    backwardKernel.setArg(3, sizeof(int), &inputWidth);

    _commandQueue.enqueueKernel(backwardKernel, {_inputBatch.size(0)});
}

#else
void Dropout::feedForward(const Tensor& _input)
{
    output = _input;

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        if (Random::next<Tensor::value_type>() < rate)
            output[i] = 0.0;
}


void Dropout::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    inputGrad = _outputGrad;

    for (unsigned i(0) ; i < _input.nElements() ; i++)
        if (output[i] == 0.0f)
            inputGrad[i] = 0.0f;
}
#endif // USE_OPENCL

void Dropout::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << rate << std::endl;
}

}
