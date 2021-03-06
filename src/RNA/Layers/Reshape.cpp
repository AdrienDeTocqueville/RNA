#include "RNA/Layers/Reshape.h"

#include <fstream>

namespace rna
{

Reshape::Reshape(coords_t _dimensions, bool _useMinibatch):
    Layer("Reshape"),
    outputSize(_dimensions),
    useMinibatch(false)
{
    setBatchMode(_useMinibatch);
}

Reshape::Reshape(std::ifstream& _file):
    Layer("Reshape")
{
    size_t nDimensions;
    _file >> nDimensions >> useMinibatch;

    outputSize.resize(nDimensions);
    for (unsigned i(0) ; i < nDimensions ; i++)
        _file >> outputSize[i];
}

void Reshape::setBatchMode(bool _useMinibatch)
{
    if (_useMinibatch && !useMinibatch)
        outputSize.insert(outputSize.begin(), 0);

    else if (!_useMinibatch && useMinibatch)
        outputSize.erase(outputSize.begin());

    useMinibatch = _useMinibatch;
}

#ifdef USE_OPENCL
void Reshape::openCL(cl::Context& _context)
{
    setBatchMode(true);
}

void Reshape::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    outputSize[0] = _inputBatch.size(0);

    output.resize(outputSize);
    output.openCL(_commandQueue.getContext());

    _commandQueue.enqueueCopy(_inputBatch.getBuffer(), output.getBuffer(), _inputBatch.nElements() * sizeof(Tensor::value_type));
}

void Reshape::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    _commandQueue.enqueueCopy(_outputGradBatch.getBuffer(), inputGrad.getBuffer(), _outputGradBatch.nElements() * sizeof(Tensor::value_type));
}

#else
void Reshape::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    inputGrad = _outputGrad;
    inputGrad.resizeAs(_input);
}

void Reshape::feedForward(const Tensor& _input)
{
    if (useMinibatch)
        outputSize[0] = _input.size(0);

    output = _input;
    output.resize(outputSize);
}
#endif // USE_OPENCL

void Reshape::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << outputSize.size() << " " << useMinibatch << std::endl;

    for (unsigned i(0) ; i < outputSize.size() ; i++)
        _file << outputSize[i] << " ";

    _file << std::endl;
}

}
