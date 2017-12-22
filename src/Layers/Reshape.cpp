#include "Reshape.h"

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

void Reshape::feedForwardCPU(const Tensor& _input)
{
    if (useMinibatch)
        outputSize[0] = _input.size(0);

    output = _input;
    output.resize(outputSize);
}

void Reshape::feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    clFinish(_commandQueue); // Wait for previous layers to finish

    feedForwardCPU(_inputBatch); // TODO: Move op to GPU ?

    output.openCL(context);
}

void Reshape::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput = _gradOutput;
    gradInput.resizeAs(_input);
}

void Reshape::backpropCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    clFinish(_commandQueue); // Wait for previous layers to finish

    backpropCPU(_inputBatch, _gradOutputBatch); // TODO: Move op to GPU ?

    gradInput.openCL(context);
}


void Reshape::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << outputSize.size() << " " << useMinibatch << std::endl;

    for (unsigned i(0) ; i < outputSize.size() ; i++)
        _file << outputSize[i] << " ";

    _file << std::endl;
}

}
