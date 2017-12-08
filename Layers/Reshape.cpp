#include "Reshape.h"

#include <fstream>

namespace rna
{

Reshape::Reshape(coords_t _dimensions):
    Layer("Reshape"),
    outputSize(_dimensions)
{ }

Reshape::Reshape(std::ifstream& _file):
    Layer("Reshape")
{
    size_t nDimensions;
    _file >> nDimensions;

    outputSize.resize(nDimensions);
    for (unsigned i(0) ; i < nDimensions ; i++)
        _file >> outputSize[i];
}

void Reshape::feedForwardCPU(const Tensor& _input)
{
    output = _input;
    output.resize(outputSize);
}

void Reshape::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput = _gradOutput;
    gradInput.resizeAs(_input);
}


void Reshape::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << outputSize.size() << std::endl;

    for (unsigned i(0) ; i < outputSize.size() ; i++)
        _file << outputSize[i];
}

}
