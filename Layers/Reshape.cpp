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

const Tensor& Reshape::feedForward(const Tensor& _input)
{
    output = _input;
    output.resize(outputSize);

    return output;
}

const Tensor& Reshape::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput = _gradOutput;
    gradInput.resizeAs(_input);

    return gradInput;
}


void Reshape::saveToFile(std::ofstream& _file) const
{
    _file << outputSize.size() << std::endl;

    for (unsigned i(0) ; i < outputSize.size() ; i++)
        _file << outputSize[i];
}

}
