#include "Reshape.h"

#include <fstream>

namespace rna
{

Reshape::Reshape(coords_t _dimensions):
    Layer("Reshape"),
    outputShape{_dimensions}
{ }

Reshape::Reshape(std::ifstream& _file):
    Layer("Reshape")
{
    size_t nDimensions;
    _file >> nDimensions;

    coords_t dimensions(nDimensions);
    for (unsigned i(0) ; i < nDimensions ; i++)
        _file >> dimensions[i];

    outputShape.resize(dimensions);
}

Tensor Reshape::feedForward(const Tensor& _input)
{
    inputShape.resizeAs(_input);

    output = _input;
    output.resizeAs(outputShape);

    return output;
}

Tensor Reshape::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput = _gradOutput;
    gradInput.resizeAs(inputShape);

    return gradInput;
}


void Reshape::saveToFile(std::ofstream& _file) const
{
    _file << outputShape.nDimensions() << std::endl;

    for (unsigned i(0) ; i < outputShape.nDimensions() ; i++)
        _file << outputShape.size(i) << " ";
}

}
