#include "Linear.h"

#include <fstream>
#include <iostream>

namespace rna
{

Linear::Linear(size_t _inputSize, size_t _outputSize):
    Layer("Linear"),
    weights{_outputSize, _inputSize}, bias{_outputSize}
{
    randomize();

    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

Linear::Linear(std::ifstream& _file):
    Layer("Linear")
{
    size_t inputSize, outputSize;
    _file >> inputSize >> outputSize;

    // Load weights
    weights.resize({outputSize, inputSize});
    for (unsigned i(0) ; i < outputSize ; i++)
        for (unsigned j(0) ; j < inputSize ; j++)
            _file >> weights(i, j);

    // Load bias
    bias.resize({outputSize});
    for (unsigned i(0) ; i < outputSize ; i++)
        _file >> bias(i);


    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

void Linear::randomize()
{
    weights.randomize(-1.0, 1.0);
    bias.randomize(-1.0, 1.0);
}

Tensor Linear::feedForward(const Tensor& _input)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(output, weights, _input);
        output += bias;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmmt(output, _input, weights);

        for (unsigned i(0); i < output.size(0); i++)
            for (unsigned j(0); j < output.size(1); j++)
                output(i, j) += bias(j);
    }

    return output;
}

Tensor Linear::backprop(const Tensor& _input, const Tensor& _gradOutput)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(gradInput, weights.getTranspose(), _gradOutput);

        gradWeight.addOuterProduct(_gradOutput, _input);
        gradBias += _gradOutput;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmm(gradInput, _gradOutput, weights);

        Tensor temp; mulmtm(temp, _gradOutput, _input);
        gradWeight += temp;

        for (unsigned i(0) ; i < _gradOutput.size(0) ; i++)
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                gradBias(j) += _gradOutput(i, j);
    }

    return gradInput;
}

void Linear::zeroParametersGradients()
{
    gradWeight.fill(0.0);
    gradBias.fill(0.0);
}

void Linear::updateParameters(double _learningRate, double _inertia)
{
//    deltaWeight = _inertia * deltaWeight + _learningRate * gradWeight;
//    deltaBias = _inertia * deltaBias + _learningRate * gradBias;
//
//    weights -= deltaWeight;
//    bias    -= deltaBias;

    deltaWeight = (1.0 - _inertia) * _learningRate * gradWeight + _inertia * deltaWeight;
    deltaBias   = (1.0 - _inertia) * _learningRate * gradBias   + _inertia * deltaBias;

    weights -= deltaWeight;
    bias    -= deltaBias;
}


void Linear::saveToFile(std::ofstream& _file) const
{
    _file << weights.size(1) << "   " << weights.size(0) << std::endl;

    // Save weights
    for (unsigned i(0) ; i < weights.size(0) ; i++)
    {
        for (unsigned j(0) ; j < weights.size(1) ; j++)
            _file << weights(i, j) << " ";

        _file << std::endl;
    }

    // Save bias
    for (unsigned i(0) ; i < bias.size(0) ; i++)
        _file << bias(i) << " ";
}

}
