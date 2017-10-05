#pragma once

#include "Layers.h"

namespace rna
{

class Convolutional: public Layer
{
    public:
        Convolutional(coords_t inputDimensions = {3, 32, 32}, coords_t kernelDimensions = {3, 3}, size_t _outputChannels = 3);
        Convolutional(std::ifstream& _file);

        void randomize();

        virtual Tensor feedForward(const Tensor& _input);
        virtual Tensor backprop(const Tensor& _input, const Tensor& _gradOutput);

        virtual void zeroParametersGradients();
        virtual void updateParameters(double _learningRate, double _inertia);

        virtual void saveToFile(std::ofstream& _file) const;

    private:
        Tensor weights;
        Tensor bias;

        Tensor gradWeight;
        Tensor gradBias;

        Tensor deltaWeight;
        Tensor deltaBias;
};

void convGradInput(Tensor& gradInput, const Tensor& gradOutput, const Tensor& kernel);
void convGradWeight(Tensor& gradWeight, const Tensor& input, const Tensor& gradOutput);

}
