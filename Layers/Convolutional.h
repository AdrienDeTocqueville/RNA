#pragma once

#include "Layer.h"

namespace rna
{

class Convolutional: public Layer
{
    public:
        Convolutional(coords_t inputDimensions = {3, 32, 32}, coords_t kernelDimensions = {3, 3}, size_t _outputChannels = 3);
        Convolutional(std::ifstream& _file);

        void randomize();

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);

        virtual void zeroParametersGradients() override;
        virtual void updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia) override;

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        Tensor weights;
        Tensor bias;

        Tensor gradWeight;
        Tensor gradBias;

        Tensor deltaWeight;
        Tensor deltaBias;
};

void convGradInput(Tensor& gradInput, const Tensor& kernel, const Tensor& gradOutput);
void convGradWeight(Tensor& gradWeight, const Tensor& gradOutput, const Tensor& input);

}
