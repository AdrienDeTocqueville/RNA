#pragma once

#include "Layer.h"

namespace rna
{

class Linear: public Layer
{
    public:
        Linear(size_t _inputSize, size_t _outputSize);
        Linear(std::ifstream& _file);

        void randomize();

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);

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

}
