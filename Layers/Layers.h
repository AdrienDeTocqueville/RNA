#pragma once

#include "../Tensor.h"

namespace rna
{

class Layer
{
    friend class Network;

    public:
        Layer(std::string _type): type(_type) {}
        virtual ~Layer() {}

        virtual Tensor feedForward(const Tensor& _input) = 0;
        virtual Tensor backprop(const Tensor& _input, const Tensor& _gradOutput) = 0;

        virtual void zeroParametersGradients() {}
        virtual void updateParameters(double _learningRate, double _inertia) {}

        Tensor getOutput() const {return output;}

        virtual void saveToFile(std::ofstream& _file) const {}

    protected:
        Tensor output;

        Tensor gradInput;

        std::string type;
};

class Tanh: public Layer
{
    public:
        Tanh(): Layer("Tanh") {}

        virtual Tensor feedForward(const Tensor& _input);
        virtual Tensor backprop(const Tensor& _input, const Tensor& _gradOutput);
};

class ReLU: public Layer
{
    public:
        ReLU(): Layer("ReLU") {}

        virtual Tensor feedForward(const Tensor& _input);
        virtual Tensor backprop(const Tensor& _input, const Tensor& _gradOutput);
};


}
