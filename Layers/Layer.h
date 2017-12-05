#pragma once

#include "../Tensor.h"

double sigmoid(double _x);

double dSigmoid(double _x);
double dtanh(double _x);

namespace rna
{

class Layer
{
    friend class Network;

    public:
        Layer(std::string _type): type(_type) {}
        virtual ~Layer() {}

        virtual const Tensor& feedForward(const Tensor& _input) = 0;
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput) = 0;

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

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);
};

class ReLU: public Layer
{
    public:
        ReLU(): Layer("ReLU") {}

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);
};


}
