#pragma once

#include "Layer.h"

namespace rna
{

Tensor::value_type sigmoid(Tensor::value_type _x);

Tensor::value_type dSigmoid(Tensor::value_type _x);
Tensor::value_type dtanh(Tensor::value_type _x);


class Activation: public Layer
{
    public:
        Activation(std::string _name): Layer(_name) {}

        #ifdef USE_OPENCL
        virtual void feedForward(cl::CommandQueue&, const Tensor& _inputBatch);
        virtual void backprop(cl::CommandQueue&, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        #else
        virtual void feedForward(const Tensor& _input);
        virtual void backprop(const Tensor& _input, const Tensor& _outputGrad);
        #endif // USE_OPENCL


        virtual Tensor::value_type f(Tensor::value_type _value) = 0;
        virtual Tensor::value_type df(Tensor::value_type _value) = 0;
};


class Tanh: public Activation
{
    public:
        Tanh(): Activation("Tanh") {}

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);
        #endif // USE_OPENCL

        virtual Tensor::value_type f(Tensor::value_type _value) override;
        virtual Tensor::value_type df(Tensor::value_type _value) override;
};

class ReLU: public Activation
{
    public:
        ReLU(): Activation("ReLU") {}

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);
        #endif // USE_OPENCL

        virtual Tensor::value_type f(Tensor::value_type _value) override;
        virtual Tensor::value_type df(Tensor::value_type _value) override;
};

class ELU: public Activation
{
    public:
        ELU(Tensor::value_type _alpha = 1.0): Activation("ELU"), alpha(_alpha) {}
        ELU(std::ifstream& _file);

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);
        #endif // USE_OPENCL

        virtual Tensor::value_type f(Tensor::value_type _value) override;
        virtual Tensor::value_type df(Tensor::value_type _value) override;

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        Tensor::value_type alpha;
};


}
