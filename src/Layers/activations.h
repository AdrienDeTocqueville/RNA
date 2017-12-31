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

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

        virtual Tensor::value_type f(Tensor::value_type _value) = 0;
        virtual Tensor::value_type df(Tensor::value_type _value) = 0;
};


class Tanh: public Activation
{
    public:
        Tanh(): Activation("Tanh") {}

        virtual Tensor::value_type f(Tensor::value_type _value) override;
        virtual Tensor::value_type df(Tensor::value_type _value) override;

    private:
        virtual void openCL(cl::ContextWrapper& _context) override;
};

class ReLU: public Activation
{
    public:
        ReLU(): Activation("ReLU") {}

        virtual Tensor::value_type f(Tensor::value_type _value) override;
        virtual Tensor::value_type df(Tensor::value_type _value) override;


    private:
        virtual void openCL(cl::ContextWrapper& _context) override;
};


}