#pragma once

#include <CL/opencl.h>

#include "../Utility/Tensor.h"

namespace rna
{

Tensor::value_type sigmoid(Tensor::value_type _x);

Tensor::value_type dSigmoid(Tensor::value_type _x);
Tensor::value_type dtanh(Tensor::value_type _x);

std::string loadProgram(const std::string& path);

class Layer
{
    friend class Network;

    public:
        Layer(std::string _type);
        virtual ~Layer();

        virtual const Tensor& feedForward(const Tensor& _input) = 0;
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput) = 0;

        virtual void GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch) {}

        virtual void zeroParametersGradients() {}
        virtual void updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia) {}

        const Tensor& getOutput() const;

        virtual void saveToFile(std::ofstream& _file) const {}

    protected:
        virtual void toGPU(cl_context _context, cl_device_id _device) {}
        void loadKernel(cl_context _context, cl_device_id _device, std::string _program, std::string _kernel);

        Tensor output;

        Tensor gradInput;

        std::string type;

        cl_program program;
        cl_kernel kernel;
};

class Tanh: public Layer
{
    public:
        Tanh(): Layer("Tanh") {}

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);

        virtual void GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch);

    private:
        virtual void toGPU(cl_context _context, cl_device_id _device);
};

class ReLU: public Layer
{
    public:
        ReLU(): Layer("ReLU") {}

        virtual const Tensor& feedForward(const Tensor& _input);
        virtual const Tensor& backprop(const Tensor& _input, const Tensor& _gradOutput);

        virtual void GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch);

    private:
        virtual void toGPU(cl_context _context, cl_device_id _device);
};


}
