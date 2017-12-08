#pragma once

#include "../clWrapper.h"

namespace rna
{

Tensor::value_type sigmoid(Tensor::value_type _x);

Tensor::value_type dSigmoid(Tensor::value_type _x);
Tensor::value_type dtanh(Tensor::value_type _x);


class Layer
{
    friend class Network;

    public:
        Layer(std::string _type);
        virtual ~Layer();

        virtual void feedForwardCPU(const Tensor& _input) = 0;
        virtual void feedForwardGPU(const cl_command_queue&, const Tensor&) {}

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput) = 0;
        virtual void backpropGPU(const cl_command_queue&, const Tensor&, const Tensor&) {}


        virtual void zeroParametersGradients() {}
        virtual void updateParameters(Tensor::value_type, Tensor::value_type) {}

        const Tensor& getOutput() const;
        const Tensor& getGradInput() const;

        virtual void saveToFile(std::ofstream& _file) const;

    protected:
        virtual void toGPU(const cl_context&, const cl_device_id&) {}
        virtual void leaveGPU();

        std::string type;

        Tensor output, gradInput;

        cl_kernel kernelForward, kernelBackward;
};

class Tanh: public Layer
{
    public:
        Tanh(): Layer("Tanh") {}

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

    private:
        virtual void toGPU(const cl_context& _context, const cl_device_id& _deviceId) override;
};

class ReLU: public Layer
{
    public:
        ReLU(): Layer("ReLU") {}

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch) override;

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);

    private:
        virtual void toGPU(const cl_context& _context, const cl_device_id& _deviceId) override;
};


}
