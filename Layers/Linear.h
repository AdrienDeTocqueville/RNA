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

        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput);
        virtual void backpropGPU(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch);


        virtual void zeroParametersGradients() override;
        virtual void updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia) override;

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        virtual void toGPU(const cl_context& _context, const cl_device_id& _deviceId) override;
        virtual void leaveGPU() override;

        Tensor weights;
        Tensor bias;

        Tensor gradWeight;
        Tensor gradBias;

        Tensor deltaWeight;
        Tensor deltaBias;

        cl_kernel kernelGradParam;
};

}
