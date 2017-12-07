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

        virtual void GPUfeedForward(cl_command_queue& commandQueue, const Tensor& _inputBatch);

        virtual void zeroParametersGradients();
        virtual void updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia);

        virtual void saveToFile(std::ofstream& _file) const;

    private:
        virtual void toGPU(cl_context _context, cl_device_id _device);

        Tensor weights;
        Tensor bias;

        Tensor gradWeight;
        Tensor gradBias;

        Tensor deltaWeight;
        Tensor deltaBias;
};

}
