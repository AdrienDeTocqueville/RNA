#pragma once

#include "../clWrapper.h"

namespace rna
{

class Layer
{
    friend class Network;

    public:
        Layer(std::string _type);
        virtual ~Layer();

        virtual void feedForwardCPU(const Tensor& _input) = 0;
        virtual void feedForwardCL(const cl_command_queue&, const Tensor&) {}

        virtual void backpropCPU(const Tensor& _input, const Tensor& _gradOutput) = 0;
        virtual void backpropCL(const cl_command_queue&, const Tensor&, const Tensor&) {}


        virtual void zeroParametersGradients() {}
        virtual void updateParameters(Tensor::value_type, Tensor::value_type) {}

        const Tensor& getOutput() const;
        const Tensor& getGradInput() const;

        virtual void saveToFile(std::ofstream& _file) const;

        static Tensor::value_type WEIGHT_INIT_MIN;
        static Tensor::value_type WEIGHT_INIT_MAX;

        static Tensor::value_type BIAS_INIT_MIN;
        static Tensor::value_type BIAS_INIT_MAX;

    protected:
        virtual void openCL(cl::ContextWrapper&) {}
        virtual void releaseCL();

        std::string type;

        Tensor output, gradInput;

        cl::KernelWrapper forwardKernel, backwardKernel;
};

}
