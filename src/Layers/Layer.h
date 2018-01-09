#pragma once

#include "../clWrapper.h"
#include "../Utility/Tensor.h"

namespace rna
{

class Layer
{
    friend class Network;

    public:
        Layer(std::string _type);
        virtual ~Layer();

        virtual void feedForwardCPU(const Tensor& _input) = 0;
        virtual void feedForwardCL(cl::CommandQueue&, const Tensor&) = 0;

        virtual void backpropCPU(const Tensor& _input, const Tensor& _outputGrad) = 0;
        virtual void backpropCL(cl::CommandQueue&, const Tensor&, const Tensor&) = 0;

        virtual void updateInputGrad(cl::CommandQueue&, const Tensor&, const Tensor&) {};
        virtual void updateParamsGrad(cl::CommandQueue&, const Tensor&, const Tensor&) {};


        const Tensor& getOutput() const;
        const Tensor& getInputGrad() const;

        virtual void getParams(std::vector<Tensor*>&, std::vector<Tensor*>&) {}

        virtual void saveToFile(std::ofstream& _file) const;


        static Tensor::value_type WEIGHT_INIT_MIN;
        static Tensor::value_type WEIGHT_INIT_MAX;

        static Tensor::value_type BIAS_INIT_MIN;
        static Tensor::value_type BIAS_INIT_MAX;

    protected:
        virtual void openCL(cl::Context&) = 0;
        virtual void releaseCL();

        std::string type;

        Tensor output, inputGrad;

        cl::Kernel forwardKernel, backwardKernel;
};

}
