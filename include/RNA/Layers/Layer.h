#pragma once

#include "Utility/clWrapper.h"
#include "Utility/Tensor.h"

namespace rna
{

class Layer
{
    friend class Network; // TODO: remove

    public:
        Layer(std::string _type);
        virtual ~Layer();

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context) = 0;
        virtual void releaseCL();

        virtual void feedForward(cl::CommandQueue&, const Tensor& _inputBatch) = 0;
        virtual void backprop(cl::CommandQueue&, const Tensor& _inputBatch, const Tensor& _outputGradBatch) = 0;

//        virtual void updateInputGrad(cl::CommandQueue&, const Tensor&, const Tensor&) {};
//        virtual void updateParamsGrad(cl::CommandQueue&, const Tensor&, const Tensor&) {};
        #else
        virtual void feedForward(const Tensor& _input) = 0;
        virtual void backprop(const Tensor& _input, const Tensor& _outputGrad) = 0;
        #endif // USE_OPENCL


        const Tensor& getOutput() const;
        const Tensor& getInputGrad() const;
        const std::string& getType() const;

        virtual void getParams(std::vector<Tensor*>&, std::vector<Tensor*>&) {}

        virtual void saveToFile(std::ofstream& _file) const;


        static Tensor::value_type WEIGHT_INIT_MIN;
        static Tensor::value_type WEIGHT_INIT_MAX;

        static Tensor::value_type BIAS_INIT_MIN;
        static Tensor::value_type BIAS_INIT_MAX;

    protected:
        std::string type;

        Tensor output, inputGrad;

        #ifdef USE_OPENCL
        cl::Kernel forwardKernel, backwardKernel;
        #endif // USE_OPENCL
};

}
