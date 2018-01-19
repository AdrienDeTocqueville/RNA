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

        #ifdef USE_OPENCL
        virtual void openCL(cl::Context& _context);
        virtual void releaseCL() override;

        virtual void feedForward(cl::CommandQueue&, const Tensor& _inputBatch);
        virtual void backprop(cl::CommandQueue&, const Tensor& _inputBatch, const Tensor& _outputGradBatch);

        void updateInputGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        void updateParamsGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        #else
        virtual void feedForward(const Tensor& _input);
        virtual void backprop(const Tensor& _input, const Tensor& _outputGrad);
        #endif // USE_OPENCL


        virtual void getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) override;

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        Tensor weights, weightsGrad;
        Tensor bias, biasGrad;

        #ifdef USE_OPENCL
        cl::Kernel weightsGradKernel, biasGradKernel;
        #endif // USE_OPENCL
};

}
