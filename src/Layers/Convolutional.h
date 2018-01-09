#pragma once

#include "Layer.h"

namespace rna
{

class Convolutional: public Layer
{
    public:
        Convolutional(coords_t inputDimensions = {3, 32, 32}, coords_t kernelDimensions = {3, 3}, size_t _outputChannels = 3);
        Convolutional(std::ifstream& _file);

        void randomize();


        virtual void feedForwardCPU(const Tensor& _input);
        virtual void feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch);

        virtual void backpropCPU(const Tensor& _input, const Tensor& _outputGrad);
        virtual void backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);

        virtual void updateInputGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);
        virtual void updateParamsGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch);


        virtual void getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad) override;

        virtual void saveToFile(std::ofstream& _file) const override;

    private:
        virtual void openCL(cl::Context& _context) override;
        virtual void releaseCL() override;

        Tensor weights, weightsGrad;
        Tensor bias, biasGrad;

        cl::Kernel weightsGradKernel, biasGradKernel;
};

void convGradInput(Tensor& inputGrad, const Tensor& kernel, const Tensor& outputGrad);
void convGradWeight(Tensor& weightsGrad, const Tensor& outputGrad, const Tensor& input);

}
