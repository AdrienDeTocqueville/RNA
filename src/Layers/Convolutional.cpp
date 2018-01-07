#include "Convolutional.h"

#include <fstream>
#include <iostream>

#include "../Utility/Error.h"

namespace rna
{

Convolutional::Convolutional(coords_t inputDimensions, coords_t kernelDimensions, size_t _outputChannels):
    Layer("Convolutional"),
    weights{_outputChannels, inputDimensions[0], kernelDimensions[0], kernelDimensions[1]},
    bias{_outputChannels, inputDimensions[1]-kernelDimensions[0]+1, inputDimensions[2]-kernelDimensions[1]+1}
{
    randomize();

    weightsGrad.resizeAs(weights);
    biasGrad.resizeAs(bias);
}

Convolutional::Convolutional(std::ifstream& _file):
    Layer("Convolutional")
{
    coords_t weightsDimensions(4), biasDimensions(3);
    _file >> weightsDimensions[0] >> weightsDimensions[1] >> weightsDimensions[2] >> weightsDimensions[3];
    _file >> biasDimensions[0] >> biasDimensions[1] >> biasDimensions[2];

    weights.resize(weightsDimensions);
    bias.resize(biasDimensions);

    // Load weights
    for (unsigned i(0) ; i < weights.size(0) ; i++)
        for (unsigned j(0) ; j < weights.size(1) ; j++)
            for (unsigned k(0) ; k < weights.size(2); k++)
                for (unsigned l(0) ; l < weights.size(3); l++)
                    _file >> weights({i, j, k, l});

    // Load bias
    for (unsigned i(0) ; i < bias.size(0) ; i++)
        for (unsigned j(0) ; j < bias.size(1) ; j++)
            for (unsigned k(0) ; k < bias.size(2); k++)
                _file >> bias(i, j, k);


    weightsGrad.resizeAs(weights);
    biasGrad.resizeAs(bias);
}

void Convolutional::randomize()
{
    weights.randomize(Layer::WEIGHT_INIT_MIN, Layer::WEIGHT_INIT_MAX);
    bias.randomize(Layer::BIAS_INIT_MIN, Layer::BIAS_INIT_MAX);
}

void Convolutional::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/convolutional.cl");

    forwardKernel.create(p, "feedForwardConvolutional");
    backwardKernel.create(p, "backpropConvolutional");

    paramsGradKernel.create(p, "paramsGradConvolutional");


    weights.openCL(_context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    bias.openCL(_context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    weightsGrad.openCL(_context());
    biasGrad.openCL(_context());


    forwardKernel.setArg(2, weights);
    forwardKernel.setArg(3, bias);
    forwardKernel.setArg(4, weights.size(1));
    forwardKernel.setArg(5, weights.size(2));
    forwardKernel.setArg(6, weights.size(3));

    backwardKernel.setArg(2, weights);
    backwardKernel.setArg(3, weights.size(0));
    backwardKernel.setArg(4, weights.size(2));
    backwardKernel.setArg(5, weights.size(3));
}

void Convolutional::releaseCL()
{
	Layer::releaseCL();

    paramsGradKernel.release();
}

void Convolutional::feedForwardCPU(const Tensor& _input)
{
    convolve(output, weights, _input);
    output += bias;
}

void Convolutional::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resize({_inputBatch.size(0), weights.size(0), _inputBatch.size(2)-weights.size(2)+1, _inputBatch.size(3)-weights.size(3)+1});
    output.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        forwardKernel.setArg(7, i);
        _commandQueue.enqueue(forwardKernel, bias.size());
    }
}

void Convolutional::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    convGradInput(inputGrad, weights, _gradOutput);

    convGradWeight(weightsGrad, _gradOutput, _input);
    biasGrad += _gradOutput;
}

void Convolutional::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    updateInputGrad(_commandQueue, _inputBatch, _gradOutputBatch);
    updateParamsGrad(_commandQueue, _inputBatch, _gradOutputBatch);
}

void Convolutional::updateInputGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_gradOutputBatch);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        backwardKernel.setArg(6, i);
        _commandQueue.enqueue(backwardKernel, {inputGrad.size(1), inputGrad.size(2), inputGrad.size(3)});
    }
}

// TODO: use two kernels
void Convolutional::updateParamsGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    // weightsGrad, biasGrad
//    paramsGradKernel.setArg(0, weightsGrad);
//    paramsGradKernel.setArg(1, biasGrad);
//    paramsGradKernel.setArg(2, _gradOutputBatch);
//    paramsGradKernel.setArg(3, _inputBatch);
//
//    for (int i(0) ; i < (int)weights.size(0) ; i++)
//    {
//        paramsGradKernel.setArg(6, i);
//        paramsGradKernel.enqueue(_commandQueue, {inputGrad.size(1), inputGrad.size(2), inputGrad.size(3)});
//    }
}

void Convolutional::getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
{
    _params.push_back(&weights);
    _params.push_back(&bias);

    _paramsGrad.push_back(&weightsGrad);
    _paramsGrad.push_back(&biasGrad);
}


void Convolutional::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << weights.size(0) << "   " << weights.size(1) << "   " << weights.size(2) << "   " << weights.size(3) << std::endl;
    _file << bias.size(0) << "   " << bias.size(1) << "   " << bias.size(2) << std::endl;

    // Save weights
    for (unsigned i(0) ; i < weights.size(0) ; i++)
    {
        for (unsigned j(0) ; j < weights.size(1) ; j++)
        {
            for (unsigned k(0) ; k < weights.size(2) ; k++)
            {
                for (unsigned l(0) ; l < weights.size(3) ; l++)
                {
                    _file << weights({i, j, k, l}) << " ";
                }
                _file << std::endl;
            }
            _file << std::endl;
        }
        _file << std::endl;
    }

    // Save bias
    for (unsigned i(0) ; i < bias.size(0) ; i++)
    {
        for (unsigned j(0) ; j < bias.size(1) ; j++)
        {
            for (unsigned k(0) ; k < bias.size(2); k++)
            {
                _file << bias(i, j, k) << " ";
            }
            _file << std::endl;
        }
        _file << std::endl;
    }
}


void convGradInput(Tensor& inputGrad, const Tensor& kernel, const Tensor& gradOutput)
{
    #ifdef TENSOR_SAFE
        if (gradOutput.size(0) != kernel.size(0))
            std::cout << "convGradInput -> Kernel and source image don't have same the number of channels." << std::endl;
    #endif

    coords_t resSize{ kernel.size(1), gradOutput.size(1)+kernel.size(2)-1, gradOutput.size(2)+kernel.size(3)-1 };
    inputGrad.resize(resSize);

    for (unsigned k(0) ; k < inputGrad.size(0) ; k++)
    {
        for (unsigned i(0) ; i < inputGrad.size(1) ; i++)
        {
            for (unsigned j(0) ; j < inputGrad.size(2) ; j++)
            {
                inputGrad(k, i, j) = 0.0;

                for (unsigned u(0) ; u < kernel.size(2) ; u++)
                {
                    for (unsigned v(0) ; v < kernel.size(3) ; v++)
                    {
                        unsigned ii = i-kernel.size(2)+1 +u;
                        unsigned jj = j-kernel.size(3)+1 +v;

                        if (ii < gradOutput.size(1) && jj < gradOutput.size(2))
                        {
                            for (unsigned c(0) ; c < gradOutput.size(0) ; c++)
                                inputGrad(k, i, j) += kernel({c, k, u, v}) * gradOutput(c, ii, jj);
                        }
                    }
                }
            }
        }
    }
}

void convGradWeight(Tensor& weightsGrad, const Tensor& gradOutput, const Tensor& input)
{
    #ifdef TENSOR_SAFE
        if (input.size(0) != weightsGrad.size(1))
            std::cout << "convGradWeight -> Kernel and source image don't have same the number of channels." << std::endl;

        if (weightsGrad.size(0) != gradOutput.size(0) || weightsGrad.size(2) != input.size(1)-gradOutput.size(1)+1 || weightsGrad.size(3) != input.size(2)-gradOutput.size(2)+1)
            std::cout << "convGradWeight -> Result do not fit in tensor." << std::endl;
    #endif

    for (unsigned k(0) ; k < weightsGrad.size(0) ; k++)
    {
        for (unsigned c(0) ; c < weightsGrad.size(1) ; c++)
        {
            for (unsigned i(0) ; i < weightsGrad.size(2) ; i++)
            {
                unsigned shiftu = weightsGrad.size(2)-1-i;

                for (unsigned j(0) ; j < weightsGrad.size(3) ; j++)
                {
                    unsigned shiftv = weightsGrad.size(3)-1-j;

                    for (unsigned u(0) ; u < gradOutput.size(1) ; u++)
                        for (unsigned v(0) ; v < gradOutput.size(2) ; v++)
                            weightsGrad({k, c, i, j}) += gradOutput(k, u, v) * input(c, u+shiftu, v+shiftv);
                }
            }
        }
    }
}


}
