#include "RNA/Layers/Convolutional.h"
#include "Utility/Error.h"

#include <fstream>
#include <iostream>

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

#ifdef USE_OPENCL
void Convolutional::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/convolutional.cl");

    forwardKernel.create(p, "feedForwardConvolutional");
    backwardKernel.create(p, "backpropConvolutional");

    weightsGradKernel.create(p, "weightsGradConvolutional");
    biasGradKernel.create(p, "biasGradConvolutional");


    weights.openCL(_context);
    bias.openCL(_context);

    weightsGrad.openCL(_context);
    biasGrad.openCL(_context);


    forwardKernel.setArg(2, weights);
    forwardKernel.setArg(3, bias);
    forwardKernel.setArg(4, weights.size(1));
    forwardKernel.setArg(5, weights.size(2));
    forwardKernel.setArg(6, weights.size(3));

    backwardKernel.setArg(2, weights);
    backwardKernel.setArg(3, weights.size(0));
    backwardKernel.setArg(4, weights.size(2));
    backwardKernel.setArg(5, weights.size(3));

    weightsGradKernel.setArg(0, weightsGrad);
    weightsGradKernel.setArg(4, bias.size(0));
    weightsGradKernel.setArg(5, bias.size(1));
    weightsGradKernel.setArg(6, bias.size(2));

    biasGradKernel.setArg(0, biasGrad);
}

void Convolutional::releaseCL()
{
	Layer::releaseCL();

    weightsGradKernel.release();
    biasGradKernel.release();
}

void Convolutional::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resize({_inputBatch.size(0), weights.size(0), _inputBatch.size(2)-weights.size(2)+1, _inputBatch.size(3)-weights.size(3)+1});
    output.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        forwardKernel.setArg(7, i);
        _commandQueue.enqueueKernel(forwardKernel, bias.size());
    }
}

void Convolutional::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    // Start with paramsGrad to enqueue barrier on inputGrad
    updateParamsGrad(_commandQueue, _inputBatch, _outputGradBatch);
    updateInputGrad(_commandQueue, _inputBatch, _outputGradBatch);
}

void Convolutional::updateInputGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_outputGradBatch);

//    std::vector<cl_event> events(_inputBatch.size(0), nullptr);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        backwardKernel.setArg(6, i);
        _commandQueue.enqueueKernel(backwardKernel, {inputGrad.size(1), inputGrad.size(2), inputGrad.size(3)} /*, &events[i]*/ );
    }

//    _commandQueue.enqueueBarrier(events);
}

void Convolutional::updateParamsGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    // weightsGrad
    weightsGradKernel.setArg(1,_outputGradBatch);
    weightsGradKernel.setArg(2,_inputBatch);
    weightsGradKernel.setArg(3,_outputGradBatch.size(0));

    for (int i(0) ; i < (int)weights.size(0) ; i++)
    {
        weightsGradKernel.setArg(7, i);
        _commandQueue.enqueueKernel(weightsGradKernel, {weightsGrad.size(1), weightsGrad.size(2), weightsGrad.size(3)});
    }

    // biasGrad
    biasGradKernel.setArg(1,_outputGradBatch);
    biasGradKernel.setArg(2,_outputGradBatch.size(0));

    _commandQueue.enqueueKernel(biasGradKernel, biasGrad.size());
}

#else
void Convolutional::feedForward(const Tensor& _input)
{
    convolve(output, weights, _input);
    output += bias;
}

void Convolutional::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    convGradInput(inputGrad, weights, _outputGrad);

    convGradWeight(weightsGrad, _outputGrad, _input);
    biasGrad += _outputGrad;
}
#endif // USE_OPENCL

void Convolutional::setParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
{
    // biasGrad
        biasGrad = *_paramsGrad.back();
        _paramsGrad.pop_back();

    // weightsGrad
        weightsGrad = *_paramsGrad.back();
        _paramsGrad.pop_back();

    // bias
        bias = *_params.back();
        _params.pop_back();

    // weights
        weights = *_params.back();
        _params.pop_back();
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


#ifndef USE_OPENCL
void convGradInput(Tensor& inputGrad, const Tensor& kernel, const Tensor& outputGrad)
{
    #ifdef TENSOR_SAFE
        if (outputGrad.size(0) != kernel.size(0))
            std::cout << "convGradInput -> Kernel and source image don't have same the number of channels." << std::endl;
    #endif

    coords_t resSize{ kernel.size(1), outputGrad.size(1)+kernel.size(2)-1, outputGrad.size(2)+kernel.size(3)-1 };
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

                        if (ii < outputGrad.size(1) && jj < outputGrad.size(2))
                        {
                            for (unsigned c(0) ; c < outputGrad.size(0) ; c++)
                                inputGrad(k, i, j) += kernel({c, k, u, v}) * outputGrad(c, ii, jj);
                        }
                    }
                }
            }
        }
    }
}

void convGradWeight(Tensor& weightsGrad, const Tensor& outputGrad, const Tensor& input)
{
    #ifdef TENSOR_SAFE
        if (input.size(0) != weightsGrad.size(1))
            std::cout << "convGradWeight -> Kernel and source image don't have same the number of channels." << std::endl;

        if (weightsGrad.size(0) != outputGrad.size(0) || weightsGrad.size(2) != input.size(1)-outputGrad.size(1)+1 || weightsGrad.size(3) != input.size(2)-outputGrad.size(2)+1)
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

                    for (unsigned u(0) ; u < outputGrad.size(1) ; u++)
                        for (unsigned v(0) ; v < outputGrad.size(2) ; v++)
                            weightsGrad({k, c, i, j}) += outputGrad(k, u, v) * input(c, u+shiftu, v+shiftv);
                }
            }
        }
    }
}
#endif // USE_OPENCL

}
