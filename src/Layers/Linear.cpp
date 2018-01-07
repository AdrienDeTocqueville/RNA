#include "Linear.h"

#include <fstream>

#include "../Utility/Error.h"

namespace rna
{

Linear::Linear(size_t _inputSize, size_t _outputSize):
    Layer("Linear"),
    weights{_outputSize, _inputSize}, bias{_outputSize}
{
    randomize();

    weightsGrad.resizeAs(weights);
    biasGrad.resizeAs(bias);
}

Linear::Linear(std::ifstream& _file):
    Layer("Linear")
{
    size_t inputSize, outputSize;
    _file >> inputSize >> outputSize;

    // Load weights
    weights.resize({outputSize, inputSize});
    for (unsigned i(0) ; i < outputSize ; i++)
        for (unsigned j(0) ; j < inputSize ; j++)
            _file >> weights(i, j);

    // Load bias
    bias.resize({outputSize});
    for (unsigned i(0) ; i < outputSize ; i++)
        _file >> bias(i);


    weightsGrad.resizeAs(weights);
    biasGrad.resizeAs(bias);
}

void Linear::randomize()
{
    weights.randomize(Layer::WEIGHT_INIT_MIN, Layer::WEIGHT_INIT_MAX);
    bias.randomize(Layer::BIAS_INIT_MIN, Layer::BIAS_INIT_MAX);
}

void Linear::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/linear.cl");

    forwardKernel.create(p, "feedForwardLinear");
    backwardKernel.create(p, "backpropLinear");
    paramsGradKernel.create(p, "paramsGradLinear");

    weights.openCL(_context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    bias.openCL(_context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    weightsGrad.openCL(_context());
    biasGrad.openCL(_context());


    forwardKernel.setArg(2, weights);
    forwardKernel.setArg(3, bias);
    forwardKernel.setArg(4, weights.size(1));

    backwardKernel.setArg(2, weights);
    backwardKernel.setArg(3, weights.size(0));

    paramsGradKernel.setArg(0, weightsGrad);
    paramsGradKernel.setArg(1, biasGrad);
}

void Linear::releaseCL()
{
	Layer::releaseCL();

    paramsGradKernel.release();
}

void Linear::feedForwardCPU(const Tensor& _input)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(output, weights, _input);
        output += bias;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmmt(output, _input, weights);

        for (unsigned i(0); i < output.size(0); i++)
            for (unsigned j(0); j < output.size(1); j++)
                output(i, j) += bias(j);
    }
}

void Linear::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resize({_inputBatch.size(0), bias.size(0)});
    output.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);

    _commandQueue.enqueue(forwardKernel, output.size());
}

void Linear::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(inputGrad, weights.getTranspose(), _gradOutput);

        weightsGrad.addOuterProduct(_gradOutput, _input);
        biasGrad += _gradOutput;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmm(inputGrad, _gradOutput, weights);

        Tensor temp; mulmtm(temp, _gradOutput, _input);
        weightsGrad += temp;

        for (unsigned i(0) ; i < _gradOutput.size(0) ; i++)
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                biasGrad(j) += _gradOutput(i, j);
    }
}

void Linear::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    updateInputGrad(_commandQueue, _inputBatch, _gradOutputBatch);
    updateParamsGrad(_commandQueue, _inputBatch, _gradOutputBatch);
}

void Linear::updateInputGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    inputGrad.resizeAs({_gradOutputBatch.size(0), weights.size(1)});
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_gradOutputBatch);

    _commandQueue.enqueue(backwardKernel, inputGrad.size());
}

// TODO: use two kernels
void Linear::updateParamsGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    paramsGradKernel.setArg(2, _gradOutputBatch);
    paramsGradKernel.setArg(3, _inputBatch);
    paramsGradKernel.setArg(4, _inputBatch.size(0));
    paramsGradKernel.setArg(5,  weights.size(1));

    _commandQueue.enqueue(backwardKernel, { weights.size(0) });
}

void Linear::getParams(std::vector<Tensor*>& _params, std::vector<Tensor*>& _paramsGrad)
{
    _params.push_back(&weights);
    _params.push_back(&bias);

    _paramsGrad.push_back(&weightsGrad);
    _paramsGrad.push_back(&biasGrad);
}

void Linear::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << weights.size(1) << "   " << weights.size(0) << std::endl;

    // Save weights
    for (unsigned i(0) ; i < weights.size(0) ; i++)
    {
        for (unsigned j(0) ; j < weights.size(1) ; j++)
            _file << weights(i, j) << " ";

        _file << std::endl;
    }

    // Save bias
    for (unsigned i(0) ; i < bias.size(0) ; i++)
        _file << bias(i) << " ";

    _file << std::endl;
}

}
