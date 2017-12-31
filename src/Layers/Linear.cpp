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

    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
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


    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

void Linear::randomize()
{
	// TODO: Init weights with parametrable values
    weights.randomize(Layer::WEIGHT_INIT_MIN, Layer::WEIGHT_INIT_MAX);
    bias.randomize(Layer::BIAS_INIT_MIN, Layer::BIAS_INIT_MAX);
}

void Linear::openCL(cl::ContextWrapper& _context)
{
    auto& p = _context.getProgram("res/OpenCL/linear.cl");

    forwardKernel.create(p, "feedForwardLinear");
    backwardKernel.create(p, "backpropLinear");
    paramsGradKernel.create(p, "paramsGradLinear");

    weights.openCL(_context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    bias.openCL(_context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    gradWeight.openCL(_context());
    gradBias.openCL(_context());


    forwardKernel.setArg(2, weights);
    backwardKernel.setArg(2, weights);

    forwardKernel.setArg(3, bias);

    paramsGradKernel.setArg(0, gradWeight);
    paramsGradKernel.setArg(1, gradBias);
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

void Linear::feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resize({_inputBatch.size(0), bias.size(0)});
    output.openCL(context);

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);
    forwardKernel.setArg(4,_inputBatch.size(1));

    forwardKernel.enqueue(_commandQueue, { output.size(0), output.size(1) });
	output.readBuffer(_commandQueue);
}

void Linear::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(gradInput, weights.getTranspose(), _gradOutput);

        gradWeight.addOuterProduct(_gradOutput, _input);
        gradBias += _gradOutput;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmm(gradInput, _gradOutput, weights);

        Tensor temp; mulmtm(temp, _gradOutput, _input);
        gradWeight += temp;

        for (unsigned i(0) ; i < _gradOutput.size(0) ; i++)
            for (unsigned j(0) ; j < _gradOutput.size(1) ; j++)
                gradBias(j) += _gradOutput(i, j);
    }
}

void Linear::backpropCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    gradInput.resizeAs({_gradOutputBatch.size(0), weights.size(1)});
    gradInput.openCL(context);

    // gradInput
    backwardKernel.setArg(0, gradInput);
    backwardKernel.setArg(1,_gradOutputBatch);
    backwardKernel.setArg(3,_gradOutputBatch.size(1));

    backwardKernel.enqueue(_commandQueue, { gradInput.size(0), gradInput.size(1) });
	gradInput.readBuffer(_commandQueue);

    // gradWeight, gradBias
    paramsGradKernel.setArg(2, _gradOutputBatch);
    paramsGradKernel.setArg(3, _inputBatch);
    paramsGradKernel.setArg(4, _gradOutputBatch.size(0));
    paramsGradKernel.setArg(5, _inputBatch.size(1));

    backwardKernel.enqueue(_commandQueue, { _gradOutputBatch.size(1) });
	gradWeight.readBuffer(_commandQueue);
	gradBias.readBuffer(_commandQueue);
}

void Linear::zeroParametersGradients()
{
    gradWeight.fill(0.0);
    gradBias.fill(0.0);
}

void Linear::updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia)
{
    deltaWeight = _inertia * deltaWeight - _learningRate * gradWeight;
    deltaBias = _inertia * deltaBias - _learningRate * gradBias;

    weights += deltaWeight;
    bias    += deltaBias;
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
