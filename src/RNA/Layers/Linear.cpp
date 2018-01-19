#include "RNA/Layers/Linear.h"
#include "Utility/Error.h"

#include <fstream>

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

#ifdef USE_OPENCL
void Linear::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/linear.cl");

    forwardKernel.create(p, "feedForwardLinear");
    backwardKernel.create(p, "backpropLinear");

    weightsGradKernel.create(p, "weightsGradLinear");
    biasGradKernel.create(p, "biasGradLinear");

    weights.openCL(_context);
    bias.openCL(_context);

    weightsGrad.openCL(_context);
    biasGrad.openCL(_context);


    forwardKernel.setArg(2, weights);
    forwardKernel.setArg(3, bias);
    forwardKernel.setArg(4, weights.size(1));

    backwardKernel.setArg(2, weights);
    backwardKernel.setArg(3, weights.size(0));

    weightsGradKernel.setArg(0, weightsGrad);
    biasGradKernel.setArg(0, biasGrad);
}

void Linear::releaseCL()
{
	Layer::releaseCL();

    weightsGradKernel.release();
    biasGradKernel.release();
}

void Linear::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resize({_inputBatch.size(0), bias.size(0)});
    output.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1,_inputBatch);

    _commandQueue.enqueueKernel(forwardKernel, output.size());
}

void Linear::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    // Start with paramsGrad to enqueue barrier on inputGrad
    updateParamsGrad(_commandQueue, _inputBatch, _outputGradBatch);
    updateInputGrad(_commandQueue, _inputBatch, _outputGradBatch);
}

void Linear::updateInputGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs({_outputGradBatch.size(0), weights.size(1)});
    inputGrad.openCL(_commandQueue.getContext());

    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_outputGradBatch);

    cl_event event;
    _commandQueue.enqueueKernel(backwardKernel, inputGrad.size(), &event);
    _commandQueue.enqueueBarrier({event});
}

void Linear::updateParamsGrad(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    // weightsGrad
    weightsGradKernel.setArg(1,_outputGradBatch);
    weightsGradKernel.setArg(2,_inputBatch);
    weightsGradKernel.setArg(3,_outputGradBatch.size(0));

    _commandQueue.enqueueKernel(weightsGradKernel, weightsGrad.size());

    // biasGrad
    biasGradKernel.setArg(1,_outputGradBatch);
    biasGradKernel.setArg(2,_outputGradBatch.size(0));

    _commandQueue.enqueueKernel(biasGradKernel, biasGrad.size());
}

#else
void Linear::feedForward(const Tensor& _input)
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

void Linear::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    if (_input.nDimensions() == 1)
    {
        mulmv(inputGrad, weights.getTranspose(), _outputGrad);

        weightsGrad.addOuterProduct(_outputGrad, _input);
        biasGrad += _outputGrad;
    }
    else if (_input.nDimensions() == 2)
    {
        mulmm(inputGrad, _outputGrad, weights);

        Tensor temp; mulmtm(temp, _outputGrad, _input);
        weightsGrad += temp;

        for (unsigned i(0) ; i < _outputGrad.size(0) ; i++)
            for (unsigned j(0) ; j < _outputGrad.size(1) ; j++)
                biasGrad(j) += _outputGrad(i, j);
    }
}
#endif // USE_OPENCL

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
