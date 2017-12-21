#include "Convolutional.h"

#include <fstream>
#include <iostream>

#include "../Utility/Error.h"

namespace rna
{

Convolutional::Convolutional(coords_t inputDimensions, coords_t kernelDimensions, size_t _outputChannels):
    Layer("Convolutional"),
    weights{_outputChannels, inputDimensions[0], kernelDimensions[0], kernelDimensions[1]},
    bias{_outputChannels, inputDimensions[1]-kernelDimensions[0]+1, inputDimensions[2]-kernelDimensions[1]+1},
    kernelGradParam(0)
{
    randomize();

    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

Convolutional::Convolutional(std::ifstream& _file):
    Layer("Convolutional"),
    kernelGradParam(0)
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


    gradWeight.resizeAs(weights);
    gradBias.resizeAs(bias);

    deltaWeight.resizeAs(weights, 0.0);
    deltaBias.resizeAs(bias, 0.0);
}

void Convolutional::randomize()
{
    weights.randomize(-1.0, 1.0);
    bias.randomize(-1.0, 1.0);
}

void Convolutional::openCL(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!kernelForward)
        kernelForward = loadKernel(_context, _deviceId, "src/OpenCL/convolutional.cl", "convolutionalForward");

    if (!kernelBackward)
        kernelBackward = loadKernel(_context, _deviceId, "src/OpenCL/convolutional.cl", "convolutionalBackward");

    if (!kernelGradParam)
        kernelGradParam = loadKernel(_context, _deviceId, "src/OpenCL/convolutional.cl", "convolutionalParametersGradients");

    weights.openCL(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    bias.openCL(_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    gradWeight.openCL(_context);
    gradBias.openCL(_context);


    cl_int inputChannels = weights.size(1);
    cl_int weightWidth = weights.size(2);
    cl_int weightHeight = weights.size(3);

    clSetKernelArg(kernelForward, 2, sizeof(cl_mem), &weights.getBuffer());
    clSetKernelArg(kernelForward, 3, sizeof(cl_mem), &bias.getBuffer());
    clSetKernelArg(kernelForward, 4, sizeof(cl_int), &inputChannels);
    clSetKernelArg(kernelForward, 5, sizeof(cl_int), &weightWidth);
    clSetKernelArg(kernelForward, 6, sizeof(cl_int), &weightHeight);
}

void Convolutional::releaseCL()
{
	Layer::releaseCL();

    clReleaseKernel(kernelGradParam); kernelGradParam = 0;
}

void Convolutional::feedForwardCPU(const Tensor& _input)
{
    convolve(output, weights, _input);
    output += bias;
}

void Convolutional::feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resize({_inputBatch.size(0), weights.size(0), _inputBatch.size(2)-weights.size(2)+1, _inputBatch.size(3)-weights.size(3)+1});
    output.openCL(context);

    clSetKernelArg(kernelForward, 0, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelForward, 1, sizeof(cl_mem), &_inputBatch.getBuffer());

    for (cl_int i(0) ; i < _inputBatch.size(0) ; i++)
    {
        clSetKernelArg(kernelForward, 7, sizeof(cl_int), &i);
        execKernel(_commandQueue, kernelForward, bias.size());
    }

	output.readBuffer(_commandQueue);
}

void Convolutional::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    convGradInput(gradInput, weights, _gradOutput);

    convGradWeight(gradWeight, _gradOutput, _input);
    gradBias += _gradOutput;
}

void Convolutional::zeroParametersGradients()
{
    gradWeight.fill(0.0);
    gradBias.fill(0.0);
}

void Convolutional::updateParameters(Tensor::value_type _learningRate, Tensor::value_type _inertia)
{
    deltaWeight = _inertia * deltaWeight - _learningRate * gradWeight;
    deltaBias = _inertia * deltaBias - _learningRate * gradBias;

    weights += deltaWeight;
    bias    += deltaBias;
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


void convGradInput(Tensor& gradInput, const Tensor& kernel, const Tensor& gradOutput)
{
    #ifdef TENSOR_SAFE
        if (gradOutput.size(0) != kernel.size(0))
            std::cout << "convGradInput -> Kernel and source image don't have same the number of channels." << std::endl;
    #endif

    coords_t resSize{ kernel.size(1), gradOutput.size(1)+kernel.size(2)-1, gradOutput.size(2)+kernel.size(3)-1 };
    gradInput.resize(resSize);

    for (unsigned k(0) ; k < gradInput.size(0) ; k++)
    {
        for (unsigned i(0) ; i < gradInput.size(1) ; i++)
        {
            for (unsigned j(0) ; j < gradInput.size(2) ; j++)
            {
                gradInput(k, i, j) = 0.0;

                for (unsigned u(0) ; u < kernel.size(2) ; u++)
                {
                    for (unsigned v(0) ; v < kernel.size(3) ; v++)
                    {
                        unsigned ii = i-kernel.size(2)+1 +u;
                        unsigned jj = j-kernel.size(3)+1 +v;

                        if (ii < gradOutput.size(1) && jj < gradOutput.size(2))
                        {
                            for (unsigned c(0) ; c < gradOutput.size(0) ; c++)
                                gradInput(k, i, j) += kernel({c, k, u, v}) * gradOutput(c, ii, jj);
                        }
                    }
                }
            }
        }
    }
}

void convGradWeight(Tensor& gradWeight, const Tensor& gradOutput, const Tensor& input)
{
    #ifdef TENSOR_SAFE
        if (input.size(0) != gradWeight.size(1))
            std::cout << "convGradWeight -> Kernel and source image don't have same the number of channels." << std::endl;

        if (gradWeight.size(0) != gradOutput.size(0) || gradWeight.size(2) != input.size(1)-gradOutput.size(1)+1 || gradWeight.size(3) != input.size(2)-gradOutput.size(2)+1)
            std::cout << "convGradWeight -> Result do not fit in tensor." << std::endl;
    #endif

    for (unsigned k(0) ; k < gradWeight.size(0) ; k++)
    {
        for (unsigned c(0) ; c < gradWeight.size(1) ; c++)
        {
            for (unsigned i(0) ; i < gradWeight.size(2) ; i++)
            {
                unsigned shiftu = gradWeight.size(2)-1-i;

                for (unsigned j(0) ; j < gradWeight.size(3) ; j++)
                {
                    unsigned shiftv = gradWeight.size(3)-1-j;

                    for (unsigned u(0) ; u < gradOutput.size(1) ; u++)
                        for (unsigned v(0) ; v < gradOutput.size(2) ; v++)
                            gradWeight({k, c, i, j}) += gradOutput(k, u, v) * input(c, u+shiftu, v+shiftv);
                }
            }
        }
    }
}


}
