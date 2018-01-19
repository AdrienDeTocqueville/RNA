#include "RNA/Layers/MaxPooling.h"

#include <cfloat>
#include <fstream>

namespace rna
{

MaxPooling::MaxPooling(size_t _poolWidth, size_t _poolHeight):
    Layer("MaxPooling"),
    poolWidth(_poolWidth), poolHeight(_poolHeight)
{}

MaxPooling::MaxPooling(std::ifstream& _file):
    Layer("MaxPooling")
{
    _file >> poolWidth >> poolHeight;
}

#ifdef USE_OPENCL
void MaxPooling::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("Kernels/maxPooling.cl");

    forwardKernel.create(p, "feedForwardMaxPooling");
    backwardKernel.create(p, "backpropMaxPooling");

    forwardKernel.setArg(3, poolWidth);
    forwardKernel.setArg(4, poolHeight);
}

void MaxPooling::feedForward(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resize( {_inputBatch.size(0), _inputBatch.size(1), _inputBatch.size(2) / poolWidth, _inputBatch.size(3) / poolHeight} );
    output.openCL(_commandQueue.getContext());

    indices.resizeAs(output);
    indices.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1, indices);
    forwardKernel.setArg(2,_inputBatch);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        forwardKernel.setArg(5, i);
        _commandQueue.enqueueKernel(forwardKernel, {indices.size(1), indices.size(2), indices.size(3)});
    }
}

void MaxPooling::backprop(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _outputGradBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.openCL(_commandQueue.getContext());
    inputGrad.fill(0.0);
    inputGrad.writeBuffer(_commandQueue);

    // inputGrad
    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_outputGradBatch);
    backwardKernel.setArg(2, indices);
    backwardKernel.setArg(3, _inputBatch.size(0));

    cl_event event;
    _commandQueue.enqueueKernel(backwardKernel, {indices.size(1), indices.size(2), indices.size(3)}, &event);
    _commandQueue.enqueueBarrier({event});
}

#else
void MaxPooling::feedForward(const Tensor& _input)
{
    output.resize( {_input.size(0), _input.size(1) / poolWidth, _input.size(2) / poolHeight} );
    indices.resizeAs(output);

    for (unsigned c(0) ; c < output.size(0) ; c++)
    for (unsigned i(0) ; i < output.size(1) ; i++)
    {
        for (unsigned j(0) ; j < output.size(2) ; j++)
        {
            float maxInput = FLT_MIN;
            int maxIndex = -1;

            for (int u = 0 ; u < poolWidth ; ++u)
            {
                for (int v = 0 ; v < poolHeight ; ++v)
                {
                    int inputIndex = _input.getIndex({c, poolWidth*i + u, poolHeight*j + v});

                    if (_input[inputIndex] > maxInput)
                    {
                        maxInput = _input[inputIndex];
                        maxIndex = inputIndex;
                    }
                }
            }

            output(c, i, j) = maxInput;
            indices(c, i, j) = maxIndex;
        }
    }
}

void MaxPooling::backprop(const Tensor& _input, const Tensor& _outputGrad)
{
    inputGrad.resizeAs(_input);
    inputGrad.fill(0.0);

    for (unsigned c(0) ; c < indices.size(0) ; c++)
    for (unsigned i(0) ; i < indices.size(1) ; i++)
    {
        for (unsigned j(0) ; j < indices.size(2) ; j++)
        {
            inputGrad[indices(c, i, j)] = _outputGrad(c, i, j);
        }
    }
}
#endif // USE_OPENCL

void MaxPooling::saveToFile(std::ofstream& _file) const
{
    Layer::saveToFile(_file);

    _file << poolWidth  << "  " << poolHeight << std::endl;
}

}
