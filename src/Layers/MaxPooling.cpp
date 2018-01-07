#include "MaxPooling.h"

namespace rna
{

void MaxPooling::openCL(cl::Context& _context)
{
    auto& p = _context.getProgram("res/OpenCL/maxPooling.cl");

    forwardKernel.create(p, "feedForwardMaxPooling");
    backwardKernel.create(p, "backpropMaxPooling");
}

void MaxPooling::feedForwardCPU(const Tensor& _input)
{
    output.resize( {_input.size(0), _input.size(1) / 2, _input.size(2) / 2} );
    indices.resizeAs(output);

    Tensor pool{4};

    for (unsigned c(0) ; c < output.size(0) ; c++)
    for (unsigned i(0) ; i < output.size(1) ; i++)
    {
        for (unsigned j(0) ; j < output.size(2) ; j++)
        {
            pool(0) = _input(c, 2 * i, 2 * j);
            pool(1) = _input(c, 2 * i, 2 * j + 1);
            pool(2) = _input(c, 2 * i + 1, 2 * j);
            pool(3) = _input(c, 2 * i + 1, 2 * j + 1);

            size_t am = pool.argmax()[0];

            indices(c, i, j) = am;
            output(c, i, j) = pool[am];
        }
    }
}

void MaxPooling::feedForwardCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch)
{
    output.resize( {_inputBatch.size(0), _inputBatch.size(1), _inputBatch.size(2) / 2, _inputBatch.size(3) / 2} );
    output.openCL(_commandQueue.getContext());

    indices.resizeAs(output);
    indices.openCL(_commandQueue.getContext());

    forwardKernel.setArg(0, output);
    forwardKernel.setArg(1, indices);
    forwardKernel.setArg(2,_inputBatch);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        forwardKernel.setArg(3, i);
        forwardKernel.enqueue(_commandQueue, {indices.size(1), indices.size(2), indices.size(3)});
    }
}

void MaxPooling::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    inputGrad.resizeAs(_input);
    inputGrad.fill(0.0);


    for (unsigned c(0) ; c < indices.size(0) ; c++)
    for (unsigned i(0) ; i < indices.size(1) ; i++)
    {
        for (unsigned j(0) ; j < indices.size(2) ; j++)
        {
            coords_t coord = {c, i, j};
            if (indices(c, i, j) == 1)
                coord[2]++;
            else if (indices(c, i, j) == 2)
                coord[1]++;
            else if (indices(c, i, j) == 3)
            {
                coord[1]++;
                coord[2]++;
            }

            inputGrad(coord) = _gradOutput(c, i, j);
        }
    }
}

void MaxPooling::backpropCL(cl::CommandQueue& _commandQueue, const Tensor& _inputBatch, const Tensor& _gradOutputBatch)
{
    inputGrad.resizeAs(_inputBatch);
    inputGrad.fill(0.0);
    inputGrad.openCL(_commandQueue.getContext());

    // inputGrad
    backwardKernel.setArg(0, inputGrad);
    backwardKernel.setArg(1,_gradOutputBatch);
    backwardKernel.setArg(2, indices);

    for (int i(0) ; i < (int)_inputBatch.size(0) ; i++)
    {
        backwardKernel.setArg(3, i);
        backwardKernel.enqueue(_commandQueue, {indices.size(1), indices.size(2), indices.size(3)});
    }
}

}
