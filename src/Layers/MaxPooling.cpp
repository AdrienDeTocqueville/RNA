#include "MaxPooling.h"

namespace rna
{

void MaxPooling::openCL(const cl_context& _context, const cl_device_id& _deviceId)
{
    if (!kernelForward)
        kernelForward = loadKernel(_context, _deviceId, "src/OpenCL/maxPooling.cl", "maxPoolingForward");

    if (!kernelBackward)
        kernelBackward = loadKernel(_context, _deviceId, "src/OpenCL/maxPooling.cl", "maxPoolingBackward");
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

void MaxPooling::feedForwardCL(const cl_command_queue& _commandQueue, const Tensor& _inputBatch)
{
    cl_context context;
    clGetCommandQueueInfo(_commandQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);

    output.resize( {_inputBatch.size(0), _inputBatch.size(1) / 2, _inputBatch.size(2) / 2} );
    indices.resizeAs(output);

    output.openCL(context);
    indices.openCL(context);

    clSetKernelArg(kernelForward, 0, sizeof(cl_mem), &output.getBuffer());
    clSetKernelArg(kernelForward, 1, sizeof(cl_mem), &indices.getBuffer());
    clSetKernelArg(kernelForward, 2, sizeof(cl_mem), &_inputBatch.getBuffer());

    execKernel(_commandQueue, kernelForward, output.size());
	output.readBuffer(_commandQueue);
	indices.readBuffer(_commandQueue);
}

void MaxPooling::backpropCPU(const Tensor& _input, const Tensor& _gradOutput)
{
    gradInput.resizeAs(_input);
    gradInput.fill(0.0);


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

            gradInput(coord) = _gradOutput(c, i, j);
        }
    }
}

}
