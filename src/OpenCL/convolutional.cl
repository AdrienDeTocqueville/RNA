__kernel void convolutionalForward(__global float* _output, __global float* _input, __global float* _kernel, __global float* _bias, int _inputChannels, int _kernelWidth, int _kernelHeight, int _batch)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    int biasIndex = tc * get_global_size(1)*get_global_size(2) + tx * get_global_size(2) + ty;
    int outputIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2) + biasIndex;

    float value = 0.0f;

    int mu = _kernelWidth-1;
    int mv = _kernelHeight-1;

    int inputWidth = get_global_size(1)+mu;
    int inputHeight = get_global_size(2)+mv;

    for (int c = 0; c < _inputChannels; ++c)
    {
        for (int u = 0; u < _kernelWidth; ++u)
        {
            for (int v = 0; v < _kernelHeight; ++v)
            {
                float weight = _kernel[tc*_inputChannels*_kernelWidth*_kernelHeight + c*_kernelWidth*_kernelHeight + (mu-u)*_kernelHeight + (mv-v)];
                float input = _input[_batch*_inputChannels*inputWidth*inputHeight + c*inputWidth*inputHeight + (tx+u)*inputHeight + (ty+v)];

                value += weight * input;
            }
        }
    }

    _output[outputIndex] = value + _bias[biasIndex];
}

__kernel void convolutionalBackward(__global float* _gradInput, __global float* _gradOutput, __global float* _weights, int _gradOutputWidth)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    float value = 0.0f;
    for (int k = 0; k < _gradOutputWidth; ++k)
    {
        float weight = _weights[k * get_global_size(1) + ty];
        float gradOutput = _gradOutput[tx * _gradOutputWidth + k];
        value += weight * gradOutput;
    }

    _gradInput[tx * get_global_size(1) + ty] = value;
}

__kernel void convolutionalParametersGradients(__global float* _gradWeight, __global float* _gradBias, __global float* _gradOutput, __global float* _input, int _gradOutputHeight, int _inputWidth)
{
    const int j = get_global_id(0);

    for (int i = 0; i < _gradOutputHeight; ++i)
    {
        float gradOutput = _gradOutput[i * get_global_size(0) + j];

        for (int k = 0; k < _inputWidth; ++k)
            _gradWeight[j * _inputWidth + k] += gradOutput * _input[i * _inputWidth + k];

        _gradBias[j] += gradOutput;
    }
}

/*
__kernel void convolve(__global float* _output, __global float* _input, __global float* _weights, const int _inputWidth, const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * _inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
			sum += _weights[(r * maskWidth)  + c] * input[idxIntmp + c];
        }
    }

	output[y * get_global_size(0) + x] = sum;
}
*/
