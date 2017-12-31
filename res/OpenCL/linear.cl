__kernel void feedForwardLinear(__global float* _output, __global float* _input, __global float* _weights, __global float* _bias, int _inputWidth)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    float value = 0.0f;
    for (int k = 0; k < _inputWidth; ++k)
    {
        float weight = _weights[ty * _inputWidth + k];
        float input = _input[tx * _inputWidth + k];
        value += weight * input;
    }

    _output[tx * get_global_size(1) + ty] = value + _bias[ty];
}

__kernel void backpropLinear(__global float* _gradInput, __global float* _gradOutput, __global float* _weights, int _gradOutputWidth)
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

__kernel void paramsGradLinear(__global float* _gradWeight, __global float* _gradBias, __global float* _gradOutput, __global float* _input, int _gradOutputHeight, int _inputWidth)
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
