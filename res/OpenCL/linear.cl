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

__kernel void backpropLinear(__global float* _inputGrad, __global float* _outputGrad, __global float* _weights, int _outputGradWidth)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    float value = 0.0f;
    for (int k = 0; k < _outputGradWidth; ++k)
    {
        float weight = _weights[k * get_global_size(1) + ty];
        float outputGrad = _outputGrad[tx * _outputGradWidth + k];
        value += weight * outputGrad;
    }

    _inputGrad[tx * get_global_size(1) + ty] = value;
}

__kernel void paramsGradLinear(__global float* _weightsGrad, __global float* _biasGrad, __global float* _outputGrad, __global float* _input, int _batchSize, int _inputWidth)
{
    const int j = get_global_id(0);

    for (int i = 0; i < _batchSize; ++i)
    {
        float outputGrad = _outputGrad[i * get_global_size(0) + j];

        for (int k = 0; k < _inputWidth; ++k)
            _weightsGrad[j * _inputWidth + k] += outputGrad * _input[i * _inputWidth + k];

        _biasGrad[j] += outputGrad;
    }
}
