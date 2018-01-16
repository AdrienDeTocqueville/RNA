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

__kernel void weightsGradLinear(__global float* _weightsGrad, __global float* _outputGrad, __global float* _input, int _numBatches)
{
    const int tx = get_global_id(0);
    const int ty = get_global_id(1);

    float value = 0.0f;
    for (int batch = 0; batch < _numBatches; batch++)
    {
        float outputGrad = _outputGrad[batch * get_global_size(0) + tx];
        float input = _input[batch * get_global_size(1) + ty];

        value += outputGrad * input;
    }

    _weightsGrad[tx * get_global_size(1) + ty] += value;
}

__kernel void biasGradLinear(__global float* _biasGrad, __global float* _outputGrad, int _numBatches)
{
    const int j = get_global_id(0);

    float value = 0.0f;
    for (int batch = 0; batch < _numBatches; batch++)
        value += _outputGrad[batch * get_global_size(0) + j];

    _biasGrad[j] += value;
}
