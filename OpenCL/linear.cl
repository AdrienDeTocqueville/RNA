__kernel void linear(__global float* _output, __global float* _input, __global float* _weights, __global float* _bias, int _inputWidth, int _weightsHeight)
{
    int tx = get_global_id(0);
    int ty = get_global_id(1);

    float value = 0.0f;
    for (int k = 0; k < _inputWidth; ++k)
    {
        float input = _input[tx * _inputWidth + k];
        float weight = _weights[ty * _inputWidth + k];
        value += input * weight;
    }

    _output[tx * _weightsHeight + ty] = value + _bias[ty];
}
