float dtanh(float _x)
{
    float t = tanh(_x);
    return 1.0f - t*t;
}


__kernel void feedForwardTanh(__global float* _output, __global float* _input, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _output[index + k] = tanh(_input[index + k]);
}

__kernel void backpropTanh(__global float* _inputGrad, __global float* _input, __global float* _gradOutput, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _inputGrad[index + k] = dtanh(_input[index + k]) * _gradOutput[index + k];
}


__kernel void feedForwardReLU(__global float* _output, __global float* _input, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _output[index + k] = max(_input[index + k], 0.0f);
}

__kernel void backpropReLU(__global float* _inputGrad, __global float* _input, __global float* _gradOutput, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _inputGrad[index + k] = step(0.0f, _input[index + k]) * _gradOutput[index + k];
}
