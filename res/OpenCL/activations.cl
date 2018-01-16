/// TanH
__kernel void feedForwardTanh(__global float* _output, __global float* _input, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _output[index + k] = tanh(_input[index + k]);
}

__kernel void backpropTanh(__global float* _inputGrad, __global float* _input, __global float* _output, __global float* _outputGrad, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _inputGrad[index + k] = (1.0f - _output[index + k]*_output[index + k]) * _outputGrad[index + k];
}


/// ReLU
__kernel void feedForwardReLU(__global float* _output, __global float* _input, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _output[index + k] = max(_input[index + k], 0.0f);
}

__kernel void backpropReLU(__global float* _inputGrad, __global float* _input, __global float* _output, __global float* _outputGrad, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _inputGrad[index + k] = step(0.0f, _input[index + k]) * _outputGrad[index + k];
}


/// ELU
__kernel void feedForwardELU(__global float* _output, __global float* _input, int _inputWidth, float _alpha)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _output[index + k] = _input[index + k] < 0.0f? _alpha * (exp(_input[index + k])-1.0f): _input[index + k];
}

__kernel void backpropELU(__global float* _inputGrad, __global float* _input, __global float* _output, __global float* _outputGrad, int _inputWidth, float _alpha)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _inputGrad[index + k] = (_output[index + k] < 0.0f? _output[index + k] + _alpha: 1.0f) * _outputGrad[index + k];
}
