__kernel void feedForwardDropout(__global float* _output, __global float* _input, int _inputWidth, __global float* _rands, float rate)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _output[index + k] = _rands[index + k] < rate? 0.0f: _input[index + k];
}

__kernel void backpropDropout(__global float* _inputGrad, __global float* _output, __global float* _outputGrad, int _inputWidth)
{
    const int index = get_global_id(0)*_inputWidth;

    for (int k = 0; k < _inputWidth; k++)
        _inputGrad[index + k] = (_output[index + k] == 0.0f? 0.0f: 1.0f) * _outputGrad[index + k];
}
