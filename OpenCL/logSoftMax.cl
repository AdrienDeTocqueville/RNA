__kernel void logSoftMaxForward(__global float* _output, __global float* _input, int _inputWidth)
{
    const int start = get_global_id(0)*_inputWidth;

    float logSum = 0.0f;
    float maxInput = _input[start];

    for (int i = 1; i < _inputWidth; i++)
        maxInput = max(_input[start+i], maxInput);

    for (int i = 0; i < _inputWidth; i++)
       logSum += exp(_input[start+i] - maxInput);

    logSum = maxInput + log(logSum);

    for (int i = 0; i < _inputWidth; i++)
        _output[start+i] = _input[start+i] - logSum;
}

__kernel void logSoftMaxBackward(__global float* _gradInput, __global float* _input, __global float* _gradOutput, __global float* _output, int _gradOutputWidth)
{
    const int start = get_global_id(0)*_gradOutputWidth;

    float sum = 0.0f;
    for (int i = 0; i < _gradOutputWidth; i++)
        sum += _gradOutput[start+i];

    for (int i = 0; i < _gradOutputWidth; i++)
        _gradInput[start+i] = _gradOutput[start+i] - exp(_output[start+i])*sum;
}
