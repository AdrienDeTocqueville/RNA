__kernel void logSoftMax(__global float* _output, __global float* _input, int _inputWidth)
{
    int i = get_global_id(0);

    float logSum = 0.0f;
    float maxInput = _input[i*_inputWidth];

    for (int j = 1; j < _inputWidth; j++)
        maxInput = max(_input[i*_inputWidth +j], maxInput);

    for (int j = 0; j < _inputWidth; j++)
       logSum += exp(_input[i*_inputWidth +j] - maxInput);

    logSum = maxInput + log(logSum);

    for (int j = 0; j < _inputWidth; j++)
        _output[i*_inputWidth +j] = _input[i*_inputWidth +j] - logSum;
}
