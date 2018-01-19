__kernel void feedForwardLogSoftMax(__global float* _output, __global float* _input, int _inputWidth)
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

__kernel void backpropLogSoftMax(__global float* _inputGrad, __global float* _input, __global float* _outputGrad, __global float* _output, int _outputGradWidth)
{
    const int start = get_global_id(0)*_outputGradWidth;

    float sum = 0.0f;
    for (int i = 0; i < _outputGradWidth; i++)
        sum += _outputGrad[start+i];

    for (int i = 0; i < _outputGradWidth; i++)
        _inputGrad[start+i] = _outputGrad[start+i] - exp(_output[start+i])*sum;
}
