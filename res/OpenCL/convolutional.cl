__kernel void feedForwardConvolutional(__global float* _output, __global float* _input, __constant float* _kernel, __constant float* _bias, int _inputChannels, int _kernelWidth, int _kernelHeight, int _batch)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    unsigned mu = _kernelWidth-1;
    unsigned mv = _kernelHeight-1;

    unsigned inputWidth = get_global_size(1)+mu;
    unsigned inputHeight = get_global_size(2)+mv;

    int biasIndex = tc * get_global_size(1)*get_global_size(2) + tx * get_global_size(2) + ty;
    int outputIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2) + biasIndex;


    float value = 0.0f;

    for (int c = 0; c < _inputChannels; ++c)
    {
        for (int u = 0; u < _kernelWidth; ++u)
        {
            for (int v = 0; v < _kernelHeight; ++v)
            {
                float weight = _kernel[tc*_inputChannels*_kernelWidth*_kernelHeight + c*_kernelWidth*_kernelHeight + (mu-u)*_kernelHeight + (mv-v)];
                float input = _input[_batch*_inputChannels*inputWidth*inputHeight + c*inputWidth*inputHeight + (tx+u)*inputHeight + (ty+v)];

                value += weight * input;
            }
        }
    }

    _output[outputIndex] = value + _bias[biasIndex];
}

__kernel void backpropConvolutional(__global float* _inputGrad, __global float* _gradOutput, __global float* _kernel, int _outputChannels, int _kernelWidth, int _kernelHeight, int _batch)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    unsigned mu = _kernelWidth-1;
    unsigned mv = _kernelHeight-1;

    unsigned outputWidth = get_global_size(1)-mu;
    unsigned outputHeight = get_global_size(2)-mv;

    int inputBatchIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2);
    int outputBatchIndex = _batch * _outputChannels*outputWidth*outputHeight;

    int inputIndex = inputBatchIndex + tc * get_global_size(1)*get_global_size(2) + tx * get_global_size(2) + ty;


    float value = 0.0f;

    for (int u = 0; u < _kernelWidth; ++u)
    {
        for (int v = 0; v < _kernelHeight; ++v)
        {
            unsigned i = tx-mu +u;
            unsigned j = ty-mv +v;

            if (i < outputWidth && j < outputHeight)
            for (int c = 0; c < _outputChannels; ++c)
            {
                float weight = _kernel[c*get_global_size(0)*_kernelWidth*_kernelHeight + tc*_kernelWidth*_kernelHeight + u*_kernelHeight + v];
                float gradOutput = _gradOutput[outputBatchIndex + c*outputWidth*outputHeight + i*outputHeight + j];

                value += weight * gradOutput;
            }
        }
    }

    _inputGrad[inputIndex] = value;
}

__kernel void paramsGradConvolutional(__global float* _weightsGrad, __global float* _biasGrad, __global float* _gradOutput, __global float* _input, int batch)
{
//    const int tc = get_global_id(0);
//    const int tu = get_global_id(1);
//    const int tv = get_global_id(2);
//
//    unsigned shiftu = weightsGrad.size(2)-1-i;
//    unsigned shiftv = weightsGrad.size(3)-1-j;
//
//
//    float value = 0.0f;
//
//    for (unsigned u(0) ; u < _gradOutput.size(1) ; u++)
//        for (unsigned v(0) ; v < _gradOutput.size(2) ; v++)
//            value += _gradOutput(k, u, v) * input(c, u+shiftu, v+shiftv);
//
//    weightsGrad({k, c, i, j}) = value;
}
