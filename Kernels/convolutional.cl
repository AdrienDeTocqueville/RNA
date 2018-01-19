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

__kernel void backpropConvolutional(__global float* _inputGrad, __global float* _outputGrad, __global float* _kernel, int _outputChannels, int _kernelWidth, int _kernelHeight, int _batch)
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
            {
                for (int c = 0; c < _outputChannels; ++c)
                {
                    float weight = _kernel[c*get_global_size(0)*_kernelWidth*_kernelHeight + tc*_kernelWidth*_kernelHeight + u*_kernelHeight + v];
                    float outputGrad = _outputGrad[outputBatchIndex + c*outputWidth*outputHeight + i*outputHeight + j];

                    value += weight * outputGrad;
                }
            }
        }
    }

    _inputGrad[inputIndex] = value;
}

__kernel void weightsGradConvolutional(__global float* _weightsGrad, __global float* _outputGrad, __global float* _input, int _numBatches, int _outputChannels, int _outputWidth, int _outputHeight, int _outputChannel)
{
    const int tc = get_global_id(0); // inputChannel
    const int ti = get_global_id(1); // kernel width
    const int tj = get_global_id(2); // kernel height

    unsigned inputWidth = _outputWidth+get_global_size(1)-1;
    unsigned inputHeight = _outputHeight+get_global_size(2)-1;

    unsigned shiftu = get_global_size(1)-1-ti;
    unsigned shiftv = get_global_size(2)-1-tj;

    int chanOutIndex = _outputChannel * get_global_size(0)*get_global_size(1)*get_global_size(2);
    int chanInIndex = tc * get_global_size(1)*get_global_size(2);

    int outputGradBatchStride = _outputChannels*_outputWidth*_outputHeight;
    int inputBatchStride = get_global_size(0)*inputWidth*inputHeight;

    int outputGradChannelIndex = _outputChannel * _outputWidth*_outputHeight;

    float value = 0.0f;

    for (int batch = 0; batch < _numBatches; batch++)
    {
        for (int u = 0; u < _outputWidth; ++u)
        {
            for (int v = 0; v < _outputHeight; ++v)
            {
                float outputGrad = _outputGrad[batch*outputGradBatchStride + outputGradChannelIndex + u*_outputHeight + v];
                float input = _input[batch*inputBatchStride + tc*inputWidth*inputHeight + (u+shiftu)*inputHeight + v+shiftv];

                value += outputGrad * input;
            }
        }
    }

    _weightsGrad[chanOutIndex + chanInIndex + ti * get_global_size(2) + tj] += value;
}

__kernel void biasGradConvolutional(__global float* _biasGrad, __global float* _outputGrad, int _numBatches)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    int biasIndex = tc * get_global_size(1)*get_global_size(2) + tx * get_global_size(2) + ty;
    int batchStride = get_global_size(0)*get_global_size(1)*get_global_size(2);

    float value = 0.0f;
    for (int batch = 0; batch < _numBatches; batch++)
        value += _outputGrad[batch * batchStride + biasIndex];

    _biasGrad[biasIndex] += value;
}
