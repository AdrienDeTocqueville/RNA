__kernel void feedForwardMaxPooling(__global float* _output, __global float* _indices, __global float* _input, int _poolWidth, int _poolHeight, int _batch)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    int inputBatchIndex = _batch * get_global_size(0) * _poolWidth*get_global_size(1) * _poolHeight*get_global_size(2);
    int outputBatchIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2);

    int inputChannelIndex = tc * _poolWidth*get_global_size(1) * _poolHeight*get_global_size(2);
    int outputChannelIndex = tc * get_global_size(1)*get_global_size(2);

    int outputIndex = outputBatchIndex + outputChannelIndex + tx * get_global_size(2) + ty;

    float maxInput = -FLT_MAX;
    int maxIndex = -1;

    for (int i = 0 ; i < _poolWidth ; ++i)
    {
        for (int j = 0 ; j < _poolHeight ; ++j)
        {
            int inputIndex = inputBatchIndex + inputChannelIndex + (_poolWidth*tx+i) * _poolHeight*get_global_size(2) + (_poolHeight*ty+j);

            if (_input[inputIndex] > maxInput)
            {
                maxInput = _input[inputIndex];
                maxIndex = inputIndex;
            }
        }
    }

    _output[outputIndex] = maxInput;
    _indices[outputIndex] = (float)maxIndex;
}

__kernel void backpropMaxPooling(__global float* _inputGrad, __global float* _outputGrad, __global float* _indices, int _numBatches)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    int colorChannel = tc * get_global_size(1)*get_global_size(2);

    for (int batch = 0; batch < _numBatches; ++batch)
    {
        int batchIndex = batch * get_global_size(0)*get_global_size(1)*get_global_size(2);
        int outputIndex = batchIndex + colorChannel + tx * get_global_size(2) + ty;

        _inputGrad[(int)_indices[outputIndex]] = _outputGrad[outputIndex];
    }
}
