__kernel void feedForwardMaxPooling(__global float* _output, __global int* _indices, __global float* _input, int _batch)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    int batchIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2);
    int colorChannel = tc * get_global_size(1)*get_global_size(2);
    int outputIndex = batchIndex + colorChannel + tx * get_global_size(2) + ty;

    float maxInput = -FLT_MAX;
    int maxIndex = -1;

    for (int i = 0 ; i < 2 ; ++i)
    {
        for (int j = 0 ; j < 2 ; ++j)
        {
            int inputIndex = 4*batchIndex + 4*colorChannel + (2*tx+i) * 2*get_global_size(2) + (2*ty+j);

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

__kernel void backpropMaxPooling(__global float* _gradInput, __global float* _gradOutput, __global float* _indices, int _batch)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

    int batchIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2);
    int colorChannel = tc * get_global_size(1)*get_global_size(2);
    int outputIndex = batchIndex + colorChannel + tx * get_global_size(2) + ty;

    _gradInput[(int)_indices[outputIndex]] = _gradOutput[outputIndex];
}
