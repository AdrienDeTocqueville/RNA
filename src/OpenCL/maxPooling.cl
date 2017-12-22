__kernel void maxPoolingForward(__global float* _output, __global float* _indices, __global float* _input)
{
    const int tc = get_global_id(0);
    const int tx = get_global_id(1);
    const int ty = get_global_id(2);

//    int outputIndex = _batch * get_global_size(0)*get_global_size(1)*get_global_size(2) + ...;
    int colorChannel = tc * get_global_size(1)*get_global_size(2);
    int outputIndex = colorChannel + tx * get_global_size(2) + ty;

    float maxInput = -FLT_MAX;
    int maxArg = -1;

    for (int i = 0 ; i < 2 ; ++i)
    {
        for (int j = 0 ; j < 2 ; ++j)
        {
            int inputArg = 4*colorChannel + (2*tx+i) * 2*get_global_size(2) + (2*ty+j);

            if (_input[inputArg] > maxInput)
            {
                maxInput = _input[inputArg];
                maxArg = inputArg;
            }
        }
    }

    _output[outputIndex] = maxInput;
    _indices[outputIndex] = maxArg;
}

__kernel void maxPoolingBackward(__global float* _gradInput, __global float* _input, __global float* _gradOutput)
{
}
