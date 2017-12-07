__kernel void tanhLayer(__global float* _output, __global float* _input)
{
    int x = get_global_id(0);

    _output[x] = tanh(_input[x]);
}

__kernel void reluLayer(__global float* _output, __global float* _input)
{
    int x = get_global_id(0);

    _output[x] = max(_input[x], 0.0f);
}
