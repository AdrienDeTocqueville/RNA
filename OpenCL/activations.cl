float dtanh(float _x)
{
    float t = tanh(_x);
    return 1.0f - t*t;
}


__kernel void tanhForward(__global float* _output, __global float* _input)
{
    const int index = get_global_id(0)*get_global_size(1) + get_global_id(1);

    _output[index] = tanh(_input[index]);
}

__kernel void tanhBackward(__global float* _gradInput, __global float* _input, __global float* _gradOutput)
{
    const int index = get_global_id(0)*get_global_size(1) + get_global_id(1);

    _gradInput[index] = dtanh(_input[index]) * _gradOutput[index];
}


__kernel void reluForward(__global float* _output, __global float* _input)
{
    const int index = get_global_id(0)*get_global_size(1) + get_global_id(1);

    _output[index] = max(_input[index], 0.0f);
}

__kernel void reluBackward(__global float* _gradInput, __global float* _input, __global float* _gradOutput)
{
    const int index = get_global_id(0)*get_global_size(1) + get_global_id(1);

	_gradInput[index] = (_input[index] < 0.0f)? 0.0f: _gradOutput[index];
}
