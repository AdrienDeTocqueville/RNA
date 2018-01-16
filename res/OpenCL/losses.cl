__kernel void gradientMSE(__global float* _output, __global float* _estimation, __global float* _target)
{
    const int index = get_global_id(0)*get_global_size(1) + get_global_id(1);

    _output[index] = _estimation[index] - _target[index];
}

__kernel void gradientNLL(__global float* _output, __global float* _target, int _inputWidth)
{
    const int index = get_global_id(0);

    for (int i = 0; i < _inputWidth; i++)
        _output[index*_inputWidth +i] = 0.0f;

    _output[index*_inputWidth + (int)_target[index]] = -1.0f;
}
