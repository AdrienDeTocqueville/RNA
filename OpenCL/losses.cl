// GWS: minibatch, inputWidth
__kernel void mseGradient(__global float* _output, __global float* _estimation, __global float* _target)
{
    const int index = get_global_id(0)*get_global_size(1) + get_global_id(1);

    _output[index] = 2.0f * (_estimation[index] - _target[index]);
}

// GWS: minibatch
// Note: init output to 0
__kernel void nllGradient(__global float* _output, __global float* _estimation, __global float* _target, int _inputWidth)
{
    const int index = get_global_id(0);

    output[index*_inputWidth + _target[index]] = -1.0;
}
