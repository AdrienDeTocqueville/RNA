__kernel void updateParam(__global float* _param, __global float* _paramGrad, __global float* _paramDelta, float _learningRate, float _inertia)
{
    const int i = get_global_id(0);

    _paramDelta[i] = _inertia * _paramDelta[i] - _learningRate * _paramGrad[i];
    _param[i] += _paramDelta[i];

    _paramGrad[i] = 0.0f;
}
