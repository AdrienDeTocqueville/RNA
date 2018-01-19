__kernel void updateParam(__global float* _param, __global float* _paramGrad, __global float* _r, float _learningRate, float _rho, float _delta)
{
    const int i = get_global_id(0);

    _r[i] = _rho * _r[i] + (1.0f-_rho) * _paramGrad[i]* _paramGrad[i];

    float delta = -(_learningRate * _paramGrad[i]) * rsqrt(_delta+_r[i]);
    _param[i] += delta;

    _paramGrad[i] = 0.0f;
}
