#pragma once

#include "RNA/RNA.h"

namespace RL
{

void test();

bool envStep(const size_t& action, double& reward, Tensor& nextState, bool v = false);

void SARSA();
void SARSALambda();
void QLearning();

}
