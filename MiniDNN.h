#pragma once

#include <Eigen/Core>

#include "Config.h"

#include "RNG.h"

#include "Layer.h"
#include "Layer/FullyConnected.h"
#include "Layer/Convolutional.h"
#include "Layer/MaxPooling.h"

#include "Activation/Indentity.h"
#include "Activation/Mish.h"
#include "Activation/ReLU.h"
#include "Activation/Sigmoid.h"
#include "Activation/Tanh.h"
#include "Activation/Softmax.h"