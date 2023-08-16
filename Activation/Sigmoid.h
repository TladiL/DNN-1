#pragma once

#include <Eigen/Core>
#include "../Config.h"

namespace MiniDNN
{
	class Sigmoid
	{
	private:
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

	public:
		static inline void activation(const Matrix& Z, Matrix& A)
		{ A.array() = Scalar(1) / (Scalar(1) + (-Z.array()).exp()); }

		static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
		{ G.array() = A.array() * (Scalar(1) - A.array()) * F.array(); }

		static std::string return_type() { return "Sigmoid"; }
	};
}