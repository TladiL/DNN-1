#pragma once

#include <Eigen/Core>
#include "../Config.h"

namespace MiniDNN
{
	class Softmax
	{
	private:
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
		typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArray;

	public:
		static inline void activation(const Matrix& Z, Matrix& A)
		{
			A.array() = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
			RowArray colsums = A.colwise().sum();
			A.array().rowwise() /= colsums;
		}

		static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
		{
			RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
			G.array() = A.array() * (F.array().rowwise() - a_dot_f);
		}

		static std::string return_type() { return "Softmax"; }
	};
}