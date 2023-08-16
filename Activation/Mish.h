#pragma once

#include <Eigen/Core>
#include "../Config.h"

namespace MiniDNN
{
	class Mish
	{
	private:
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

	public:
		static inline void activate(const Matrix& Z, Matrix& A)
		{
			Matrix S = (-Z.array().abs()).exp();
			A.array() = (S.array() + Scalar(1)).square();
			S.noalias() = (Z.array() >= Scalar(0)).select(S.cwiseAbs2(), Scalar(1));
			A.array() = (A.array() - S.array()) / (A.array() + S.array());
			A.array() *= Z.array();
		}

		static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
		{
			G.noalias() = (Z.array() == Scalar(0)).select(Scalar(0.6), A.cwiseQuotient(Z));
			G.array() += (Z.array() - A.array() * G.array()) / (Scalar(1) + (-Z).array().exp());
			G.array() *= F.array();
		}

		static std::string return_type() { return "Mish"; }
	};
}