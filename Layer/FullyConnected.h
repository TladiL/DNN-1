#pragma once

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Random.h"
#include "../Utils/IO.h"
#include "../Utils/Enum.h"

namespace MiniDNN
{
	template <typename Activation>
	class FullyConnected : public Layer
	{
	private:
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
		typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
		typedef Vector::AlignedMapType AlignedMapVec;
		typedef std::map<std::string, int> MetaInfo;

		Matrix m_weight;
		Vector m_bias;
		Matrix m_dw;
		Vector m_db;
		Matrix m_z;
		Matrix m_a;
		Matrix m_din;

	public:
		FullyConnected(const int in_size, const int out_size) :
		Layer(in_size, out_size) {}

		void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
		{
			init();

			internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
			internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
		}

		void init()
		{
			m_weight.resize(this->m_in_size, this->m_out_size);
			m_bias.resize(this->m_out_size);
			m_dw.resize(this->m_in_size, this->m_out_size);
			m_db.resize(this->m_out_size);
		}

		void forward(const Matrix& prev_layer_data)
		{
			const int nobs = prev_layer_data.cols();

			m_z.resize(this->m_out_size, nobs);
			m_z.noalias() = m_weight.transpose() * prev_layer_data;
			m_z.colwise() += m_bias;

			m_a.resize(this->m_out_size, nobs);
			Activation::actiavate(m_z, m_a);
		}

		const Matrix& output() const { return m_a; }

		void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
		{
			const int nobs = prev_layer_data.cols();

			Matrix& dLz = m_z;
			Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);

			m_dw.noalias() = prev_layer_data * dLz.transpose() / nobs;

			m_db.noalias() = dLz.rowwise().mean();

			m_din.resize(this->m_in_size, nobs);
			m_din.noalias() = m_weight * dLz;
		}

		const Matrix& backprop_data() const { return m_din; }

		void update(Optimizer& opt)
		{
			ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
			ConstAlignedMapVec db(m_db.data(), m_db.size());
			AlignedMapVec	   w(m_weight.data(), m_weight.size());
			AlignedMapVec	   b(m_bias.data(), m_bias.size());
			opt.update(dw, w);
			opt.update(db, b);
		}

		std::vector<Scalar> get_parameters() const
		{
			std::vector<Scalar> res(m_weight.size() + m_bias.size());

			std::copy(m_weight.data(), m_weight.data() + m_weight.size(), res.begin());
			std::copy(m_bias.data(), m_bias.data() + m_bias.size(), res.begin() + m_weight.size());

			return res;
		}

		void set_parameters(const std::vector<Scalar>& param)
		{
			if (static_cast<int>(param.size()) != m_weight.size() + m_bias.size())
			{
				throw std::invalid_argument("[Class FullyConnected]: Parameter Size Does Not Match");
			}

			std::copy(param.begin(), param.begin() + m_weight.size(), m_weight.data());
			std::copy(param.begin() + m_weight.size(), param.end(), m_bias.data());
		}

		std::vector<Scalar> get_derivatives() const
		{
			std::vector<Scalar> res(m_dw.size() + m_db.size());

			std::copy(m_dw.data(), m_dw.data() + m_dw.size(), res.begin());
			std::copy(m_db.data(), m_db.data() + m_db.size(), res.begin() + m_dw.size());
			return res;
		}

		std::string layer_type() const { return "FullyConnected"; }

		std::string activataion_type() const { return Activation::return_type(); }

		void fill_meta_info(MetaInfo& map, int index) const
		{
			std::string ind = internal::to_string(index);
			map.insert(std::make_pair("Layer " + ind, internal::layer_id(layer_type())));
			map.insert(std::make_pair("Actiavtion " + ind, internal::activation_id(activataion_type())));
			map.insert(std::make_pair("in_size " + ind, in_size()));
			map.insert(std::make_pair("out_size " + ind, out_size()));
		}
	};
}