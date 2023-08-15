#pragma once

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"
#include "../Utils/Convolution.h"
#include "../Utils/Random.h"
#include "../Utils/IO.h"
#include "../Utils/Enum.h"

namespace MiniDNN
{
//	template <typename Activation>
	class Convolutional : public Layer
	{
	private:
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
		typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
		typedef Vector::ConstAlignedMapType ConstAlignedMapvec;
		typedef Vector::AlignedMapType AlignedMapVec;
		typedef std::map<std::string, int> MataInfo;

		const internal::ConvDims m_dim;

		Vector m_filter_data;
		Vector m_df_data;

		Vector m_bias;
		Vector m_db;

		Matrix m_z;
		Matrix m_a;
		Matrix m_din;
		
	public:
		Convolutional(const int in_width, const int in_height,
					  const int in_channels, const int out_channels,
					  const int window_width, const int window_height) :
		
		Layer(in_width * in_height * in_channels,
			  (in_width - window_width + 1) * (in_height - window_height + 1) * out_channels),
		m_dim(in_channels, out_channels, in_height, in_width, window_height, window_width)
		{}

		void init(const Scalar& mu, const Scalar& sigma, RNG& rng)
		{
			init();

			const int filter_data_size = m_dim.in_channels * m_dim.out_channels *
										m_dim.filter_rows * m_dim.filter_cols;
			internal::set_normal_random(m_filter_data.data(), filter_data_size, rng, mu, sigma);

			internal::set_normal_random(m_bias.data(), m_dim.out_channels, rng, mu, sigma);
		}

		void init()
		{
			const int filter_data_size = m_dim.in_channels * m_dim.out_channels*
										m_dim.filter_rows * m_dim.filter_cols;

			m_filter_data.resize(filter_data_size);
			m_df_data.resize(filter_data_size);

			m_bias.resize(m_dim.out_channels);
			m_db.resize(m_dim.out_channels);

		}

		void forward(const Matrix& prev_layer_data)
		{
			const int nobs = prev_layer_data.cols();
			m_z.resize(this->m_out_size, nobs);

			internal::convolve_valid(m_dim, prev_layer_data.data(), true, nobs,
									m_filter_data.data(), m_z.data());
			int channel_start_row = 0;
			const int channel_nelem = m_dim.conv_rows * m_dim.conv_cols;

			for (int i = 0; i < m_dim.out_channels; i++, channel_start_row += channel_nelem)
			{
				m_z.block(channel_start_row, 0, channel_nelem, nobs).array() += m_bias[i];
			}

			m_a.resize(this->m_out_size, nobs);
//			Activation::activate(m_z, m_a);
		}

		const Matrix& output() const { return m_a; }

		void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data)
		{
			const int nobs = prev_layer_data.cols();

			Matrix& dLz = m_z;
//			Activation::apply_jacobian(m_z, m_a, next_layer_data, dLz);

			internal::ConvDims back_conv_dim(nobs, m_dim.out_channels, m_dim.channel_rows,
											m_dim.channel_cols, m_dim.conv_rows, m_dim.conv_cols);
			internal::convolve_valid(back_conv_dim, prev_layer_data.data(), false, m_dim.in_channels,
									dLz.data(), m_df_data.data());

			m_df_data /= nobs;

			ConstAlignedMapMat dLz_by_channel(dLz.data(), m_dim.conv_rows * m_dim.conv_cols,
											m_dim.out_channels * nobs);
			Vector dLb = dLz_by_channel.colwise().sum();

			ConstAlignedMapMat dLb_by_obs(dLb.data(), m_dim.out_channels, nobs);
			m_db.noalias() = dLb_by_obs.rowwise().mean();

			m_din.resize(this->m_in_size, nobs);
			internal::ConvDims conv_full_dim(m_dim.out_channels, m_dim.in_channels, m_dim.conv_rows,
											m_dim.conv_cols, m_dim.filter_rows, m_dim.filter_cols);
			internal::convolve_full(conv_full_dim, dLz.data(), nobs, m_filter_data.data(), m_din.data());
		}

		const Matrix& backprop_data() const { return m_din; }

		void update(Optimizer& opt) {}

		std::vector<Scalar> get_parameters() const {}

		void set_parameters(const std::vector<Scalar>& param) {};

		std::vector<Scalar> get_derivatives() const {}

		std::string layer_type() const {}

		std::string activataion_type() const {}

		void fill_meta_info(MetaInfo& map, int index) const {}
	};
}