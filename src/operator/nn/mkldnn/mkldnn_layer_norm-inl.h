/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file mkldnn_layer_norm-inl.h
 * \brief
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LAYER_NORM_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LAYER_NORM_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <utility>
#include <vector>
#include <mkldnn.hpp>
#include "../layer_norm-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

inline const int GetAxis(const size_t& ndim, const int& axis) {
  const int out_axis = axis < 0 ? axis + ndim : axis;
  CHECK(out_axis >= 0 && out_axis < ndim) << "Channel axis out of range: " << axis;
  return out_axis;
}

inline void InitShapeAndStrides(mkldnn::memory::dims& shape,
                                mkldnn::memory::dims& strides,
                                const mxnet::TShape& _shape,
                                const size_t& ndim) {
  shape[0] = _shape[0];
  for (int i = ndim - 1; i > 0; --i) {
    shape[i] = _shape[i];
    strides[i - 1] = strides[i] * shape[i];
  }
}

inline mkldnn::memory::desc GetInputDataDesc(const int &_axis,
                                             const mkldnn::memory::data_type& dtype,
                                             const mxnet::TShape& _shape) {
  const auto ndim = _shape.ndim();
  const auto axis = GetAxis(ndim, _axis);

  mkldnn::memory::dims shape(ndim, 1), strides(ndim, 1);
  InitShapeAndStrides(shape, strides, _shape, ndim);

  if (axis != ndim -1) {
    // swap shape[axis] with shape[ndim - 1] and strides[axis] with strides[ndim - 1]
    std::swap(shape[ndim - 1], shape[axis]);
    strides[ndim - 1] = strides[axis];
    strides[axis] = 1;
  }

  return mkldnn::memory::desc{shape, dtype, strides};
}

inline mkldnn::memory GetScaleShiftMem(const NDArray& gamma, const NDArray& beta) {
  // oneDNN takes gamma and beta as one SCALE_SHIFT tensor
  constexpr size_t gammaAndBeta = 2;
  CHECK_EQ(gamma.shape()[0], beta.shape()[0]);
  const mkldnn::memory::desc scale_shift_md(mkldnn::memory::dims{gammaAndBeta, gamma.shape()[0]},
                                            get_mkldnn_type(gamma.dtype()), mkldnn::memory::format_tag::nc);
  auto scale_shift_mem = mkldnn::memory(scale_shift_md, CpuEngine::Get()->get_engine());
  char *ptr = (char*)scale_shift_mem.get_data_handle();
  const size_t bytes = scale_shift_md.get_size() / gammaAndBeta;
  memcpy(ptr, gamma.data().dptr_, bytes);
  memcpy(ptr + bytes, beta.data().dptr_, bytes);
  return scale_shift_mem;
}

inline mkldnn::layer_normalization_forward::primitive_desc GetLayerNormFwdDesc(
    const LayerNormParam &param, const bool is_train, const mkldnn::memory::desc &src_md) {
  auto kind = mkldnn::prop_kind::forward_training;
  if (is_train || param.output_mean_var) { // MKLDNN will return mean and var only for forward_training
    kind = mkldnn::prop_kind::forward_training;
  } else {
    kind = mkldnn::prop_kind::forward_scoring;
  }

  mkldnn::layer_normalization_forward::desc fwd_desc(kind, src_md, param.eps, dnnl::normalization_flags::use_scale_shift);
  mkldnn::engine &engine = CpuEngine::Get()->get_engine();
  return mkldnn::layer_normalization_forward::primitive_desc(fwd_desc, engine);
}

inline mkldnn::layer_normalization_backward::primitive_desc GetLayerNormBwdDesc(
    const LayerNormParam &param, const mkldnn::memory::desc &data_md,
    const mkldnn::memory::desc &diff_md,
    const mkldnn::layer_normalization_forward::primitive_desc &layer_normFwd_desc) {
  mkldnn::layer_normalization_backward::desc layer_normBwd_desc(dnnl::prop_kind::backward, diff_md, data_md,
                param.eps, dnnl::normalization_flags::use_scale_shift);
  mkldnn::engine &engine = CpuEngine::Get()->get_engine();
  return mkldnn::layer_normalization_backward::primitive_desc(layer_normBwd_desc,
                               engine, layer_normFwd_desc);
}

typedef ParamOpSign<LayerNormParam> MKLDNNLayerNormSignature;

// LayerNorm Forward Class
class MKLDNNLayerNormFwd {
 public:
  MKLDNNLayerNormFwd(const LayerNormParam& param,
                     const bool is_train,
                     const NDArray &data) {
    _Init(param, is_train, data);
  }

  ~MKLDNNLayerNormFwd() {}

  void Execute(const LayerNormParam &param,
               const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const OpReqType req,
               const std::vector<NDArray> &outputs);

 private:
  std::shared_ptr<mkldnn::layer_normalization_forward> fwd;
  mkldnn::layer_normalization_forward::primitive_desc fwd_pd;

 private:
  void _Init(const LayerNormParam &param, const bool is_train, const NDArray &data);
  inline mkldnn::memory::desc GetOutputMd(const NDArray& data,
                                          const mxnet::TShape& out_shape,
                                          const int& _axis);
};  // End of LayerNorm Forward Class

void MKLDNNLayerNormFwd::_Init(const LayerNormParam &param,
                               const bool is_train,
                               const NDArray &data) {
  const mkldnn::memory::desc data_md = GetInputDataDesc(param.axis, get_mkldnn_type(data.dtype()), data.shape());
  this->fwd_pd = GetLayerNormFwdDesc(param, is_train, data_md);
  this->fwd = std::shared_ptr<mkldnn::layer_normalization_forward>(new mkldnn::layer_normalization_forward(this->fwd_pd));
}

inline mkldnn::memory::desc MKLDNNLayerNormFwd::GetOutputMd(const NDArray& data,
                                                            const mxnet::TShape& out_shape,
                                                            const int& _axis) {
  const auto ndim = data.shape().ndim();
  const auto axis = GetAxis(ndim, _axis);
  if (axis == ndim - 1) { // early return for default case
    return (this->fwd_pd).dst_desc();
  }

  mkldnn::memory::dims shape(ndim, 1), strides(ndim, 1);
  InitShapeAndStrides(shape, strides, data.shape(), ndim);

  std::swap(shape[ndim - 1], shape[axis]);
  strides[axis] = (strides[axis] * shape[ndim - 1]) / shape[axis];
  strides[ndim - 1] = 1;

  return mkldnn::memory::desc{shape,
                              get_mkldnn_type(data.dtype()),
                              strides};
}

void MKLDNNLayerNormFwd::Execute(const LayerNormParam &param,
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const OpReqType req,
                                 const std::vector<NDArray> &outputs) {
  auto out_md = GetOutputMd(inputs[layernorm::kData], outputs[layernorm::kOut].shape(), param.axis);
  auto output_mem = CreateMKLDNNMem(outputs[layernorm::kOut], out_md, req);
  auto mean_mem = mkldnn_output_t(OutDataOp::CopyBack, TmpMemMgr::Get()->Alloc((this->fwd_pd).mean_desc()));
  auto variance_mem = mkldnn_output_t(OutDataOp::CopyBack, TmpMemMgr::Get()->Alloc((this->fwd_pd).variance_desc()));
  auto scale_shift_mem = GetScaleShiftMem(inputs[layernorm::kGamma], inputs[layernorm::kBeta]);

  mkldnn_args_map_t args = {
    { MKLDNN_ARG_SRC, *inputs[layernorm::kData].GetMKLDNNData() },
    { MKLDNN_ARG_DST, *output_mem.second },
    { MKLDNN_ARG_MEAN, *mean_mem.second },
    { MKLDNN_ARG_VARIANCE, *variance_mem.second },
    { MKLDNN_ARG_SCALE_SHIFT, scale_shift_mem }
  };

  MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd), args);
  CommitOutput(outputs[layernorm::kOut], output_mem);
  if (param.output_mean_var || ctx.is_train) {
    CommitOutput(outputs[layernorm::kMean], mean_mem);
    CommitOutput(outputs[layernorm::kStd], variance_mem);
  }
  MKLDNNStream::Get()->Submit();
}

// End of LayerNorm Forward Class functions

static MKLDNNLayerNormFwd &GetLayerNormFwd(const LayerNormParam& param,
                                           const OpContext &ctx,
                                           const NDArray &data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNLayerNormSignature,
                                         MKLDNNLayerNormFwd,
                                         OpHash> layer_norm_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNLayerNormSignature,
                                            MKLDNNLayerNormFwd,
                                            OpHash> layer_norm_fwds;
#endif
  auto kind_ = ctx.is_train ? mkldnn::prop_kind::forward_training
                            : mkldnn::prop_kind::forward_scoring;

  MKLDNNLayerNormSignature key(param);
  key.AddSign(static_cast<int>(kind_));
  key.AddSign(data);
  key.AddSign(GetAxis(data.shape().ndim(), param.axis));
  key.AddSign(param.eps);

  auto it = layer_norm_fwds.find(key);
  if (it == layer_norm_fwds.end()) {
    MKLDNNLayerNormFwd fwd(param, ctx.is_train, data);
    it = AddToCache(&layer_norm_fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNLayerNormForward(const nnvm::NodeAttrs &attrs,
                            const OpContext &ctx,
                            const std::vector<NDArray> &inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray> &outputs) {
  if (req[layernorm::kOut] == kNullOp) return;
  CHECK_NE(req[layernorm::kOut], kAddTo);
  const LayerNormParam &param = nnvm::get<LayerNormParam>(attrs.parsed);
  MKLDNNLayerNormFwd fwd = GetLayerNormFwd(param, ctx, inputs[layernorm::kData]);
  fwd.Execute(param, ctx, inputs, req[layernorm::kOut], outputs);
}

// LayerNorm Backward Class
class MKLDNNLayerNormBwd {
  std::shared_ptr<mkldnn::layer_normalization_backward> bwd;

 public:
  const mkldnn::layer_normalization_forward::primitive_desc fwd_pd;
  const mkldnn::layer_normalization_backward::primitive_desc bwd_pd;

  ~MKLDNNLayerNormBwd() {}

  MKLDNNLayerNormBwd(const LayerNormParam &param,
                     const std::vector<NDArray> &inputs,
                     const mkldnn::memory::desc& data_md,
                     const mkldnn::memory::desc& diff_md)
      : fwd_pd(GetLayerNormFwdDesc(param, true, data_md)),
        bwd_pd(GetLayerNormBwdDesc(param, data_md, diff_md, this->fwd_pd)) {
          bwd = std::make_shared<mkldnn::layer_normalization_backward>(bwd_pd);
        }

  const mkldnn::layer_normalization_backward &GetBwd() const { return *bwd; }

  void Execute(const std::vector<NDArray> &inputs,
               const std::vector<NDArray> &outputs,
               const std::vector<OpReqType> &req) {
    auto scale_shift_mem = GetScaleShiftMem(inputs[layernorm::kBGamma], inputs[layernorm::kBBeta]);

    auto diff_weights_ndarray = NDArray(scale_shift_mem.get_desc());
    mkldnn_output_t diff_src_mem = CreateMKLDNNMem(outputs[layernorm::kBDgrad], this->bwd_pd.diff_src_desc(), req[layernorm::kBDgrad]);
    mkldnn_output_t diff_weights_mem = CreateMKLDNNMem(diff_weights_ndarray, this->bwd_pd.diff_weights_desc(), req[layernorm::kBGgrad]);

    mkldnn_args_map_t args = {
      { MKLDNN_ARG_DIFF_DST, *inputs[layernorm::kBOgrad].GetMKLDNNData() },
      { MKLDNN_ARG_SRC, *inputs[layernorm::kBData].GetMKLDNNData() },
      { MKLDNN_ARG_SCALE_SHIFT, scale_shift_mem },
      { MKLDNN_ARG_MEAN, *inputs[layernorm::kBMean].GetMKLDNNData() },
      { MKLDNN_ARG_VARIANCE, *inputs[layernorm::kBStd].GetMKLDNNData() },
      { MKLDNN_ARG_DIFF_SRC, *diff_src_mem.second },
      { MKLDNN_ARG_DIFF_SCALE_SHIFT, *diff_weights_mem.second }
    };

    MKLDNNStream::Get()->RegisterPrimArgs(*(this->bwd), args);
    CommitOutput(outputs[layernorm::kBDgrad], diff_src_mem);
    CommitOutput(diff_weights_ndarray, diff_weights_mem);
    MKLDNNStream::Get()->Submit();
    // Commit scale_shift diff
    auto bytes = inputs[layernorm::kBGamma].shape()[0] * sizeof(float);
    memcpy(outputs[layernorm::kBGgrad].data().dptr_, diff_weights_ndarray.data().dptr_, bytes);
    memcpy(outputs[layernorm::kBBgrad].data().dptr_, diff_weights_ndarray.data().dptr_ + bytes, bytes);
  }
};  // End of LayerNorm Backward Class

static MKLDNNLayerNormBwd &GetLayerNormBwd(const LayerNormParam &param,
                                           const std::vector<NDArray> &inputs) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
      std::unordered_map<MKLDNNLayerNormSignature, MKLDNNLayerNormBwd, OpHash> layer_norm_bwds;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<MKLDNNLayerNormSignature, MKLDNNLayerNormBwd, OpHash> layer_norm_bwds;
#endif
  auto ndim = inputs[layernorm::kBData].shape().ndim();
  auto axis = GetAxis(ndim, param.axis);
  MKLDNNLayerNormSignature key(param);
  key.AddSign(inputs[layernorm::kBOgrad]);
  key.AddSign(inputs[layernorm::kBData]);
  key.AddSign(inputs[layernorm::kBGamma]);
  key.AddSign(inputs[layernorm::kBMean]);
  key.AddSign(inputs[layernorm::kBStd]);
  key.AddSign(inputs[layernorm::kBBeta]);
  key.AddSign(axis);
  key.AddSign(param.eps);

  auto it = layer_norm_bwds.find(key);
  if (it == layer_norm_bwds.end()) {
    const mkldnn::memory::desc data_md = GetInputDataDesc(param.axis,
                                                          get_mkldnn_type(inputs[layernorm::kBData].dtype()),
                                                          inputs[layernorm::kBData].shape());
    const mkldnn::memory::desc diff_md = inputs[layernorm::kBOgrad].GetMKLDNNData()->get_desc();
    MKLDNNLayerNormBwd bwd(param, inputs, data_md, diff_md);
    it = AddToCache(&layer_norm_bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNLayerNormBackward(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
  const LayerNormParam &param = nnvm::get<LayerNormParam>(attrs.parsed);

  MKLDNNLayerNormBwd &bwd = GetLayerNormBwd(param, inputs);
  bwd.Execute(inputs, outputs, req);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LAYER_NORM_INL_H__
