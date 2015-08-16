/*!
 * Copyright (c) 2015 by Contributors
 * \file utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_COMMON_UTILS_H_
#define MXNET_COMMON_UTILS_H_

#if DMLC_USE_CXX11
#include <memory>
#include <type_traits>
#include <utility>
#endif  // DMLC_USE_CXX11

namespace common {

#if DMLC_USE_CXX11

namespace helper {

template <class T>
struct UniqueIf {
  using SingleObject = std::unique_ptr<T>;
};

template <class T>
struct UniqueIf<T[]> {
  using UnknownBound = std::unique_ptr<T[]>;
};

template <class T, size_t kSize>
struct UniqueIf<T[kSize]> {
  using KnownBound = void;
};

}  // namespace helper

template <class T, class... Args>
typename helper::UniqueIf<T>::SingleObject MakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
};

template <class T>
typename helper::UniqueIf<T>::UnknownBound MakeUnique(size_t n) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[n]{});
}

template <class T, class... Args>
typename helper::UniqueIf<T>::KnownBound MakeUnique(Args&&...) = delete;

#endif  // DMLC_USE_CXX11

}  // namespace common

#endif  // MXNET_COMMON_UTILS_H_
