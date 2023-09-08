/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/cuda_util.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <gtest/gtest.h>

// explicit instantiation for test coverage purposes
template class rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;

namespace rmm::test {
namespace {
using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;

TEST(PinnedPoolTest, ThrowOnNullUpstream)
{
  auto construct_nullptr = []() { pool_mr mr{nullptr}; };
  EXPECT_THROW(construct_nullptr(), rmm::logic_error);
}

TEST(PinnedPoolTest, ThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{&pinned_mr, initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PinnedPoolTest, ReferenceThrowMaxLessThanInitial)
{
  // Make sure first argument is enough larger than the second that alignment rounding doesn't
  // make them equal
  auto max_less_than_initial = []() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    const auto initial{1024};
    const auto maximum{256};
    pool_mr mr{pinned_mr, initial, maximum};
  };
  EXPECT_THROW(max_less_than_initial(), rmm::logic_error);
}

TEST(PinnedPoolTest, AllocateNinetyPercent)
{
  auto allocate_ninety = []() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    auto const [free, total] = rmm::detail::available_device_memory();
    (void)total;
    auto const ninety_percent_pool =
      rmm::detail::align_up(static_cast<std::size_t>(static_cast<double>(free) * 0.9),
                            rmm::detail::CUDA_ALLOCATION_ALIGNMENT);
    pool_mr mr{pinned_mr, ninety_percent_pool};
  };
  EXPECT_NO_THROW(allocate_ninety());
}

TEST(PinnedPoolTest, TwoLargeBuffers)
{
  auto two_large = []() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    auto const [free, total] = rmm::detail::available_device_memory();
    (void)total;
    pool_mr mr{pinned_mr};
    auto* ptr1 = mr.allocate(free / 4);
    auto* ptr2 = mr.allocate(free / 4);
    mr.deallocate(ptr1, free / 4);
    mr.deallocate(ptr2, free / 4);
  };
  EXPECT_NO_THROW(two_large());
}

TEST(PinnedPoolTest, DeletedStream)
{
  rmm::mr::pinned_memory_resource pinned_mr{};
  pool_mr mr{pinned_mr, 0};
  cudaStream_t stream{};  // we don't use rmm::cuda_stream here to make destruction more explicit
  const int size = 10000;
  EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  EXPECT_NO_THROW(rmm::device_buffer buff(size, cuda_stream_view{stream}, &mr));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  EXPECT_NO_THROW(mr.allocate(size));
}

// Issue #527
TEST(PinnedPoolTest, InitialAndMaxPoolSizeEqual)
{
  EXPECT_NO_THROW([]() {
    rmm::mr::pinned_memory_resource pinned_mr{};
    pool_mr mr(pinned_mr, 1000192, 1000192);
    mr.allocate(1000);
  }());
}

TEST(PinnedPoolTest, NonAlignedPoolSize)
{
  EXPECT_THROW(
    []() {
      rmm::mr::pinned_memory_resource pinned_mr{};
      pool_mr mr(pinned_mr, 1000031, 1000192);
      mr.allocate(1000);
    }(),
    rmm::logic_error);

  EXPECT_THROW(
    []() {
      rmm::mr::pinned_memory_resource pinned_mr{};
      pool_mr mr(pinned_mr, 1000192, 1000200);
      mr.allocate(1000);
    }(),
    rmm::logic_error);
}

}  // namespace
}  // namespace rmm::test
