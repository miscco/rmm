# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Use CPM to find or clone cccl
function(find_and_configure_cccl VERSION)
  rapids_cpm_find(
    cccl ${VERSION}
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY TRUE)

  set(CCCL_INCLUDE_DIR
      "${cccl_SOURCE_DIR}/libcudacxx/include"
      PARENT_SCOPE)
endfunction()

set(RMM_MIN_VERSION_cccl 2.2.0)

find_and_configure_cccl(${RMM_MIN_VERSION_cccl})
