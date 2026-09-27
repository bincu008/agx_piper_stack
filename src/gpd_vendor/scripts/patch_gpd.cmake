set(gpd_cmake "${GPD_SOURCE_DIR}/CMakeLists.txt")
file(READ "${gpd_cmake}" contents)

string(REPLACE
  "set(CMAKE_BUILD_TYPE Debug)"
  "if(NOT CMAKE_BUILD_TYPE)\n  set(CMAKE_BUILD_TYPE Release)\nendif()"
  contents "${contents}")

set(upstream_flags "set(CMAKE_CXX_FLAGS \"-O3 -march=native -mtune=intel -msse4.2 -mavx2 -mfma -flto -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17\")")
set(portable_flags "set(CMAKE_CXX_FLAGS \"-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17\")")
if(contents MATCHES "-mtune=intel|-msse4.2|-mavx2|-mfma")
  string(REPLACE "${upstream_flags}" "${portable_flags}" contents "${contents}")
elseif(NOT contents MATCHES "-O3 -fopenmp -fPIC")
  message(FATAL_ERROR "Pinned GPD compiler flags changed; review the portability patch")
endif()

if(contents MATCHES "-mtune=intel|-msse4.2|-mavx2|-mfma")
  message(FATAL_ERROR "x86-only GPD flags remain after portability patch")
endif()
file(WRITE "${gpd_cmake}" "${contents}")

# Upstream GPD writes hundreds of progress and per-grasp score lines directly
# to stdout on every inference. In a composed ROS process those messages cannot
# be filtered by the ROS logger. Route only the pinned library's routine console
# output to a null stream; the adapter publishes concise ROS summaries and
# propagates failures through the service status.
set(quiet_header "${GPD_SOURCE_DIR}/include/gpd/util/quiet_logging.h")
file(WRITE "${quiet_header}" [=[
#ifndef GPD_UTIL_QUIET_LOGGING_H_
#define GPD_UTIL_QUIET_LOGGING_H_

#include <ostream>
#include <streambuf>

namespace gpd {
namespace util {

class NullLogBuffer final : public std::streambuf {
 protected:
  int overflow(int character) override { return character; }
};

inline std::ostream &quietStream() {
  static NullLogBuffer buffer;
  static std::ostream stream(&buffer);
  return stream;
}

template <typename... Args>
inline int quietPrintf(const char *, Args &&...) {
  return 0;
}

}  // namespace util
}  // namespace gpd

#endif  // GPD_UTIL_QUIET_LOGGING_H_
]=])

set(quiet_sources
  "src/gpd/grasp_detector.cpp"
  "src/gpd/candidate/candidates_generator.cpp"
  "src/gpd/candidate/frame_estimator.cpp"
  "src/gpd/candidate/hand_search.cpp"
  "src/gpd/descriptor/image_generator.cpp"
  "src/gpd/net/eigen_classifier.cpp"
  "src/gpd/util/cloud.cpp")
foreach(relative_source IN LISTS quiet_sources)
  set(source_file "${GPD_SOURCE_DIR}/${relative_source}")
  file(READ "${source_file}" source_contents)
  if(NOT source_contents MATCHES "gpd/util/quiet_logging.h")
    set(source_contents "#include <gpd/util/quiet_logging.h>\n${source_contents}")
  endif()
  string(REPLACE "std::cout" "gpd::util::quietStream()"
    source_contents "${source_contents}")
  string(REPLACE "printf(" "gpd::util::quietPrintf("
    source_contents "${source_contents}")
  file(WRITE "${source_file}" "${source_contents}")
endforeach()
