# ARM64 JIT Backend
add_library(xenia-cpu-backend-a64 STATIC)
file(GLOB_RECURSE _a64_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)
target_sources(xenia-cpu-backend-a64 PRIVATE ${_a64_sources})
target_include_directories(xenia-cpu-backend-a64 PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
)
target_link_libraries(xenia-cpu-backend-a64 PUBLIC fmt xenia-base xenia-cpu)
xe_target_defaults(xenia-cpu-backend-a64)
