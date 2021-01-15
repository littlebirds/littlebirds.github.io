project(perfbench)

#find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(perfbench perfbench.cpp)
target_link_libraries(perfbench "${TORCH_LIBRARIES}")
target_link_libraries(perfbench "${OpenCV_LIBS}")
#target_link_libraries(example-app  PUBLIC TorchVision::TorchVision)
set_property(TARGET perf PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET perfbench
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
