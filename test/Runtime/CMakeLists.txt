# Nova Runtime Tests

add_executable(DeviceTest DeviceTest.cpp)
target_link_libraries(DeviceTest PRIVATE NovaRuntime)

add_executable(BufferTest BufferTest.cpp)
target_link_libraries(BufferTest PRIVATE NovaRuntime)

# Add tests to CTest
enable_testing()
add_test(NAME DeviceTest COMMAND DeviceTest)
add_test(NAME BufferTest COMMAND BufferTest)