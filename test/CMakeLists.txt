enable_testing ()

#
# replace XX with the version you have
#
set (Boost_ADDITIONAL_VERSIONS "1.64" "1.64.0")

#
# Find boost, triggers an error otherwise
#
find_package (Boost COMPONENTS unit_test_framework REQUIRED)

#
# Add all sources in test folder
#
file (GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

#
#Run through each source
#
foreach (testSrc ${TEST_SRCS})
  #
  # Extract the filename without an extension (NAME_WE)
  #
  get_filename_component (testName ${testSrc} NAME_WE)

  #
  # Add compile target
  #
  add_executable (${testName} ${testSrc})

  #
  # Link to Boost libraries AND your targets and dependencies
  #
  target_link_libraries (${testName}
    ${Boost_LIBRARIES}
    ${MsFEComplex_LIBRARY}
    )

  #
  # Add includes for test
  #
  target_include_directories (${testName}
    PRIVATE ${Boost_INCLUDE_DIRS}
    ${INCLUDE_DIR}
    )

  #
  # Shared library variant
  #
  # target_compile_definitions (${testName} PRIVATE "BOOST_TEST_DYN_LINK=1")

  #
  # I like to move testing binaries into a testBin directory
  #
  set_target_properties (${testName} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/test
    )

  #
  # Finally add it to test execution - Notice the WORKING_DIRECTORY
  # and COMMAND
  #
  add_test (NAME ${testName} 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test
    COMMAND ${CMAKE_BINARY_DIR}/test/${testName}
    )
endforeach(testSrc)
