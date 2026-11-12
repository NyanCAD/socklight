cmake_minimum_required(VERSION 3.18)
project(socklight LANGUAGES C)

# socklight.osdi: one OSDI device per model in socklight's Python registry.
# Built against the Python headers only; the Py* symbols resolve from the
# process that loads it (VACASK's Python extension), so libpython is not linked.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

set(SOCKLIGHT_MAX_MODELS 1024 CACHE STRING "Descriptor slots in socklight.osdi (the most models one process can register)")
if(NOT SOCKLIGHT_MAX_MODELS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "SOCKLIGHT_MAX_MODELS must be a positive integer, got '${SOCKLIGHT_MAX_MODELS}'")
endif()

# One SL_SLOT(k) line per descriptor slot; socklight_osdi.c expands it into the
# slot's trampolines and their table.
set(_slots "")
math(EXPR _last "${SOCKLIGHT_MAX_MODELS} - 1")
foreach(k RANGE 0 ${_last})
  string(APPEND _slots "SL_SLOT(${k})\n")
endforeach()
set(_gen "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/socklight_generated")
file(CONFIGURE OUTPUT "${_gen}/sl_slots.inc" CONTENT "${_slots}")

add_library(socklight_osdi MODULE
  csrc/socklight_osdi.c
  csrc/device.c
  csrc/bridge.c
)
target_include_directories(socklight_osdi PRIVATE csrc "${_gen}")
target_link_libraries(socklight_osdi PRIVATE Python::Module)
target_compile_definitions(socklight_osdi PRIVATE SOCKLIGHT_MAX_MODELS=${SOCKLIGHT_MAX_MODELS})
target_link_libraries(socklight_osdi PRIVATE m)
set_target_properties(socklight_osdi PROPERTIES
  OUTPUT_NAME socklight
  PREFIX ""
  SUFFIX ".osdi"
  C_STANDARD 11
  C_EXTENSIONS ON
  C_VISIBILITY_PRESET hidden
)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(socklight_osdi PRIVATE -Wall -Wextra)
endif()

# scikit-build-core's editable.mode = "inplace" configures in the source
# directory: put the library next to the package sources.
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  set_target_properties(socklight_osdi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/socklight")
endif()

install(TARGETS socklight_osdi LIBRARY DESTINATION socklight)

# Verilog-A helpers (va/*.va), compiled with OpenVAF-reloaded into socklight/va/<name>.osdi.
set(OPENVAF_COMPILER "openvaf-r" CACHE STRING "openvaf-r, the OpenVAF compiler used for va/*.va (name on PATH or full path)")
find_program(_openvaf NAMES "${OPENVAF_COMPILER}" REQUIRED)
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(_va_out "${CMAKE_SOURCE_DIR}/src/socklight/va")
else()
  set(_va_out "${CMAKE_CURRENT_BINARY_DIR}/va")
endif()
file(MAKE_DIRECTORY "${_va_out}")
set(_va_osdi "")
foreach(_name optical_source optical_term photodetector heater)
  set(_src "${CMAKE_CURRENT_SOURCE_DIR}/va/${_name}.va")
  set(_dst "${_va_out}/${_name}.osdi")
  add_custom_command(
    OUTPUT "${_dst}"
    COMMAND "${_openvaf}" "${_src}" -o "${_dst}"
    DEPENDS "${_src}"
    COMMENT "OpenVAF ${_name}.va"
    VERBATIM)
  list(APPEND _va_osdi "${_dst}")
endforeach()
add_custom_target(socklight_va ALL DEPENDS ${_va_osdi})
install(FILES ${_va_osdi} DESTINATION socklight/va)
