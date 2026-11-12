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
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # Pyodide: a WebAssembly side module, loaded by VACASK's dlopen like the native
  # library. The Py* symbols stay undefined and resolve against the Pyodide main
  # module, like any extension module's.
  target_link_options(socklight_osdi PRIVATE -sSIDE_MODULE=1)
endif()

# scikit-build-core's editable.mode = "inplace" configures in the source
# directory: put the library next to the package sources.
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  set_target_properties(socklight_osdi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/socklight")
endif()

install(TARGETS socklight_osdi LIBRARY DESTINATION socklight)

# Verilog-A helpers (va/*.va), compiled with OpenVAF-reloaded into socklight/va/<name>.osdi.
# Default: the openvaf-r that vacask-bin ships (a build requirement, see
# pyproject.toml; since 0.3.4.dev5 it has the wasm32-unknown-emscripten target),
# else openvaf-r on PATH.
set(OPENVAF_COMPILER "" CACHE STRING "openvaf-r, the OpenVAF compiler used for va/*.va (name on PATH or full path; default: vacask-bin's)")
if(NOT OPENVAF_COMPILER)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import vacask_bin; print(vacask_bin.OPENVAF_CMD)"
    OUTPUT_VARIABLE _vacask_bin_openvaf OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _res ERROR_QUIET)
  if(_res EQUAL 0 AND EXISTS "${_vacask_bin_openvaf}")
    set(OPENVAF_COMPILER "${_vacask_bin_openvaf}")
  else()
    set(OPENVAF_COMPILER "openvaf-r")
  endif()
endif()
message(STATUS "OpenVAF compiler: ${OPENVAF_COMPILER}")
find_program(_openvaf NAMES "${OPENVAF_COMPILER}" REQUIRED NO_CMAKE_FIND_ROOT_PATH)
# Extra openvaf-r arguments for the target, and the environment it runs in.
# openvaf-r defaults to --target_cpu native: models built on one machine can then
# die with "Illegal instruction" on another. Wheels must run anywhere.
set(OPENVAF_TARGET_CPU "generic" CACHE STRING "CPU the va/*.va models are compiled for (openvaf-r --target_cpu; 'native' for local builds)")
set(_openvaf_target --target_cpu "${OPENVAF_TARGET_CPU}")
set(_openvaf_launcher "")
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # As VACASK's Pyodide build: the models become WebAssembly side modules,
  # compiled for wasm32-unknown-emscripten and linked by the wasm-ld of the
  # Emscripten SDK that builds socklight.osdi.
  execute_process(COMMAND "${_openvaf}" --supported-targets
    OUTPUT_VARIABLE _targets RESULT_VARIABLE _res)
  if(NOT _res EQUAL 0 OR NOT _targets MATCHES "wasm32-unknown-emscripten")
    message(FATAL_ERROR "${_openvaf} cannot compile for WebAssembly (no wasm32-unknown-emscripten target)")
  endif()
  get_filename_component(_cc_dir "${CMAKE_C_COMPILER}" DIRECTORY)
  find_program(WASM_LD wasm-ld
    HINTS "${EMSCRIPTEN_ROOT_PATH}/../bin" "$ENV{EMSDK}/upstream/bin" "${_cc_dir}/../bin"
    NO_CMAKE_FIND_ROOT_PATH REQUIRED)
  message(STATUS "WebAssembly linker for va/*.va: ${WASM_LD}")
  set(_openvaf_target --target wasm32-unknown-emscripten)
  set(_openvaf_launcher "${CMAKE_COMMAND}" -E env "OPENVAF_WASM_LD=${WASM_LD}")
endif()
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
    COMMAND ${_openvaf_launcher} "${_openvaf}" ${_openvaf_target} "${_src}" -o "${_dst}"
    DEPENDS "${_src}"
    COMMENT "OpenVAF ${_name}.va"
    VERBATIM)
  list(APPEND _va_osdi "${_dst}")
endforeach()
add_custom_target(socklight_va ALL DEPENDS ${_va_osdi})
install(FILES ${_va_osdi} DESTINATION socklight/va)
