cmake_minimum_required(VERSION 3.18.4)
project(catlass_kernel)

set(KERNEL_OBJ_FILES "")
set(ALL_KERNEL_TARGETS "")

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(CMAKE_BISHENG_COMPILER bisheng)
set(BISHENG_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce -fPIC
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${CMAKE_CURRENT_SOURCE_DIR}/src/common
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    -I${CATLASS_INCLUDE_DIR}
    -Wno-macro-redefined -Wno-ignored-attributes
)

set(BISHENG_LINK_OPTIONS
    --cce-fatobj-link
    -L${ASCEND_HOME_PATH}/lib64
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

if(DEFINED PROF)
    list(APPEND ${BISHENG_LINK_OPTIONS} -lprofapi)
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/obj)

function(catlass_add_kernel NAME ARCH)
    set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/obj)
    set(OUTPUT_NAME ${NAME}.o)
    set(OUTPUT_FULL_NAME ${OUTPUT_DIR}/${OUTPUT_NAME})

    add_custom_command(
        OUTPUT ${OUTPUT_FULL_NAME}
        COMMAND ${CMAKE_BISHENG_COMPILER} --cce-aicore-arch=${ARCH} ${BISHENG_COMPILER_OPTIONS} -c ${ARGN} -o ${OUTPUT_FULL_NAME}
        DEPENDS ${ARGN} ${HOST_SRC} ${CATLASS_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling single kernel obj ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${OUTPUT_FULL_NAME})

    list(APPEND KERNEL_OBJ_FILES ${OUTPUT_FULL_NAME})
    set(KERNEL_OBJ_FILES "${KERNEL_OBJ_FILES}" PARENT_SCOPE)

    list(APPEND ALL_KERNEL_TARGETS ${NAME})
    set(ALL_KERNEL_TARGETS "${ALL_KERNEL_TARGETS}" PARENT_SCOPE)
endfunction()

# if aicore-arch is different, maybe crash, to find the solution
catlass_add_kernel(basic_matmul dav-c220 ${CMAKE_CURRENT_SOURCE_DIR}/src/host/mla_launcher.cpp)

message("Kernel Object Files: ${KERNEL_OBJ_FILES}")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/libcatlass_kernel.so
    COMMAND ${CMAKE_BISHENG_COMPILER} ${BISHENG_LINK_OPTIONS} ${KERNEL_OBJ_FILES} --shared -o ${CMAKE_BINARY_DIR}/libcatlass_kernel.so
    DEPENDS ${KERNEL_OBJ_FILES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building shared library libcatlass_kernel.so"
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/libcatlass_kernel.a
    COMMAND ${CMAKE_BISHENG_COMPILER} ${BISHENG_LINK_OPTIONS} ${KERNEL_OBJ_FILES} --cce-build-static-lib -o ${CMAKE_BINARY_DIR}/libcatlass_kernel.a
    DEPENDS ${KERNEL_OBJ_FILES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building static library libcatlass_kernel.a"
)

add_custom_target(catlass_kernel ALL DEPENDS
    ${CMAKE_BINARY_DIR}/libcatlass_kernel.so
    ${CMAKE_BINARY_DIR}/libcatlass_kernel.a
)

add_dependencies(catlass_kernel ${ALL_KERNEL_TARGETS})
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    install(FILES ${CMAKE_BINARY_DIR}/libcatlass_kernel.so DESTINATION lib)
    install(FILES ${CMAKE_BINARY_DIR}/libcatlass_kernel.a DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/catlass_kernel.h DESTINATION include)
endif()
