add_ops_compile_options(
        OP_NAME NgramSpecDecode
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnInner PRIVATE
        ngram_spec_decode_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_ngram_spec_decode.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        aclnn_ngram_spec_decode.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        aclnn_ngram_spec_decode.cpp
    )
endif ()

target_sources(optiling PRIVATE
        ngram_spec_decode_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE)

file(GLOB _Ngram_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_ngram_spec_decode.h")

install(FILES ${_Ngram_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
