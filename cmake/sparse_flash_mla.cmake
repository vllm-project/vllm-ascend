# SparseFlashMLA uses the installed CANN implementation on A5, but the
# repository's custom operator on A2/A3. Build against the same component
# version that will be used at runtime, as with the other native interfaces.
function(get_sparse_flash_mla_metadata_size OUT SOC CANN_ROOT)
  set(METADATA_SIZE 1024)
  set(VERSION_FILE "${CANN_ROOT}/share/info/ops_transformer/version.info")
  if(SOC MATCHES "^ascend950" AND EXISTS "${VERSION_FILE}")
    file(STRINGS "${VERSION_FILE}" VERSION_LINE REGEX "^Version=")
    if(NOT VERSION_LINE MATCHES "^Version=([0-9]+\\.[0-9]+\\.[0-9]+)")
      message(FATAL_ERROR "Cannot read ops-transformer version from ${VERSION_FILE}")
    endif()
    if(CMAKE_MATCH_1 VERSION_GREATER_EQUAL "9.3.0")
      set(METADATA_SIZE 2048)
    endif()
  endif()
  set(${OUT} ${METADATA_SIZE} PARENT_SCOPE)
endfunction()
