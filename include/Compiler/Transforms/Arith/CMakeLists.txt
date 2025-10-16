add_mlir_library(Arithcanonicalizer
  Canonicalizer.cpp
  Canonicalizer.h

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Arith

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    MLIRNova
)

target_include_directories(Arithcanonicalizer PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include

)