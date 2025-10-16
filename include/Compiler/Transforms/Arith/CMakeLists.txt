add_mlir_library(Arithcanoonicalizer
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

target_include_directories(Arithcanoonicalizer PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include

)