add_mlir_library(AffineFullUnroll
  AffineFullUnroll.cpp
  AffineFullUnroll.h

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Compiler/Transforms/Affine

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
)

target_include_directories(AffineFullUnroll PRIVATE
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include

)

