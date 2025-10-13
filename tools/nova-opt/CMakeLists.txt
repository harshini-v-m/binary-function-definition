add_llvm_executable(nova-opt nova-opt.cpp)
llvm_update_compile_flags(nova-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(nova-opt PRIVATE
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRNova
  ${dialect_libs}
  ${conversion_libs}
  AffineFullUnroll 
)