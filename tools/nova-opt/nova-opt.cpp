#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Transforms/Affine/AffineFullUnroll.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"


int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // Register only the dialects we need
   registry.insert<mlir::nova::NovaDialect>();
  mlir::registerAllDialects(registry);

  mlir::PassRegistration<mlir::nova::AffineFullUnrollPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Nova dialect optimizer\n", registry));
}
