// AffineFullUnroll.cpp
#include "AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

void AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](mlir::affine::AffineForOp op) {
    if (failed(mlir::affine::loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
    else {
      llvm::outs() << "Successfully unrolled loop\n";
    }
  });
}

} // namespace nova
} // namespace mlir