#ifndef LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
#define LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


namespace mlir {
namespace nova {    

class AffineFullUnrollPass : public PassWrapper<AffineFullUnrollPass, OperationPass<mlir::func::FuncOp>> {
private:
    void runOnOperation() override;
    StringRef getArgument() const final { return "affine-full-unroll"; }
    StringRef getDescription() const final { return "Fully unroll affine.for ops with constant bounds"; 
    }

};

}
}

#endif 