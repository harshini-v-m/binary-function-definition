#ifndef LIB_TRANSFORM_AFFINE_ARITH_CONSTANT_FOLD_H_
#define LIB_TRANSFORM_AFFINE_ARITH_CONSTANT_FOLD_H_

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"


namespace mlir {
namespace nova {    

class Novapass : public PassWrapper<Novapass, OperationPass<mlir::func::FuncOp>> {
private:
    void runOnOperation() override;
    StringRef getArgument() const final { return "arith-const-fold"; }
    StringRef getDescription() const final { return "Canonicalize arith.add ops with constant operands"; 
    }

};

}
}

#endif 