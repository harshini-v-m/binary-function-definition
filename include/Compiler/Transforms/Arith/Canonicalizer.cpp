// Canonicalize.cpp
#include "Canonicalizer.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Compiler/Dialect/nova/NovaOps.h"

using namespace mlir;
using namespace mlir::nova;
namespace mlir {
namespace nova {
struct NovaaddconstPattern : public OpRewritePattern<mlir::nova::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::nova::AddOp op,
                                PatternRewriter &rewriter) const override {
                                  
    auto lhs = op.getLhs().getDefiningOp<mlir::nova::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<mlir::nova::ConstantOp>();

    // Match only if both operands are constants
    if (!lhs || !rhs)
      return mlir::failure();
    //getting values from the operands
    auto lhsAttr = lhs.getValue().dyn_cast<DenseElementsAttr>();
    auto rhsAttr = rhs.getValue().dyn_cast<DenseElementsAttr>();

    //only for integer types
    llvm::SmallVector<APInt> results;
    // Perform addition for integer types
    for (auto it : llvm::zip(lhsAttr.getValues<APInt>(), rhsAttr.getValues<APInt>())) {
      results.push_back(std::get<0>(it) + std::get<1>(it));
    }
    // Replacing the original op with a new constant
    auto resultType = lhsAttr.getType();
    auto resultAttr = DenseElementsAttr::get(resultType, results);
    rewriter.replaceOpWithNewOp<mlir::nova::ConstantOp>(op, resultType, resultAttr);
    return mlir::success();
  }
};


//Defining pattern for subtract op
struct NovasubconstPattern : public OpRewritePattern<mlir::nova::SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::nova::SubOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<mlir::nova::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<mlir::nova::ConstantOp>();

    // Match only if both operands are constants
    if (!lhs || !rhs)
      return mlir::failure();
    //getting values from the operands
    auto lhsAttr = lhs.getValue().dyn_cast<DenseElementsAttr>();
    auto rhsAttr = rhs.getValue().dyn_cast<DenseElementsAttr>();

    //only for integer types
    llvm::SmallVector<APInt> results;
    // Perform addition for integer types
    for (auto it : llvm::zip(lhsAttr.getValues<APInt>(), rhsAttr.getValues<APInt>())) {
      results.push_back(std::get<0>(it) - std::get<1>(it));
    }
    // Replacing the original op with a new constant
    auto resultType = lhsAttr.getType();
    auto resultAttr = DenseElementsAttr::get(resultType, results);
    rewriter.replaceOpWithNewOp<mlir::nova::ConstantOp>(op, resultType, resultAttr);
    return mlir::success();
  }
};

void Novapass::runOnOperation(){
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<NovaaddconstPattern>(&getContext());
  patterns.add<NovasubconstPattern>(&getContext());
 
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace nova
} // namespace mlir