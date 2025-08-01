#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "streamhls/Transforms/Passes.h"
// #include "streamhls/Dialect/Dataflow/Transforms/Utils.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"

#include "streamhls/Support/Utils.h"
#include "streamhls/Support/AffineMemAccess.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/IR/AffineMap.h"

#include "mlir/IR/IntegerSet.h"


using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-systolic-array-parameterized"

namespace{
  struct SystolicArrayParameterizedPattern : public OpRewritePattern<AffineForOp> {
    using OpRewritePattern<AffineForOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
      AffineForOp forOp,
      PatternRewriter &rewriter
    ) const override {
      if(hasSALoopUnrollAttr(forOp)){
        llvm::dbgs() << "Unrolling loop with unroll factor: " << getSALoopUnrollAttr(forOp) << "\n";
        return success();
      }
      return failure();
    }
      
  };
} // namespace
namespace {
struct SystolicArrayParameterized
    : public SystolicArrayParameterizedBase<SystolicArrayParameterized> {
  SystolicArrayParameterized() = default;
  void runOnOperation() override {
    llvm::dbgs() << "Running SystolicArrayParameterized Pass...\n";
    auto func = getOperation();
    auto context = func.getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<SystolicArrayParameterizedPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> streamhls::createSystolicArrayParameterizedPass() {
  return std::make_unique<SystolicArrayParameterized>();
}