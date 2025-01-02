/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/IntegerSet.h"


#include "streamhls/Transforms/Passes.h"
#include "streamhls/Support/Utils.h"
#include "streamhls/Support/AffineMemAccess.h"
#include "streamhls/Support/DFG.h"

#include <regex>

using namespace mlir;
using namespace streamhls;
using namespace dataflow;

#define DEBUG_TYPE "streamhls-node-level-parallelization"

namespace {
struct CombinedOptimization : public CombinedOptimizationBase<CombinedOptimization> {
  CombinedOptimization() = default;
  CombinedOptimization(
    std::string argReportFile,
    bool argParallelizeNodes,
    uint argDSPs,
    uint argTilingLimit,
    uint argTimeLimitMinutes
  ) {
    reportFile = argReportFile;
    parallelizeNodes = argParallelizeNodes;
    DSPs = argDSPs;
    tilingLimit = argTilingLimit;
    timeLimitMinutes = argTimeLimitMinutes;
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto context = func.getContext();
    OpBuilder builder(context);
    auto block = &func.front();
    DFG graph(*block);
    if (!graph.init(false)) {
      LLVM_DEBUG(llvm::dbgs() << "DFG init failed\n");
      return;
    }
    graph.createCombinedOptimizationPerformanceModel(reportFile, DSPs, tilingLimit, timeLimitMinutes);

    graph.callCombinedOptimizationSolver(reportFile);

    graph.createCombinedOptimizationPythonModel(reportFile);
    
    graph.applyCombinedOptimization();
 
  }
};
}

std::unique_ptr<Pass> streamhls::createCombinedOptimizationPass(
  std::string reportFile,
  bool parallelizeNodes,
  uint DSPs,
  uint tilingLimit,
  uint timeLimitMinutes
) {
  return std::make_unique<CombinedOptimization>(
    reportFile,
    parallelizeNodes,
    DSPs,
    tilingLimit,
    timeLimitMinutes
  );
}