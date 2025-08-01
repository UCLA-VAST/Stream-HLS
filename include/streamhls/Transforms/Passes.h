/*
 * Copyright (c) 2024 Suhail Basalama
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#ifndef STREAMHLS_TRANSFORMS_PASSES_H
#define STREAMHLS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
// #include "streamhls/Dialect/Dataflow/Dataflow.h"
#include "streamhls/InitAllDialects.h"

#include <memory>
#include <string>

namespace mlir {
class Pass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir {
namespace streamhls {
  
enum CreateSubviewMode { Point, Reduction };

void registerTransformsPasses();
void registerStreamHLSBaseKernelPipeline();
void registerStreamHLSKernelPipeline();
void registerStreamHLSHostPipeline();

std::unique_ptr<Pass> createCreateDataflowFromAffinePass();
std::unique_ptr<Pass> createCountOperationsPass();
std::unique_ptr<Pass> createConvertMemRefsToFIFOsPass(bool parallelize = false);
std::unique_ptr<Pass> createRemoveRedundantMemRefsPass();
std::unique_ptr<Pass> createRemoveRedundantOpsPass();
std::unique_ptr<Pass> createRemoveLoopsOfUnitIterPass();
std::unique_ptr<Pass> createConvertToSingleProducerSingleConsumerPass();
std::unique_ptr<Pass> createLowerSubviewToAffinePass();
std::unique_ptr<Pass> createConstantFIFOPropogationPass();
std::unique_ptr<Pass> createCreateTasksPass();
std::unique_ptr<Pass> createCreateTapaTopTaskPass();
std::unique_ptr<Pass> createPipelineInnerLoopsPass();
std::unique_ptr<Pass> createBufferizeFuncArgsPass();
std::unique_ptr<Pass> createStencilDataReusePass();
std::unique_ptr<Pass> createEliminateArrayOfStreamsPass();
std::unique_ptr<Pass> createSystolicArrayParameterizedPass();

std::unique_ptr<Pass> createPrintDataflowGraphPass(std::string dotFileName = "graph.dot", bool mergeNodes = true);
std::unique_ptr<Pass> createNodeGraphPipeliningPass(
  std::string reportFile = "report.csv", 
  std::string loopPermutationType = "Default", 
  bool optimizeSchedule = false,
  uint timeLimitMinutes = 1440
);
std::unique_ptr<Pass> createNodeParallelizationPass(
  std::string reportFile = "model.mod", 
  bool parallelizeNodes = false,
  uint DSPs = 512,
  uint tilingLimit = 8,
  uint timeLimitMinutes = 1440
);
std::unique_ptr<Pass> createCombinedOptimizationPass(
  std::string reportFile = "model.mod", 
  bool parallelizeNodes = false,
  uint DSPs = 512,
  uint tilingLimit = 8,
  uint timeLimitMinutes = 1440
);

std::unique_ptr<Pass> createMinimalBufferSizesPass();
std::unique_ptr<Pass> createPrintIRPass(std::string filePath = "dnn.mlir");

std::unique_ptr<Pass> createCreateWeightBinsPass(
  bool keepWeights = true,
  std::string topFuncName = "forward"
);
std::unique_ptr<Pass> createCreateHostPass();

// ScaleHLS
std::unique_ptr<Pass> createLowerCopyToAffinePass(bool internalCopyOnly = true);
std::unique_ptr<Pass> createCreateDataflowFromLinalgPass();
std::unique_ptr<Pass> createAffineLoopTilePass(unsigned loopTileSize = 1);
std::unique_ptr<Pass> createLinalgFakeQuantizePass();

// std::unique_ptr<Pass> createQoREstimationPass(std::string qorTargetSpec = "");

#define GEN_PASS_CLASSES
#include "streamhls/Transforms/Passes.h.inc"

// #define GEN_PASS_REGISTRATION
// #include "streamhls/Transforms/Passes.h.inc"

} // namespace streamhls
} // namespace mlir

#endif // STREAMHLS_TRANSFORMS_PASSES_H