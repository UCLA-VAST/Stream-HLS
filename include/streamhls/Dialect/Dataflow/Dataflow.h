//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

/*
 * Modified by Suhail Basalama in 2024.
 *
 * This software is also released under the MIT License:
 * https://opensource.org/licenses/MIT
 */

#ifndef STREAMHLS_DIALECT_DATAFLOW_H
#define STREAMHLS_DIALECT_DATAFLOW_H

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "streamhls/Dialect/Dataflow/DataflowDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "streamhls/Dialect/Dataflow/DataflowTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "streamhls/Dialect/Dataflow/DataflowAttributes.h.inc"

namespace mlir {
namespace streamhls {
namespace dataflow {

#include "streamhls/Dialect/Dataflow/DataflowInterfaces.h.inc"

enum class PartitionKind { CYCLIC, BLOCK, NONE };
enum class OperandKind { INPUT, OUTPUT, PARAM };
enum class MemoryKind {
  LUTRAM_1P = 1,
  LUTRAM_2P,
  LUTRAM_S2P,
  BRAM_1P,
  BRAM_2P,
  BRAM_S2P,
  BRAM_T2P,
  URAM_1P,
  URAM_2P,
  URAM_S2P,
  URAM_T2P,
  DRAM
};

bool isRam1P(MemoryKind kind);
bool isRam2P(MemoryKind kind);
bool isRamS2P(MemoryKind kind);
bool isRamT2P(MemoryKind kind);
bool isDram(MemoryKind kind);

namespace StreamEffects {
struct Instantiate : public MemoryEffects::Effect::Base<Instantiate> {};
struct Push : public MemoryEffects::Effect::Base<Push> {};
struct Pop : public MemoryEffects::Effect::Base<Pop> {};
} // namespace StreamEffects

/// Timing attribute utils.
TimingAttr getTiming(Operation *op);
void setTiming(Operation *op, TimingAttr timing);
void setTiming(Operation *op, int64_t begin, int64_t end, int64_t latency,
               int64_t interval);

/// Resource attribute utils.
ResourceAttr getResource(Operation *op);
void setResource(Operation *op, ResourceAttr resource);
void setResource(Operation *op, int64_t lut, int64_t dsp, int64_t bram);

/// Loop information attribute utils.
LoopInfoAttr getLoopInfo(Operation *op);
void setLoopInfo(Operation *op, LoopInfoAttr loopInfo);
void setLoopInfo(Operation *op, int64_t flattenTripCount, int64_t iterLatency,
                 int64_t minII);

/// Loop directives attribute utils.
LoopDirectiveAttr getLoopDirective(Operation *op);
void setLoopDirective(Operation *op, LoopDirectiveAttr loopDirective);
void setLoopDirective(Operation *op, bool pipeline, int64_t targetII,
                      bool dataflow, bool flatten);

/// Parrallel and Reduction loop attribute utils.
bool hasParallelAttr(Operation *op);
void setParallelAttr(Operation *op);
bool hasReductionAttr(Operation *op);
void setReductionAttr(Operation *op);
void setPEAttr(Operation *op);
bool hasPEAttr(Operation *op);
void setColAttr(Operation *op);
bool hasColAttr(Operation *op);
void setRowAttr(Operation *op);
bool hasRowAttr(Operation *op);

// SA unroll attribute utils.
void setSALoopUnrollAttr(Operation *op, int64_t unroll);
bool hasSALoopUnrollAttr(Operation *op);
int64_t getSALoopUnrollAttr(Operation *op);

/// Function directives attribute utils.
FuncDirectiveAttr getFuncDirective(Operation *op);
void setFuncDirective(Operation *op, FuncDirectiveAttr FuncDirective);
void setFuncDirective(Operation *op, bool pipeline, int64_t targetInterval,
                      bool dataflow);

/// Top and runtime function attribute utils.
bool hasTopFuncAttr(Operation *op);
void setTopFuncAttr(Operation *op);
bool hasRuntimeAttr(Operation *op);
void setRuntimeAttr(Operation *op);

} // namespace dataflow
} // namespace streamhls
} // namespace mlir

#define GET_OP_CLASSES
#include "streamhls/Dialect/Dataflow/Dataflow.h.inc"

#endif // STREAMHLS_DIALECT_DATAFLOW_H

