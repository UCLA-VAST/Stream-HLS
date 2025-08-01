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

#ifndef STREAMHLS_TRANSFORMS_PASSES_TD
#define STREAMHLS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Transform Passes
//===----------------------------------------------------------------------===//

def EliminateArrayOfStreams : Pass<"streamhls-eliminate-array-of-streams", "func::FuncOp"> {
  let summary = "Eliminate array of streams";
  let constructor = "mlir::streamhls::createEliminateArrayOfStreamsPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def CreateDataflowFromAffine :
      Pass<"streamhls-create-dataflow-from-affine", "func::FuncOp"> {
  let summary = "Create dataflow hierarchy from affine loops";
  let constructor = "mlir::streamhls::createCreateDataflowFromAffinePass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def CountOperations :
      Pass<"streamhls-count-operations", "func::FuncOp"> {
  let summary = "Count operations of each loop nest";
  let constructor = "mlir::streamhls::createCountOperationsPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def ConvertMemRefsToFIFOs : Pass<"streamhls-convert-memrefs-to-fifos", "func::FuncOp"> {
  let summary = "Convert memrefs to FIFOs";
  let constructor = "mlir::streamhls::createConvertMemRefsToFIFOsPass()";
  let options = [
    Option<"parallelizeNodes", "parallelize-nodes", "bool",
           /*default=*/"false", "wether to parallelize nodes of DFG">
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];

}

def RemoveRedundantMemRefs : Pass<"streamhls-remove-redundant-memrefs", "func::FuncOp"> {
  let summary = "Remove redundant memrefs";
  let constructor = "mlir::streamhls::createRemoveRedundantMemRefsPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def RemoveRedundantOps : Pass<"streamhls-remove-redundant-ops", "ModuleOp"> {
  let summary = "Remove ml global seed";
  let constructor = "mlir::streamhls::createRemoveRedundantOpsPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def RemoveLoopsOfUnitIter : Pass<"streamhls-remove-loops-of-unit-iter", "func::FuncOp"> {
  let summary = "Remove loops of unit iteration";
  let constructor = "mlir::streamhls::createRemoveLoopsOfUnitIterPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def ConvertToSingleProducerSingleConsumer : Pass<"streamhls-convert-to-single-producer-single-consumer", "func::FuncOp"> {
  let summary = "Convert to single producer single consumer";
  let constructor = "mlir::streamhls::createConvertToSingleProducerSingleConsumerPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def LowerSubviewToAffine : Pass<"streamhls-lower-subview-to-affine", "func::FuncOp"> {
  let summary = "Lower subview to affine";
  let constructor = "mlir::streamhls::createLowerSubviewToAffinePass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def ConstantFIFOPropogation : Pass<"streamhls-constant-fifo-propogation", "func::FuncOp"> {
  let summary = "Constant FIFO propogation";
  let constructor = "mlir::streamhls::createConstantFIFOPropogationPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def CreateTasks : Pass<"streamhls-create-tasks", "func::FuncOp"> {
  let summary = "Create tasks from loops";
  let constructor = "mlir::streamhls::createCreateTasksPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def CreateTapaTopTask : Pass<"streamhls-create-tapa-top-task", "func::FuncOp"> {
  let summary = "Create tapa top task";
  let constructor = "mlir::streamhls::createCreateTapaTopTaskPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def PipelineInnerLoops : Pass<"streamhls-pipeline-inner-loops", "func::FuncOp"> {
  let summary = "Pipeline inner loops";
  let constructor = "mlir::streamhls::createPipelineInnerLoopsPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def PrintDataflowGraph : Pass<"streamhls-print-dataflow-graph", "func::FuncOp"> {
  let summary = "Print dataflow graph";
  let constructor = "mlir::streamhls::createPrintDataflowGraphPass()";

  let options = [
    Option<"dotFileName", "outfile", "std::string",
           /*default=*/"\"graph.dot\"", "Output file name">,
    Option<"mergeNodes", "merge-nodes", "bool",
            /*default=*/"false", "Merge nodes with alloc edges">,
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def NodeGraphPipelining : Pass<"streamhls-graph-node-pipelining", "func::FuncOp"> {
  let summary = "Graph node pipelining";
  let constructor = "mlir::streamhls::createNodeGraphPipeliningPass()";

  let options = [
    Option<"reportFile", "outfile", "std::string",
           /*default=*/"\"report.csv\"", "Output file name">,
    Option<"loopPermutationType", "loop-permutation-type", "std::string",
          /*default=*/"\"default\"", "Type of loop permutation (default, minimize, maximize)">,
    Option<"optimizeSchedule", "optimize-schedule", "bool",
            /*default=*/"false", "Optimize schedule through loop permutation using MINLP">,
    Option<"timeLimitMinutes", "time-limit-minutes", "uint",
          /*default*/"1440", "MINLP solver time limit in seconds">
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];

}

def NodeParallelization : Pass<"streamhls-node-parallelization", "func::FuncOp"> {
  let summary = "Node parallelization";
  let constructor = "mlir::streamhls::createNodeParallelizationPass()";
  let options = [
    Option<"reportFile", "outfile", "std::string",
           /*default=*/"\"model.mod\"", "Output file name">,
    Option<"parallelizeNodes", "parallelize-nodes", "bool",
            /*default=*/"false", "Parallelize nodes using MINLP">,
    Option<"DSPs", "board-dsps", "uint",
            /*default=*/"512", "FPGA available DSPs">,
    Option<"tilingLimit", "tiling-limit", "uint",
            /*default=*/"8", "Limit of tiling factor">,
    Option<"timeLimitMinutes", "time-limit-minutes", "uint",
          /*default*/"1440", "MINLP solver time limit in seconds">
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def CombinedOptimization : Pass<"streamhls-combined-optimization", "func::FuncOp"> {
  let summary = "Combined optimization";
  let constructor = "mlir::streamhls::createCombinedOptimizationPass()";
  let options = [
    Option<"reportFile", "outfile", "std::string",
           /*default=*/"\"model.mod\"", "Output file name">,
    Option<"parallelizeNodes", "parallelize-nodes", "bool",
            /*default=*/"false", "Parallelize nodes using MINLP">,
    Option<"DSPs", "board-dsps", "uint",
            /*default=*/"512", "FPGA available DSPs">,
    Option<"tilingLimit", "tiling-limit", "uint",
            /*default=*/"8", "Limit of tiling factor">,
    Option<"timeLimitMinutes", "time-limit-minutes", "uint",
          /*default*/"1440", "MINLP solver time limit in seconds">
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def MinimalBufferSizes : Pass<"streamhls-minimize-buffer-sizes", "func::FuncOp"> {
  let summary = "Minimal buffer sizes";
  let constructor = "mlir::streamhls::createMinimalBufferSizesPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def BufferizeFuncArgs : Pass<"streamhls-bufferize-func-args", "func::FuncOp"> {
  let summary = "Bufferize function arguments";
  let constructor = "mlir::streamhls::createBufferizeFuncArgsPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def PrintIR : Pass<"streamhls-print-ir", "ModuleOp"> {
  let summary = "Print IR";
  let constructor = "mlir::streamhls::createPrintIRPass()";

  let options = [
    Option<"filePath", "file-path", "std::string",
           /*default=*/"\"dnn.mlir\"", "file path">
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];

}

def CreateWeightBins : Pass<"streamhls-create-weight-bins", "ModuleOp"> {
  let summary = "Create weight bins for the DNN accelerator";
  let constructor = "mlir::streamhls::createCreateWeightBinsPass()";

  let options = [
    Option<"keepWeights", "keep-weights", "bool",
           /*default=*/"true", "wether to keep a DNN weights">,
    Option<"topFuncName", "top-func-name", "std::string",
           /*default=*/"\"forward\"", "name of the top function">,
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];

}


def CreateHost :
      Pass<"streamhls-create-host", "func::FuncOp"> {
  let summary = "Creates Host file for the DNN accelerator";
  let constructor = "mlir::streamhls::createCreateHostPass()";
  let dependentDialects = [
    "mlir::streamhls::dataflow::DataflowDialect",
    "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

def StencilDataReuse :
      Pass<"streamhls-stencil-data-reuse", "func::FuncOp"> {
  let summary = "Stencil data reuse";
  let constructor = "mlir::streamhls::createStencilDataReusePass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

def SystolicArrayParameterized : Pass<"streamhls-systolic-array-parameterized", "func::FuncOp"> {
  let summary = "Systolic array parameterized";
  let description = [{
    Systolic array parameterized
  }];
  let constructor = "mlir::streamhls::createSystolicArrayParameterizedPass()";
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];
}

// ScaleHLS
def LowerCopyToAffine : Pass<"scalehls-lower-copy-to-affine", "func::FuncOp"> {
  let summary = "Convert copy and assign to affine loops";
  let constructor = "mlir::streamhls::createLowerCopyToAffinePass()";

  let options = [
    Option<"internalCopyOnly", "internal-copy-only", "bool",
           /*default=*/"false", "only convert copy between internal buffers">
  ];
  let dependentDialects = ["mlir::streamhls::dataflow::DataflowDialect"];

}






#endif // STREAMHLS_TRANSFORMS_PASSES_TD