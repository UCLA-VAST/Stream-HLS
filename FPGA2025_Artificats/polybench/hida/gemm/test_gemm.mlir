module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @test_gemm(%arg0: memref<200x240xf32>, %arg1: memref<240x220xf32>, %arg2: memref<200x220xf32>, %arg3: f32, %arg4: f32, %arg5: memref<200x220xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg6 = 0 to 200 {
      affine.for %arg7 = 0 to 220 {
        affine.for %arg8 = 0 to 240 {
          %0 = affine.load %arg0[%arg6, %arg8] : memref<200x240xf32>
          %1 = arith.mulf %arg4, %0 : f32
          %2 = affine.load %arg1[%arg8, %arg7] : memref<240x220xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = affine.load %arg5[%arg6, %arg7] : memref<200x220xf32>
          %5 = arith.addf %4, %3 : f32
          affine.store %5, %arg5[%arg6, %arg7] : memref<200x220xf32>
        }
      }
    }
    affine.for %arg6 = 0 to 200 {
      affine.for %arg7 = 0 to 220 {
        %0 = affine.load %arg2[%arg6, %arg7] : memref<200x220xf32>
        %1 = arith.mulf %arg3, %0 : f32
        %2 = affine.load %arg5[%arg6, %arg7] : memref<200x220xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %arg2[%arg6, %arg7] : memref<200x220xf32>
      }
    }
    return
  }
}
