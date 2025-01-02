module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @test_gesummv(%arg0: f32, %arg1: f32, %arg2: memref<250x250xf32>, %arg3: memref<250x250xf32>, %arg4: memref<250xf32>, %arg5: memref<250xf32>, %arg6: memref<250xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg7 = 0 to 250 {
      affine.for %arg8 = 0 to 250 {
        %5 = affine.load %arg2[%arg7, %arg8] : memref<250x250xf32>
        %6 = affine.load %arg5[%arg8] : memref<250xf32>
        %7 = arith.mulf %5, %6 : f32
        %8 = affine.load %arg4[%arg7] : memref<250xf32>
        %9 = arith.addf %8, %7 : f32
        affine.store %9, %arg4[%arg7] : memref<250xf32>
        %10 = affine.load %arg3[%arg7, %arg8] : memref<250x250xf32>
        %11 = affine.load %arg5[%arg8] : memref<250xf32>
        %12 = arith.mulf %10, %11 : f32
        %13 = affine.load %arg6[%arg7] : memref<250xf32>
        %14 = arith.addf %13, %12 : f32
        affine.store %14, %arg6[%arg7] : memref<250xf32>
      }
      %0 = affine.load %arg4[%arg7] : memref<250xf32>
      %1 = arith.mulf %arg0, %0 : f32
      %2 = affine.load %arg6[%arg7] : memref<250xf32>
      %3 = arith.mulf %arg1, %2 : f32
      %4 = arith.addf %1, %3 : f32
      affine.store %4, %arg6[%arg7] : memref<250xf32>
    }
    return
  }
}
