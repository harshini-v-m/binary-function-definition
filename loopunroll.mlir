                                                                                                                                                                                     
func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.alloc() : memref<10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 5{
    affine.store %cst, %0[%arg2] : memref<10xf32>
    affine.store %cst, %1[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %0[%arg2] : memref<10xf32>
    %3 = arith.addf %2, %2 : f32
    affine.store %3, %arg0[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %1[%arg2] : memref<10xf32>
    %3 = arith.mulf %2, %2 : f32
    affine.store %3, %arg1[%arg2] : memref<10xf32>
  }
  return
}

func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                     %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                     %arg4: memref<10x10xf32>) {
  affine.for %arg5 = 0 to 3 {
    affine.for %arg6 = 0 to 3 {
      %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
      %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
      %2 = arith.mulf %0, %1 : f32
      affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
    }
  }
  affine.for %arg5 = 0 to 3 {
    affine.for %arg6 = 0 to 3 {
      %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
      %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
    }
  }
  return
}
