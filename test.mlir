func.func @test_add() -> tensor<1x2xi32> {
  %arg0 = nova.constant {value = dense<[[1, 2]]> : tensor<1x2xi32>} : tensor<1x2xi32>
  %arg1 = nova.constant {value = dense<[[4, 4]]> : tensor<1x2xi32>} : tensor<1x2xi32>
  %0 = nova.add %arg0, %arg1 : tensor<1x2xi32>, tensor<1x2xi32>
  %1 = nova.sub %arg0, %arg1 : tensor<1x2xi32>, tensor<1x2xi32>
 // %10 = nova.add %0, %1 : tensor<1x2xi32>, tensor<1x2xi32>

  return %1 : tensor<1x2xi32>
}
