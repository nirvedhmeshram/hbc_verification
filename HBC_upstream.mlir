#map0 = affine_map<(d0) -> (0)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "HistogramBinningCalibrationByFeature"} {
  func @forward(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xi64>) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant dense<4.000000e-01> : tensor<1xf32>
    %c5001_i64 = arith.constant 5001 : i64
    %c42_i64 = arith.constant 42 : i64
    %cst_1 = arith.constant 2.000000e-04 : f64
    %c5000_i64 = arith.constant 5000 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<215000xf64>
    %cst_3 = arith.constant 9.995000e-01 : f64
    %cst_4 = arith.constant 5.000000e-04 : f64
    %true = arith.constant true
    %0 = linalg.init_tensor [1] : tensor<1xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel"]} ins(%cst_0 : tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = math.log %arg3 : f32
      linalg.yield %108 : f32
    } -> tensor<1xf32>
    %2 = tensor.dim %arg2, %c0 : tensor<?xf32>
    %3 = linalg.init_tensor [%2] : tensor<?xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map0, #map1], iterator_types = ["parallel"]} ins(%arg2, %1 : tensor<?xf32>, tensor<1xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %108 = arith.sitofp %c1_i64 : i64 to f32
      %109 = arith.mulf %arg4, %108 : f32
      %110 = arith.addf %arg3, %109 : f32
      linalg.yield %110 : f32
    } -> tensor<?xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : tensor<?xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = arith.negf %arg3 : f32
      %109 = math.exp %108 : f32
      %110 = arith.addf %109, %cst : f32
      %111 = arith.divf %cst, %110 : f32
      linalg.yield %111 : f32
    } -> tensor<?xf32>
    %6 = arith.index_cast %2 : index to i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = linalg.init_tensor [%7] : tensor<?xi32>
    %9 = linalg.fill(%c0_i32, %8) : i32, tensor<?xi32> -> tensor<?xi32> 
    %10 = tensor.dim %arg1, %c0 : tensor<?xi32>
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.addi %c1_i64, %11 : i64
    %13 = arith.cmpi sge, %c1_i64, %c0_i64 : i64
    %14 = select %13, %c1_i64, %12 : i64
    %15 = arith.cmpi slt, %14, %c0_i64 : i64
    %16 = select %15, %c0_i64, %14 : i64
    %17 = arith.cmpi sgt, %16, %11 : i64
    %18 = select %17, %11, %16 : i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.addi %c5001_i64, %11 : i64
    %21 = arith.cmpi sge, %c5001_i64, %c0_i64 : i64
    %22 = select %21, %c5001_i64, %20 : i64
    %23 = arith.cmpi slt, %22, %c0_i64 : i64
    %24 = select %23, %c0_i64, %22 : i64
    %25 = arith.cmpi sgt, %24, %11 : i64
    %26 = select %25, %11, %24 : i64
    %27 = arith.index_cast %26 : i64 to index
    %28 = arith.cmpi sge, %27, %19 : index
    %29 = select %28, %27, %19 : index
    %30 = arith.subi %29, %19 : index
    %31 = tensor.extract_slice %arg1[%19] [%30] [1] : tensor<?xi32> to tensor<?xi32>
    %32 = arith.addi %c0_i64, %11 : i64
    %33 = arith.cmpi sge, %c0_i64, %c0_i64 : i64
    %34 = select %33, %c0_i64, %32 : i64
    %35 = arith.cmpi slt, %34, %c0_i64 : i64
    %36 = select %35, %c0_i64, %34 : i64
    %37 = arith.cmpi sgt, %36, %11 : i64
    %38 = select %37, %11, %36 : i64
    %39 = arith.index_cast %38 : i64 to index
    %40 = arith.addi %c5000_i64, %11 : i64
    %41 = arith.cmpi sge, %c5000_i64, %c0_i64 : i64
    %42 = select %41, %c5000_i64, %40 : i64
    %43 = arith.cmpi slt, %42, %c0_i64 : i64
    %44 = select %43, %c0_i64, %42 : i64
    %45 = arith.cmpi sgt, %44, %11 : i64
    %46 = select %45, %11, %44 : i64
    %47 = arith.index_cast %46 : i64 to index
    %48 = arith.cmpi sge, %47, %39 : index
    %49 = select %48, %47, %39 : index
    %50 = arith.subi %49, %39 : index
    %51 = tensor.extract_slice %arg1[%39] [%50] [1] : tensor<?xi32> to tensor<?xi32>
    %52 = arith.cmpi eq, %30, %50 : index
    assert %52, "mismatched size for broadcast"
    %53 = linalg.init_tensor [%30] : tensor<?xi1>
    %54 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%31, %51 : tensor<?xi32>, tensor<?xi32>) outs(%53 : tensor<?xi1>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i1):
      %108 = arith.cmpi sgt, %arg3, %arg4 : i32
      linalg.yield %108 : i1
    } -> tensor<?xi1>
    %55 = linalg.init_tensor [%50] : tensor<?xi64>
    %56 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%51 : tensor<?xi32>) outs(%55 : tensor<?xi64>) {
    ^bb0(%arg3: i32, %arg4: i64):
      %108 = arith.extsi %arg3 : i32 to i64
      linalg.yield %108 : i64
    } -> tensor<?xi64>
    %57 = tensor.dim %56, %c0 : tensor<?xi64>
    %58 = linalg.init_tensor [%57] : tensor<?xi32>
    %59 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%56 : tensor<?xi64>) outs(%58 : tensor<?xi32>) {
    ^bb0(%arg3: i64, %arg4: i32):
      %108 = arith.index_cast %arg3 : i64 to index
      %109 = tensor.extract %arg0[%108] : tensor<?xi32>
      linalg.yield %109 : i32
    } -> tensor<?xi32>
    %60 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%59 : tensor<?xi32>) outs(%58 : tensor<?xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %108 = arith.trunci %c1_i64 : i64 to i32
      %109 = arith.muli %108, %108 : i32
      %110 = arith.addi %arg3, %109 : i32
      linalg.yield %110 : i32
    } -> tensor<?xi32>
    %61 = arith.cmpi eq, %30, %57 : index
    assert %61, "mismatched size for broadcast"
    %62 = arith.cmpi eq, %30, %7 : index
    assert %62, "mismatched size for broadcast"
    %63 = linalg.init_tensor [%30] : tensor<?xi32>
    %64 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%54, %60, %9 : tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) outs(%63 : tensor<?xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %108 = select %arg3, %arg4, %arg5 : i32
      linalg.yield %108 : i32
    } -> tensor<?xi32>
    %65 = arith.addi %c0_i64, %c1_i64 : i64
    %66 = arith.cmpi sge, %c0_i64, %c0_i64 : i64
    %67 = select %66, %c0_i64, %65 : i64
    %68 = arith.cmpi sge, %67, %c0_i64 : i64
    assert %68, "dim must be greater or equal to zero"
    %69 = arith.cmpi slt, %67, %c1_i64 : i64
    assert %69, "dim must be smaller than inputRank"
    %70 = arith.index_cast %67 : i64 to index
    %71 = tensor.dim %64, %70 : tensor<?xi32>
    %72 = arith.index_cast %71 : index to i64
    %73 = arith.index_cast %72 : i64 to index
    %74 = arith.trunci %c0_i64 : i64 to i32
    %75 = linalg.init_tensor [%73] : tensor<?xi32>
    %76 = linalg.fill(%74, %75) : i32, tensor<?xi32> -> tensor<?xi32> 
    %77 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%64 : tensor<?xi32>) outs(%53 : tensor<?xi1>) {
    ^bb0(%arg3: i32, %arg4: i1):
      %108 = arith.trunci %c42_i64 : i64 to i32
      %109 = arith.cmpi sgt, %arg3, %108 : i32
      linalg.yield %109 : i1
    } -> tensor<?xi1>
    %78 = arith.cmpi eq, %30, %73 : index
    assert %78, "mismatched size for broadcast"
    assert %true, "mismatched size for broadcast"
    %79 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%77, %76, %64 : tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) outs(%63 : tensor<?xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):
      %108 = select %arg3, %arg4, %arg5 : i32
      linalg.yield %108 : i32
    } -> tensor<?xi32>
    %80 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%5 : tensor<?xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = arith.truncf %cst_1 : f64 to f32
      %109 = arith.divf %arg3, %108 : f32
      linalg.yield %109 : f32
    } -> tensor<?xf32>
    %81 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%80 : tensor<?xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = math.ceil %arg3 : f32
      linalg.yield %108 : f32
    } -> tensor<?xf32>
    %82 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%81 : tensor<?xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = arith.sitofp %c1_i64 : i64 to f32
      %109 = arith.mulf %108, %108 : f32
      %110 = arith.subf %arg3, %109 : f32
      linalg.yield %110 : f32
    } -> tensor<?xf32>
    %83 = linalg.init_tensor [%2] : tensor<?xi64>
    %84 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%82 : tensor<?xf32>) outs(%83 : tensor<?xi64>) {
    ^bb0(%arg3: f32, %arg4: i64):
      %108 = arith.fptosi %arg3 : f32 to i64
      linalg.yield %108 : i64
    } -> tensor<?xi64>
    %85 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%79 : tensor<?xi32>) outs(%63 : tensor<?xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %108 = arith.trunci %c5000_i64 : i64 to i32
      %109 = arith.muli %arg3, %108 : i32
      linalg.yield %109 : i32
    } -> tensor<?xi32>
    %86 = arith.cmpi eq, %2, %30 : index
    assert %86, "mismatched size for broadcast"
    %87 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%84, %85 : tensor<?xi64>, tensor<?xi32>) outs(%83 : tensor<?xi64>) {
    ^bb0(%arg3: i64, %arg4: i32, %arg5: i64):
      %108 = arith.extsi %arg4 : i32 to i64
      %109 = arith.muli %108, %c1_i64 : i64
      %110 = arith.addi %arg3, %109 : i64
      linalg.yield %110 : i64
    } -> tensor<?xi64>
    %88 = tensor.dim %87, %c0 : tensor<?xi64>
    %89 = linalg.init_tensor [%88] : tensor<?xf64>
    %90 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%87 : tensor<?xi64>) outs(%89 : tensor<?xf64>) {
    ^bb0(%arg3: i64, %arg4: f64):
      %108 = arith.index_cast %arg3 : i64 to index
      %109 = tensor.extract %cst_2[%108] : tensor<215000xf64>
      linalg.yield %109 : f64
    } -> tensor<?xf64>
    %91 = tensor.dim %87, %c0 : tensor<?xi64>
    %92 = linalg.init_tensor [%91] : tensor<?xf64>
    %93 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%87 : tensor<?xi64>) outs(%92 : tensor<?xf64>) {
    ^bb0(%arg3: i64, %arg4: f64):
      %108 = arith.index_cast %arg3 : i64 to index
      %109 = tensor.extract %cst_2[%108] : tensor<215000xf64>
      linalg.yield %109 : f64
    } -> tensor<?xf64>
    %94 = arith.cmpi eq, %88, %91 : index
    assert %94, "mismatched size for broadcast"
    %95 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%90, %93 : tensor<?xf64>, tensor<?xf64>) outs(%89 : tensor<?xf64>) {
    ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):
      %108 = arith.divf %arg3, %arg4 : f64
      linalg.yield %108 : f64
    } -> tensor<?xf64>
    %96 = linalg.init_tensor [%88] : tensor<?xf32>
    %97 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%95 : tensor<?xf64>) outs(%96 : tensor<?xf32>) {
    ^bb0(%arg3: f64, %arg4: f32):
      %108 = arith.truncf %arg3 : f64 to f32
      linalg.yield %108 : f32
    } -> tensor<?xf32>
    %98 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%97 : tensor<?xf32>) outs(%96 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = arith.truncf %cst_3 : f64 to f32
      %109 = arith.mulf %arg3, %108 : f32
      linalg.yield %109 : f32
    } -> tensor<?xf32>
    %99 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%5 : tensor<?xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %108 = arith.truncf %cst_4 : f64 to f32
      %109 = arith.mulf %arg3, %108 : f32
      linalg.yield %109 : f32
    } -> tensor<?xf32>
    %100 = arith.cmpi eq, %88, %2 : index
    assert %100, "mismatched size for broadcast"
    %101 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%98, %99 : tensor<?xf32>, tensor<?xf32>) outs(%96 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %108 = arith.sitofp %c1_i64 : i64 to f32
      %109 = arith.mulf %arg4, %108 : f32
      %110 = arith.addf %arg3, %109 : f32
      linalg.yield %110 : f32
    } -> tensor<?xf32>
    %102 = linalg.init_tensor [%91] : tensor<?xi1>
    %103 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%93 : tensor<?xf64>) outs(%102 : tensor<?xi1>) {
    ^bb0(%arg3: f64, %arg4: i1):
      %108 = arith.sitofp %c0_i64 : i64 to f64
      %109 = arith.cmpf ugt, %arg3, %108 : f64
      linalg.yield %109 : i1
    } -> tensor<?xi1>
    %104 = arith.cmpi eq, %91, %88 : index
    assert %104, "mismatched size for broadcast"
    %105 = arith.cmpi eq, %91, %2 : index
    assert %105, "mismatched size for broadcast"
    %106 = linalg.init_tensor [%91] : tensor<?xf32>
    %107 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%103, %101, %5 : tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) outs(%106 : tensor<?xf32>) {
    ^bb0(%arg3: i1, %arg4: f32, %arg5: f32, %arg6: f32):
      %108 = select %arg3, %arg4, %arg5 : f32
      linalg.yield %108 : f32
    } -> tensor<?xf32>
    return %107, %87 : tensor<?xf32>, tensor<?xi64>
  }
}

