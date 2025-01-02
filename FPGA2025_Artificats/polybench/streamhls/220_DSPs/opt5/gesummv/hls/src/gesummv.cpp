
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <hls_stream.h>
#include <hls_half.h>
#include <cassert>
#include <hls_math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

using namespace std;

void node0(
  hls::stream<float> v0[2],
  float v1[250]
) {
  #pragma HLS array_partition variable=v1 cyclic dim=1 factor=2

  loop0: for (int v2 = 0; v2 < 125; v2++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop1: for (int v3 = 0; v3 < 2; v3++) {
      float v4 = v0[v3].read();
      v1[(v3 + (v2 * 2))] = v4;
    }
  }
  return ;
}

void node1(
  hls::stream<float> v5[2],
  hls::stream<float> v6[2],
  hls::stream<float> v7[2],
  double v8,
  float v9
) {
  loop2: for (int v10 = 0; v10 < 125; v10++) {
    #pragma HLS pipeline II=1
    #pragma HLS loop_flatten
    loop3: for (int v11 = 0; v11 < 2; v11++) {
      float v12 = v5[v11].read();
      float v13 = v6[v11].read();
      float v14 = v8;
      float v15 = v13 * v14;
      float v16 = v12 * v9;
      float v17 = v16 + v15;
      v7[v11].write(v17);
    }
  }
  return ;
}

void node2(
  float v18[250][250],
  float v19[250],
  hls::stream<float> v20[2],
  float v21
) {
  #pragma HLS array_partition variable=v18 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v18 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v19 cyclic dim=1 factor=10

  float v22[250];
  #pragma HLS array_partition variable=v22 cyclic dim=1 factor=2

  loop4: for (int v23 = 0; v23 < 25; v23++) {
    loop5: for (int v24 = 0; v24 < 125; v24++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop6: for (int v25 = 0; v25 < 10; v25++) {
        loop7: for (int v26 = 0; v26 < 2; v26++) {
          float v27 = v18[(v26 + (v24 * 2))][(v25 + (v23 * 10))];
          float v28 = v19[(v25 + (v23 * 10))];
          if ((v25 + (v23 * 10)) == 0) {
            v22[(v26 + (v24 * 2))] = v21;
          }
          float v29 = v22[(v26 + (v24 * 2))];
          float v30 = v27 * v28;
          float v31 = v29 + v30;
          v22[(v26 + (v24 * 2))] = v31;
          if (((v25 + (v23 * 10)) - 249) == 0) {
            float v32 = v22[(v26 + (v24 * 2))];
            v20[v26].write(v32);
          }
        }
      }
    }
  }
  return ;
}

void node3(
  float v33[250][250],
  float v34[250],
  hls::stream<float> v35[2],
  float v36
) {
  #pragma HLS array_partition variable=v33 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v33 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v34 cyclic dim=1 factor=10

  float v37[250];
  #pragma HLS array_partition variable=v37 cyclic dim=1 factor=2

  loop8: for (int v38 = 0; v38 < 25; v38++) {
    loop9: for (int v39 = 0; v39 < 125; v39++) {
      #pragma HLS pipeline II=1
      #pragma HLS loop_flatten
      loop10: for (int v40 = 0; v40 < 10; v40++) {
        loop11: for (int v41 = 0; v41 < 2; v41++) {
          float v42 = v33[(v41 + (v39 * 2))][(v40 + (v38 * 10))];
          float v43 = v34[(v40 + (v38 * 10))];
          if ((v40 + (v38 * 10)) == 0) {
            v37[(v41 + (v39 * 2))] = v36;
          }
          float v44 = v37[(v41 + (v39 * 2))];
          float v45 = v42 * v43;
          float v46 = v44 + v45;
          v37[(v41 + (v39 * 2))] = v46;
          if (((v40 + (v38 * 10)) - 249) == 0) {
            float v47 = v37[(v41 + (v39 * 2))];
            v35[v41].write(v47);
          }
        }
      }
    }
  }
  return ;
}

void forward(
  float v48[250][250],
  float v49[250][250],
  float v50[250],
  float v51[250]
) {
	#pragma HLS DATAFLOW
  #pragma HLS array_partition variable=v48 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v48 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v49 cyclic dim=1 factor=2
  #pragma HLS array_partition variable=v49 cyclic dim=2 factor=10

  #pragma HLS array_partition variable=v50 cyclic dim=1 factor=10

  #pragma HLS array_partition variable=v51 cyclic dim=1 factor=2

  hls::stream<float> v52[2];
	#pragma HLS STREAM variable=v52 depth=125
  hls::stream<float> v53[2];
	#pragma HLS STREAM variable=v53 depth=125
  hls::stream<float> v54[2];
	#pragma HLS STREAM variable=v54 depth=125
  node3(v48, v50, v54, 0.000000);
  node2(v49, v50, v53, 0.000000);
  node1(v54, v53, v52, 1.200000, 1.500000);
  node0(v52, v51);
  return ;
}

