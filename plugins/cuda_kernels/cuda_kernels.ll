; ModuleID = 'builtin.module'
source_filename = "cuda_kernels"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_0 = addrspace(3) global [256 x float] undef, align 4
declare float @__nv_sinf(float)
declare float @__nv_cosf(float)

define void @rope(i8* %v0, i64 %v1, i8* %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, float %v7, i8* %v8, i64 %v9) #0 {
entry:
  %v10 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v11 = insertvalue { i8*, i64 } %v10, i64 %v1, 1
  %v12 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v13 = insertvalue { i8*, i64 } %v12, i64 %v3, 1
  %v14 = insertvalue { i8*, i64 } undef, i8* %v8, 0
  %v15 = insertvalue { i8*, i64 } %v14, i64 %v9, 1
  br label %bb0
bb0:
  %v16 = phi { i8*, i64 } [ %v11, %entry ]
  %v17 = phi { i8*, i64 } [ %v13, %entry ]
  %v18 = phi i32 [ %v4, %entry ]
  %v19 = phi i32 [ %v5, %entry ]
  %v20 = phi i32 [ %v6, %entry ]
  %v21 = phi float [ %v7, %entry ]
  %v22 = phi { i8*, i64 } [ %v15, %entry ]
  %v80 = alloca {  }, align 1
  %v23 = bitcast {  }* %v80 to i8*
  %v24 = getelementptr i8, i8* %v23, i64 0
  %v25 = call i64 @cuda_device____internal__index_1d(i8* %v24) #0
  br label %bb1
bb1:
  %v26 = mul i32 %v18, %v19
  %v27 = zext i32 %v26 to i64
  %v28 = icmp uge i64 %v25, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb3, label %bb2
bb2:
  br label %bb13
bb3:
  %v30 = zext i32 %v19 to i64
  %v31 = icmp eq i64 %v30, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb4, label %bb17
bb4:
  %v33 = udiv i64 %v25, %v30
  %v34 = urem i64 %v25, %v30
  %v35 = mul i64 %v33, %v30
  %v36 = zext i32 %v20 to i64
  %v37 = icmp uge i64 %v34, %v36
  %v38 = xor i1 %v37, 1
  br i1 %v38, label %bb7, label %bb5
bb5:
  %v39 = extractvalue { i8*, i64 } %v16, 1
  %v40 = icmp ult i64 %v25, %v39
  br i1 %v40, label %bb6, label %bb18
bb6:
  %v41 = extractvalue { i8*, i64 } %v16, 0
  %v81 = bitcast i8* %v41 to float*
  %v82 = getelementptr inbounds float, float* %v81, i64 %v25
  %v42 = bitcast float* %v82 to i8*
  %v83 = bitcast i8* %v42 to float*
  %v43 = load float, float* %v83, align 4
  %v44 = extractvalue { i8*, i64 } %v22, 0
  %v84 = bitcast i8* %v44 to float*
  %v85 = getelementptr inbounds float, float* %v84, i64 %v25
  %v45 = bitcast float* %v85 to i8*
  %v86 = bitcast i8* %v45 to float*
  store float %v43, float* %v86, align 4
  br label %bb13
bb7:
  %v46 = urem i64 %v34, 2
  %v47 = icmp eq i64 %v46, 1
  br i1 %v47, label %bb8, label %bb9
bb8:
  br label %bb13
bb9:
  %v48 = udiv i64 %v34, 2
  %v49 = extractvalue { i8*, i64 } %v17, 1
  %v50 = icmp ult i64 %v48, %v49
  br i1 %v50, label %bb10, label %bb19
bb10:
  %v51 = extractvalue { i8*, i64 } %v17, 0
  %v87 = bitcast i8* %v51 to float*
  %v88 = getelementptr inbounds float, float* %v87, i64 %v48
  %v52 = bitcast float* %v88 to i8*
  %v89 = bitcast i8* %v52 to float*
  %v53 = load float, float* %v89, align 4
  %v54 = fmul contract float %v21, %v53
  %v55 = call float @__nv_sinf(float %v54) #0
  br label %bb15
bb11:
  %v56 = extractvalue { i8*, i64 } %v16, 0
  %v90 = bitcast i8* %v56 to float*
  %v91 = getelementptr inbounds float, float* %v90, i64 %v74
  %v57 = bitcast float* %v91 to i8*
  %v92 = bitcast i8* %v57 to float*
  %v58 = load float, float* %v92, align 4
  %v59 = add i64 %v74, 1
  %v60 = icmp ult i64 %v59, %v75
  br i1 %v60, label %bb12, label %bb20
bb12:
  %v61 = extractvalue { i8*, i64 } %v16, 0
  %v93 = bitcast i8* %v61 to float*
  %v94 = getelementptr inbounds float, float* %v93, i64 %v59
  %v62 = bitcast float* %v94 to i8*
  %v95 = bitcast i8* %v62 to float*
  %v63 = load float, float* %v95, align 4
  %v64 = fmul contract float %v58, %v73
  %v65 = fmul contract float %v63, %v55
  %v66 = extractvalue { i8*, i64 } %v22, 0
  %v96 = bitcast i8* %v66 to float*
  %v97 = getelementptr inbounds float, float* %v96, i64 %v74
  %v67 = bitcast float* %v97 to i8*
  %v68 = fsub contract float %v64, %v65
  %v98 = bitcast i8* %v67 to float*
  store float %v68, float* %v98, align 4
  %v69 = fmul contract float %v58, %v55
  %v70 = fmul contract float %v63, %v73
  %v99 = bitcast i8* %v66 to float*
  %v100 = getelementptr inbounds float, float* %v99, i64 %v59
  %v71 = bitcast float* %v100 to i8*
  %v72 = fadd contract float %v69, %v70
  %v101 = bitcast i8* %v71 to float*
  store float %v72, float* %v101, align 4
  br label %bb14
bb13:
  br label %bb14
bb14:
  ret void
bb15:
  %v73 = call float @__nv_cosf(float %v54) #0
  br label %bb16
bb16:
  %v74 = add i64 %v35, %v34
  %v75 = extractvalue { i8*, i64 } %v16, 1
  %v76 = icmp ult i64 %v74, %v75
  br i1 %v76, label %bb11, label %bb21
bb17:
  unreachable
bb18:
  unreachable
bb19:
  unreachable
bb20:
  unreachable
bb21:
  unreachable
}

define void @kv_write_row(i8* %v0, i64 %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12) #0 {
entry:
  %v13 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v16 = insertvalue { i8*, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v18 = insertvalue { i8*, i64 } %v17, i64 %v5, 1
  br label %bb0
bb0:
  %v19 = phi { i8*, i64 } [ %v14, %entry ]
  %v20 = phi { i8*, i64 } [ %v16, %entry ]
  %v21 = phi { i8*, i64 } [ %v18, %entry ]
  %v22 = phi i32 [ %v6, %entry ]
  %v23 = phi i32 [ %v7, %entry ]
  %v24 = phi i32 [ %v8, %entry ]
  %v25 = phi i32 [ %v9, %entry ]
  %v26 = phi i32 [ %v10, %entry ]
  %v27 = phi i32 [ %v11, %entry ]
  %v28 = phi i32 [ %v12, %entry ]
  %v82 = alloca {  }, align 1
  %v29 = bitcast {  }* %v82 to i8*
  %v30 = getelementptr i8, i8* %v29, i64 0
  %v31 = call i64 @cuda_device____internal__index_1d(i8* %v30) #0
  br label %bb1
bb1:
  %v32 = trunc i64 %v31 to i32
  %v33 = icmp uge i32 %v32, %v26
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb3, label %bb2
bb2:
  br label %bb11
bb3:
  %v35 = icmp eq i32 %v27, 0
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb4, label %bb12
bb4:
  %v37 = udiv i32 %v23, %v27
  %v38 = urem i32 %v23, %v27
  %v39 = mul i32 %v22, %v25
  %v40 = add i32 %v39, %v37
  %v41 = zext i32 %v40 to i64
  %v42 = extractvalue { i8*, i64 } %v21, 1
  %v43 = icmp ult i64 %v41, %v42
  br i1 %v43, label %bb5, label %bb13
bb5:
  %v44 = extractvalue { i8*, i64 } %v21, 0
  %v83 = bitcast i8* %v44 to i32*
  %v84 = getelementptr inbounds i32, i32* %v83, i64 %v41
  %v45 = bitcast i32* %v84 to i8*
  %v85 = bitcast i8* %v45 to i32*
  %v46 = load i32, i32* %v85, align 4
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v26 to i64
  %v49 = mul i64 %v48, 2
  %v50 = icmp eq i32 %v24, 0
  br i1 %v50, label %bb7, label %bb6
bb6:
  %v51 = zext i32 %v27 to i64
  %v52 = mul i64 %v51, %v49
  br label %bb8
bb7:
  br label %bb8
bb8:
  %v53 = phi i64 [ %v52, %bb6 ], [ 0, %bb7 ]
  %v54 = zext i32 %v28 to i64
  %v55 = mul i64 %v47, %v54
  %v56 = add i64 %v55, %v53
  %v57 = zext i32 %v38 to i64
  %v58 = mul i64 %v57, %v49
  %v59 = add i64 %v56, %v58
  %v60 = zext i32 %v32 to i64
  %v61 = extractvalue { i8*, i64 } %v19, 1
  %v62 = icmp ult i64 %v60, %v61
  br i1 %v62, label %bb9, label %bb14
bb9:
  %v63 = extractvalue { i8*, i64 } %v19, 0
  %v86 = bitcast i8* %v63 to float*
  %v87 = getelementptr inbounds float, float* %v86, i64 %v60
  %v64 = bitcast float* %v87 to i8*
  %v88 = bitcast i8* %v64 to float*
  %v65 = load float, float* %v88, align 4
  %v66 = call i16 @cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits(float %v65) #0
  br label %bb10
bb10:
  %v67 = and i16 %v66, 255
  %v68 = trunc i16 %v67 to i8
  %v69 = trunc i32 8 to i16
  %v70 = and i16 %v69, 15
  %v71 = lshr i16 %v66, %v70
  %v72 = trunc i16 %v71 to i8
  %v73 = mul i64 %v60, 2
  %v74 = add i64 %v59, %v73
  %v75 = extractvalue { i8*, i64 } %v20, 0
  %v76 = getelementptr inbounds i8, i8* %v75, i64 %v74
  store i8 %v68, i8* %v76, align 1
  %v77 = add i64 %v74, 1
  %v78 = getelementptr inbounds i8, i8* %v75, i64 %v77
  store i8 %v72, i8* %v78, align 1
  br label %bb11
bb11:
  ret void
bb12:
  unreachable
bb13:
  unreachable
bb14:
  unreachable
}

declare float @__nv_expf(float)

define void @attention_heads(i8* %v0, i64 %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i8* %v11, i64 %v12) #0 {
entry:
  %v13 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v16 = insertvalue { i8*, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v18 = insertvalue { i8*, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { i8*, i64 } undef, i8* %v11, 0
  %v20 = insertvalue { i8*, i64 } %v19, i64 %v12, 1
  br label %bb0
bb0:
  %v21 = phi { i8*, i64 } [ %v14, %entry ]
  %v22 = phi { i8*, i64 } [ %v16, %entry ]
  %v23 = phi { i8*, i64 } [ %v18, %entry ]
  %v24 = phi i32 [ %v6, %entry ]
  %v25 = phi i32 [ %v7, %entry ]
  %v26 = phi i32 [ %v8, %entry ]
  %v27 = phi i32 [ %v9, %entry ]
  %v28 = phi float [ %v10, %entry ]
  %v29 = phi { i8*, i64 } [ %v20, %entry ]
  %v137 = alloca {  }, align 1
  %v30 = bitcast {  }* %v137 to i8*
  %v31 = getelementptr i8, i8* %v30, i64 0
  %v32 = call i64 @cuda_device____internal__index_1d(i8* %v31) #0
  br label %bb1
bb1:
  %v33 = trunc i64 %v32 to i32
  %v34 = icmp uge i32 %v33, %v24
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb3, label %bb2
bb2:
  br label %bb35
bb3:
  %v36 = zext i32 %v26 to i64
  %v37 = zext i32 %v27 to i64
  %v38 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCsh6jSs53Rst6_12cuda_kernels(i32 %v25, i32 1) #0
  br label %bb4
bb4:
  %v39 = icmp eq i32 %v38, 0
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb5, label %bb38
bb5:
  %v41 = udiv i32 %v24, %v38
  %v42 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCsh6jSs53Rst6_12cuda_kernels(i32 %v41, i32 1) #0
  br label %bb6
bb6:
  %v43 = icmp eq i32 %v42, 0
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb7, label %bb39
bb7:
  %v45 = udiv i32 %v33, %v42
  %v46 = zext i32 %v45 to i64
  %v47 = zext i32 %v33 to i64
  %v48 = mul i64 %v47, %v36
  br label %bb8
bb8:
  %v49 = phi i64 [ 0, %bb7 ], [ %v55, %bb9 ]
  %v50 = icmp ult i64 %v49, %v36
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb10, label %bb9
bb9:
  %v52 = add i64 %v48, %v49
  %v53 = extractvalue { i8*, i64 } %v29, 0
  %v138 = bitcast i8* %v53 to float*
  %v139 = getelementptr inbounds float, float* %v138, i64 %v52
  %v54 = bitcast float* %v139 to i8*
  %v140 = bitcast i8* %v54 to float*
  store float 0.0, float* %v140, align 4
  %v55 = add i64 %v49, 1
  br label %bb8
bb10:
  br label %bb11
bb11:
  %v56 = phi float [ 0.0, %bb10 ], [ %v94, %bb27 ]
  %v57 = phi float [ 0.0, %bb10 ], [ %v129, %bb27 ]
  %v58 = phi i1 [ 0, %bb10 ], [ 1, %bb27 ]
  %v59 = phi i64 [ 0, %bb10 ], [ %v115, %bb27 ]
  %v60 = icmp ult i64 %v59, %v37
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb28, label %bb12
bb12:
  %v62 = zext i32 %v38 to i64
  %v63 = mul i64 %v59, %v62
  %v64 = mul i64 %v63, %v36
  %v65 = mul i64 %v46, %v36
  %v66 = add i64 %v64, %v65
  br label %bb13
bb13:
  %v67 = phi float [ 0.0, %bb12 ], [ %v84, %bb16 ]
  %v68 = phi i64 [ 0, %bb12 ], [ %v85, %bb16 ]
  %v69 = icmp ult i64 %v68, %v36
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb17, label %bb14
bb14:
  %v71 = add i64 %v48, %v68
  %v72 = extractvalue { i8*, i64 } %v21, 1
  %v73 = icmp ult i64 %v71, %v72
  br i1 %v73, label %bb15, label %bb40
bb15:
  %v74 = extractvalue { i8*, i64 } %v21, 0
  %v141 = bitcast i8* %v74 to float*
  %v142 = getelementptr inbounds float, float* %v141, i64 %v71
  %v75 = bitcast float* %v142 to i8*
  %v143 = bitcast i8* %v75 to float*
  %v76 = load float, float* %v143, align 4
  %v77 = add i64 %v66, %v68
  %v78 = extractvalue { i8*, i64 } %v22, 1
  %v79 = icmp ult i64 %v77, %v78
  br i1 %v79, label %bb16, label %bb41
bb16:
  %v80 = extractvalue { i8*, i64 } %v22, 0
  %v144 = bitcast i8* %v80 to float*
  %v145 = getelementptr inbounds float, float* %v144, i64 %v77
  %v81 = bitcast float* %v145 to i8*
  %v146 = bitcast i8* %v81 to float*
  %v82 = load float, float* %v146, align 4
  %v83 = fmul contract float %v76, %v82
  %v84 = fadd contract float %v67, %v83
  %v85 = add i64 %v68, 1
  br label %bb13
bb17:
  %v86 = fmul contract float %v67, %v28
  %v87 = xor i1 %v58, 1
  br i1 %v87, label %bb19, label %bb18
bb18:
  %v88 = fcmp ogt float %v86, %v56
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb21, label %bb20
bb19:
  br label %bb23
bb20:
  %v90 = fsub contract float %v56, %v86
  %v91 = call float @__nv_expf(float %v90) #0
  br label %bb36
bb21:
  br label %bb22
bb22:
  %v92 = phi float [ %v56, %bb21 ], [ %v86, %bb36 ]
  %v93 = phi float [ 1.0, %bb21 ], [ %v91, %bb36 ]
  br label %bb23
bb23:
  %v94 = phi float [ %v86, %bb19 ], [ %v92, %bb22 ]
  %v95 = phi float [ 0.0, %bb19 ], [ %v93, %bb22 ]
  %v96 = fsub contract float %v86, %v94
  %v97 = call float @__nv_expf(float %v96) #0
  br label %bb37
bb24:
  %v98 = phi i64 [ %v114, %bb26 ], [ 0, %bb37 ]
  %v99 = icmp ult i64 %v98, %v36
  %v100 = xor i1 %v99, 1
  br i1 %v100, label %bb27, label %bb25
bb25:
  %v101 = add i64 %v48, %v98
  %v102 = extractvalue { i8*, i64 } %v29, 0
  %v147 = bitcast i8* %v102 to float*
  %v148 = getelementptr inbounds float, float* %v147, i64 %v101
  %v103 = bitcast float* %v148 to i8*
  %v149 = bitcast i8* %v103 to float*
  %v104 = load float, float* %v149, align 4
  %v105 = fmul contract float %v104, %v95
  %v106 = add i64 %v132, %v98
  %v107 = extractvalue { i8*, i64 } %v23, 1
  %v108 = icmp ult i64 %v106, %v107
  br i1 %v108, label %bb26, label %bb42
bb26:
  %v109 = extractvalue { i8*, i64 } %v23, 0
  %v150 = bitcast i8* %v109 to float*
  %v151 = getelementptr inbounds float, float* %v150, i64 %v106
  %v110 = bitcast float* %v151 to i8*
  %v152 = bitcast i8* %v110 to float*
  %v111 = load float, float* %v152, align 4
  %v112 = fmul contract float %v97, %v111
  %v113 = fadd contract float %v105, %v112
  %v153 = bitcast i8* %v103 to float*
  store float %v113, float* %v153, align 4
  %v114 = add i64 %v98, 1
  br label %bb24
bb27:
  %v115 = add i64 %v59, 1
  br label %bb11
bb28:
  %v116 = fcmp ogt float %v57, 0.0
  %v117 = xor i1 %v116, 1
  br i1 %v117, label %bb30, label %bb29
bb29:
  %v118 = fdiv contract float 1.0, %v57
  br label %bb31
bb30:
  br label %bb34
bb31:
  %v119 = phi i64 [ 0, %bb29 ], [ %v127, %bb32 ]
  %v120 = icmp ult i64 %v119, %v36
  %v121 = xor i1 %v120, 1
  br i1 %v121, label %bb33, label %bb32
bb32:
  %v122 = add i64 %v48, %v119
  %v123 = extractvalue { i8*, i64 } %v29, 0
  %v154 = bitcast i8* %v123 to float*
  %v155 = getelementptr inbounds float, float* %v154, i64 %v122
  %v124 = bitcast float* %v155 to i8*
  %v156 = bitcast i8* %v124 to float*
  %v125 = load float, float* %v156, align 4
  %v126 = fmul contract float %v125, %v118
  %v157 = bitcast i8* %v124 to float*
  store float %v126, float* %v157, align 4
  %v127 = add i64 %v119, 1
  br label %bb31
bb33:
  br label %bb34
bb34:
  br label %bb35
bb35:
  ret void
bb36:
  br label %bb22
bb37:
  %v128 = fmul contract float %v57, %v95
  %v129 = fadd contract float %v128, %v97
  %v130 = mul i64 %v59, %v62
  %v131 = mul i64 %v130, %v36
  %v132 = add i64 %v131, %v65
  br label %bb24
bb38:
  unreachable
bb39:
  unreachable
bb40:
  unreachable
bb41:
  unreachable
bb42:
  unreachable
}

define void @silu_gate(i8* %v0, i64 %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v11 = insertvalue { i8*, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v14 = phi { i8*, i64 } [ %v11, %entry ]
  %v52 = alloca {  }, align 1
  %v15 = bitcast {  }* %v52 to i8*
  %v16 = getelementptr i8, i8* %v15, i64 0
  %v17 = call i64 @cuda_device____internal__index_1d(i8* %v16) #0
  br label %bb1
bb1:
  %v18 = extractvalue { i8*, i64 } %v14, 1
  %v19 = icmp ult i64 %v17, %v18
  %v20 = xor i1 %v19, 1
  br i1 %v20, label %bb8, label %bb7
bb2:
  %v21 = extractvalue { i8, i8* } %v41, 1
  %v22 = extractvalue { i8*, i64 } %v12, 1
  %v23 = icmp ult i64 %v17, %v22
  br i1 %v23, label %bb3, label %bb13
bb3:
  %v24 = extractvalue { i8*, i64 } %v12, 0
  %v53 = bitcast i8* %v24 to float*
  %v54 = getelementptr inbounds float, float* %v53, i64 %v17
  %v25 = bitcast float* %v54 to i8*
  %v55 = bitcast i8* %v25 to float*
  %v26 = load float, float* %v55, align 4
  %v27 = extractvalue { i8*, i64 } %v13, 1
  %v28 = icmp ult i64 %v17, %v27
  br i1 %v28, label %bb4, label %bb14
bb4:
  %v29 = extractvalue { i8*, i64 } %v13, 0
  %v56 = bitcast i8* %v29 to float*
  %v57 = getelementptr inbounds float, float* %v56, i64 %v17
  %v30 = bitcast float* %v57 to i8*
  %v58 = bitcast i8* %v30 to float*
  %v31 = load float, float* %v58, align 4
  %v32 = bitcast float %v26 to i32
  %v33 = xor i32 %v32, 2147483648
  %v34 = bitcast i32 %v33 to float
  %v35 = call float @__nv_expf(float %v34) #0
  br label %bb11
bb5:
  br label %bb6
bb6:
  ret void
bb7:
  %v36 = extractvalue { i8*, i64 } %v14, 0
  %v59 = bitcast i8* %v36 to float*
  %v60 = getelementptr inbounds float, float* %v59, i64 %v17
  %v37 = bitcast float* %v60 to i8*
  %v38 = insertvalue { i8, i8* } undef, i8 1, 0
  %v39 = insertvalue { i8, i8* } %v38, i8* %v37, 1
  br label %bb9
bb8:
  %v40 = insertvalue { i8, i8* } undef, i8 0, 0
  br label %bb9
bb9:
  %v41 = phi { i8, i8* } [ %v39, %bb7 ], [ %v40, %bb8 ]
  %v42 = extractvalue { i8, i8* } %v41, 0
  %v43 = zext i8 %v42 to i64
  %v44 = icmp eq i64 %v43, 1
  br i1 %v44, label %bb2, label %bb10
bb10:
  %v45 = icmp eq i64 %v43, 0
  br i1 %v45, label %bb5, label %bb12
bb11:
  %v46 = fadd contract float 1.0, %v35
  %v47 = fdiv contract float %v26, %v46
  %v48 = fmul contract float %v47, %v31
  %v63 = bitcast i8* %v21 to float*
  store float %v48, float* %v63, align 4
  br label %bb6
bb12:
  unreachable
bb13:
  unreachable
bb14:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() #0
declare float @__nv_sqrtf(float)

define void @rmsnorm_group(i8* %v0, i64 %v1, i8* %v2, i64 %v3, float %v4, i32 %v5, i32 %v6, i8* %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v10 = insertvalue { i8*, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v12 = insertvalue { i8*, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { i8*, i64 } undef, i8* %v7, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { i8*, i64 } [ %v10, %entry ]
  %v16 = phi { i8*, i64 } [ %v12, %entry ]
  %v17 = phi float [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { i8*, i64 } [ %v14, %entry ]
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v23 = mul i32 %v22, %v19
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v18 to i64
  %v26 = zext i32 %v21 to i64
  br label %bb3
bb3:
  %v27 = phi float [ 0.0, %bb2 ], [ %v38, %bb5 ]
  %v28 = phi i64 [ %v26, %bb2 ], [ %v39, %bb5 ]
  %v29 = icmp ult i64 %v28, %v25
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb6, label %bb4
bb4:
  %v31 = add i64 %v24, %v28
  %v32 = extractvalue { i8*, i64 } %v15, 1
  %v33 = icmp ult i64 %v31, %v32
  br i1 %v33, label %bb5, label %bb25
bb5:
  %v34 = extractvalue { i8*, i64 } %v15, 0
  %v90 = bitcast i8* %v34 to float*
  %v91 = getelementptr inbounds float, float* %v90, i64 %v31
  %v35 = bitcast float* %v91 to i8*
  %v92 = bitcast i8* %v35 to float*
  %v36 = load float, float* %v92, align 4
  %v37 = fmul contract float %v36, %v36
  %v38 = fadd contract float %v27, %v37
  %v39 = add i64 %v28, 256
  br label %bb3
bb6:
  %v40 = bitcast [256 x float] addrspace(3)* @__shared_mem_0 to i8 addrspace(3)*
  %v41 = zext i32 %v21 to i64
  %v93 = bitcast i8 addrspace(3)* %v40 to float addrspace(3)*
  %v94 = getelementptr inbounds float, float addrspace(3)* %v93, i64 %v41
  %v42 = bitcast float addrspace(3)* %v94 to i8 addrspace(3)*
  br label %bb7
bb7:
  %v95 = bitcast i8 addrspace(3)* %v42 to float addrspace(3)*
  store float %v27, float addrspace(3)* %v95, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb8
bb8:
  br label %bb9
bb9:
  %v44 = phi i32 [ 128, %bb8 ], [ %v58, %bb16 ]
  %v45 = icmp ugt i32 %v44, 0
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb17, label %bb10
bb10:
  %v47 = icmp ult i32 %v21, %v44
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb14, label %bb11
bb11:
  %v49 = getelementptr i8, i8 addrspace(3)* %v40, i64 0
  %v50 = add i32 %v21, %v44
  %v51 = zext i32 %v50 to i64
  %v96 = bitcast i8 addrspace(3)* %v49 to float addrspace(3)*
  %v97 = getelementptr inbounds float, float addrspace(3)* %v96, i64 %v51
  %v52 = bitcast float addrspace(3)* %v97 to i8 addrspace(3)*
  br label %bb12
bb12:
  %v98 = bitcast i8 addrspace(3)* %v52 to float addrspace(3)*
  %v53 = load float, float addrspace(3)* %v98, align 4
  %v99 = bitcast i8 addrspace(3)* %v40 to float addrspace(3)*
  %v100 = getelementptr inbounds float, float addrspace(3)* %v99, i64 %v41
  %v54 = bitcast float addrspace(3)* %v100 to i8 addrspace(3)*
  br label %bb13
bb13:
  %v101 = bitcast i8 addrspace(3)* %v54 to float addrspace(3)*
  %v55 = load float, float addrspace(3)* %v101, align 4
  %v56 = fadd contract float %v55, %v53
  %v102 = bitcast i8 addrspace(3)* %v54 to float addrspace(3)*
  store float %v56, float addrspace(3)* %v102, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier0() #0
  br label %bb16
bb16:
  %v58 = udiv i32 %v44, 2
  br label %bb9
bb17:
  %v59 = getelementptr i8, i8 addrspace(3)* %v40, i64 0
  %v103 = bitcast i8 addrspace(3)* %v59 to float addrspace(3)*
  %v104 = getelementptr inbounds float, float addrspace(3)* %v103, i64 0
  %v60 = bitcast float addrspace(3)* %v104 to i8 addrspace(3)*
  br label %bb18
bb18:
  %v105 = bitcast i8 addrspace(3)* %v60 to float addrspace(3)*
  %v61 = load float, float addrspace(3)* %v105, align 4
  %v62 = uitofp i32 %v18 to float
  %v63 = fdiv contract float %v61, %v62
  %v64 = fadd contract float %v63, %v17
  %v65 = call float @__nv_sqrtf(float %v64) #0
  br label %bb24
bb19:
  %v66 = phi i64 [ %v85, %bb22 ], [ %v41, %bb24 ]
  %v67 = icmp ult i64 %v66, %v25
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb23, label %bb20
bb20:
  %v69 = add i64 %v24, %v66
  %v70 = extractvalue { i8*, i64 } %v15, 1
  %v71 = icmp ult i64 %v69, %v70
  br i1 %v71, label %bb21, label %bb26
bb21:
  %v72 = extractvalue { i8*, i64 } %v15, 0
  %v106 = bitcast i8* %v72 to float*
  %v107 = getelementptr inbounds float, float* %v106, i64 %v69
  %v73 = bitcast float* %v107 to i8*
  %v108 = bitcast i8* %v73 to float*
  %v74 = load float, float* %v108, align 4
  %v75 = fmul contract float %v74, %v86
  %v76 = extractvalue { i8*, i64 } %v16, 1
  %v77 = icmp ult i64 %v66, %v76
  br i1 %v77, label %bb22, label %bb27
bb22:
  %v78 = extractvalue { i8*, i64 } %v16, 0
  %v109 = bitcast i8* %v78 to float*
  %v110 = getelementptr inbounds float, float* %v109, i64 %v66
  %v79 = bitcast float* %v110 to i8*
  %v111 = bitcast i8* %v79 to float*
  %v80 = load float, float* %v111, align 4
  %v81 = add i64 %v24, %v66
  %v82 = extractvalue { i8*, i64 } %v20, 0
  %v112 = bitcast i8* %v82 to float*
  %v113 = getelementptr inbounds float, float* %v112, i64 %v81
  %v83 = bitcast float* %v113 to i8*
  %v84 = fmul contract float %v75, %v80
  %v114 = bitcast i8* %v83 to float*
  store float %v84, float* %v114, align 4
  %v85 = add i64 %v66, 256
  br label %bb19
bb23:
  ret void
bb24:
  %v86 = fdiv contract float 1.0, %v65
  br label %bb19
bb25:
  unreachable
bb26:
  unreachable
bb27:
  unreachable
}

define void @q4k_gemv_row(i8* %v0, i64 %v1, i8* %v2, i64 %v3, i32 %v4, i32 %v5, i8* %v6, i64 %v7) #0 {
entry:
  %v8 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v11 = insertvalue { i8*, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { i8*, i64 } undef, i8* %v6, 0
  %v13 = insertvalue { i8*, i64 } %v12, i64 %v7, 1
  br label %bb0
bb0:
  %v14 = phi { i8*, i64 } [ %v9, %entry ]
  %v15 = phi { i8*, i64 } [ %v11, %entry ]
  %v16 = phi i32 [ %v4, %entry ]
  %v17 = phi i32 [ %v5, %entry ]
  %v18 = phi { i8*, i64 } [ %v13, %entry ]
  %v253 = alloca {  }, align 1
  %v19 = bitcast {  }* %v253 to i8*
  %v254 = alloca [2 x i8], align 1
  %v20 = bitcast [2 x i8]* %v254 to i8*
  %v255 = alloca [2 x i8], align 1
  %v21 = bitcast [2 x i8]* %v255 to i8*
  %v256 = alloca [8 x i8], align 1
  %v22 = bitcast [8 x i8]* %v256 to i8*
  %v257 = alloca [8 x i8], align 1
  %v23 = bitcast [8 x i8]* %v257 to i8*
  %v24 = getelementptr i8, i8* %v19, i64 0
  %v25 = call i64 @cuda_device____internal__index_1d(i8* %v24) #0
  br label %bb1
bb1:
  %v26 = zext i32 %v16 to i64
  %v27 = icmp uge i64 %v25, %v26
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb3, label %bb2
bb2:
  br label %bb52
bb3:
  %v29 = mul i32 %v17, 144
  %v30 = zext i32 %v29 to i64
  %v31 = mul i64 %v25, %v30
  br label %bb4
bb4:
  %v32 = phi float [ 0.0, %bb3 ], [ %v164, %bb47 ]
  %v33 = phi i32 [ 0, %bb3 ], [ %v235, %bb47 ]
  %v34 = icmp ult i32 %v33, %v17
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb48, label %bb5
bb5:
  %v36 = zext i32 %v33 to i64
  %v37 = mul i64 %v36, 144
  %v38 = add i64 %v31, %v37
  %v39 = extractvalue { i8*, i64 } %v14, 1
  %v40 = icmp ult i64 %v38, %v39
  br i1 %v40, label %bb6, label %bb58
bb6:
  %v41 = extractvalue { i8*, i64 } %v14, 0
  %v42 = getelementptr inbounds i8, i8* %v41, i64 %v38
  %v43 = load i8, i8* %v42, align 1
  %v44 = add i64 %v38, 1
  %v45 = icmp ult i64 %v44, %v39
  br i1 %v45, label %bb7, label %bb59
bb7:
  %v46 = extractvalue { i8*, i64 } %v14, 0
  %v47 = getelementptr inbounds i8, i8* %v46, i64 %v44
  %v48 = load i8, i8* %v47, align 1
  %v258 = bitcast i8* %v20 to [2 x i8]*
  %v49 = getelementptr inbounds [2 x i8], [2 x i8]* %v258, i32 0, i64 0
  store i8 %v43, i8* %v49, align 1
  %v259 = bitcast i8* %v20 to [2 x i8]*
  %v50 = getelementptr inbounds [2 x i8], [2 x i8]* %v259, i32 0, i64 1
  store i8 %v48, i8* %v50, align 1
  %v260 = bitcast i8* %v20 to [2 x i8]*
  %v51 = load [2 x i8], [2 x i8]* %v260, align 1
  %v261 = alloca [2 x i8], align 2
  %v52 = bitcast [2 x i8]* %v261 to i8*
  %v262 = bitcast i8* %v52 to [2 x i8]*
  store [2 x i8] %v51, [2 x i8]* %v262, align 2
  %v263 = bitcast i8* %v52 to i16*
  %v53 = load i16, i16* %v263, align 2
  %v54 = add i64 %v38, 2
  %v55 = icmp ult i64 %v54, %v39
  br i1 %v55, label %bb8, label %bb60
bb8:
  %v56 = extractvalue { i8*, i64 } %v14, 0
  %v57 = getelementptr inbounds i8, i8* %v56, i64 %v54
  %v58 = load i8, i8* %v57, align 1
  %v59 = add i64 %v38, 3
  %v60 = icmp ult i64 %v59, %v39
  br i1 %v60, label %bb9, label %bb61
bb9:
  %v61 = extractvalue { i8*, i64 } %v14, 0
  %v62 = getelementptr inbounds i8, i8* %v61, i64 %v59
  %v63 = load i8, i8* %v62, align 1
  %v264 = bitcast i8* %v21 to [2 x i8]*
  %v64 = getelementptr inbounds [2 x i8], [2 x i8]* %v264, i32 0, i64 0
  store i8 %v58, i8* %v64, align 1
  %v265 = bitcast i8* %v21 to [2 x i8]*
  %v65 = getelementptr inbounds [2 x i8], [2 x i8]* %v265, i32 0, i64 1
  store i8 %v63, i8* %v65, align 1
  %v266 = bitcast i8* %v21 to [2 x i8]*
  %v66 = load [2 x i8], [2 x i8]* %v266, align 1
  %v267 = alloca [2 x i8], align 2
  %v67 = bitcast [2 x i8]* %v267 to i8*
  %v268 = bitcast i8* %v67 to [2 x i8]*
  store [2 x i8] %v66, [2 x i8]* %v268, align 2
  %v269 = bitcast i8* %v67 to i16*
  %v68 = load i16, i16* %v269, align 2
  %v69 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v53) #0
  br label %bb10
bb10:
  %v70 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v68) #0
  br label %bb11
bb11:
  %v71 = add i64 %v38, 4
  %v72 = icmp ult i64 %v71, %v39
  br i1 %v72, label %bb12, label %bb62
bb12:
  %v73 = extractvalue { i8*, i64 } %v14, 0
  %v74 = getelementptr inbounds i8, i8* %v73, i64 %v71
  %v75 = load i8, i8* %v74, align 1
  %v76 = add i64 %v38, 5
  %v77 = icmp ult i64 %v76, %v39
  br i1 %v77, label %bb13, label %bb63
bb13:
  %v78 = extractvalue { i8*, i64 } %v14, 0
  %v79 = getelementptr inbounds i8, i8* %v78, i64 %v76
  %v80 = load i8, i8* %v79, align 1
  %v81 = add i64 %v38, 6
  %v82 = icmp ult i64 %v81, %v39
  br i1 %v82, label %bb14, label %bb64
bb14:
  %v83 = extractvalue { i8*, i64 } %v14, 0
  %v84 = getelementptr inbounds i8, i8* %v83, i64 %v81
  %v85 = load i8, i8* %v84, align 1
  %v86 = add i64 %v38, 7
  %v87 = icmp ult i64 %v86, %v39
  br i1 %v87, label %bb15, label %bb65
bb15:
  %v88 = extractvalue { i8*, i64 } %v14, 0
  %v89 = getelementptr inbounds i8, i8* %v88, i64 %v86
  %v90 = load i8, i8* %v89, align 1
  %v91 = add i64 %v38, 8
  %v92 = icmp ult i64 %v91, %v39
  br i1 %v92, label %bb16, label %bb66
bb16:
  %v93 = extractvalue { i8*, i64 } %v14, 0
  %v94 = getelementptr inbounds i8, i8* %v93, i64 %v91
  %v95 = load i8, i8* %v94, align 1
  %v96 = add i64 %v38, 9
  %v97 = icmp ult i64 %v96, %v39
  br i1 %v97, label %bb17, label %bb67
bb17:
  %v98 = extractvalue { i8*, i64 } %v14, 0
  %v99 = getelementptr inbounds i8, i8* %v98, i64 %v96
  %v100 = load i8, i8* %v99, align 1
  %v101 = add i64 %v38, 10
  %v102 = icmp ult i64 %v101, %v39
  br i1 %v102, label %bb18, label %bb68
bb18:
  %v103 = extractvalue { i8*, i64 } %v14, 0
  %v104 = getelementptr inbounds i8, i8* %v103, i64 %v101
  %v105 = load i8, i8* %v104, align 1
  %v106 = add i64 %v38, 11
  %v107 = icmp ult i64 %v106, %v39
  br i1 %v107, label %bb19, label %bb69
bb19:
  %v108 = extractvalue { i8*, i64 } %v14, 0
  %v109 = getelementptr inbounds i8, i8* %v108, i64 %v106
  %v110 = load i8, i8* %v109, align 1
  %v111 = add i64 %v38, 12
  %v112 = icmp ult i64 %v111, %v39
  br i1 %v112, label %bb20, label %bb70
bb20:
  %v113 = extractvalue { i8*, i64 } %v14, 0
  %v114 = getelementptr inbounds i8, i8* %v113, i64 %v111
  %v115 = load i8, i8* %v114, align 1
  %v116 = add i64 %v38, 13
  %v117 = icmp ult i64 %v116, %v39
  br i1 %v117, label %bb21, label %bb71
bb21:
  %v118 = extractvalue { i8*, i64 } %v14, 0
  %v119 = getelementptr inbounds i8, i8* %v118, i64 %v116
  %v120 = load i8, i8* %v119, align 1
  %v121 = add i64 %v38, 14
  %v122 = icmp ult i64 %v121, %v39
  br i1 %v122, label %bb22, label %bb72
bb22:
  %v123 = extractvalue { i8*, i64 } %v14, 0
  %v124 = getelementptr inbounds i8, i8* %v123, i64 %v121
  %v125 = load i8, i8* %v124, align 1
  %v126 = add i64 %v38, 15
  %v127 = icmp ult i64 %v126, %v39
  br i1 %v127, label %bb23, label %bb73
bb23:
  %v128 = extractvalue { i8*, i64 } %v14, 0
  %v129 = getelementptr inbounds i8, i8* %v128, i64 %v126
  %v130 = load i8, i8* %v129, align 1
  %v131 = call { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v75, i8 %v80, i8 %v85, i8 %v90, i8 %v95, i8 %v100, i8 %v105, i8 %v110, i8 %v115, i8 %v120, i8 %v125, i8 %v130) #0
  br label %bb24
bb24:
  %v132 = extractvalue { [8 x i8], [8 x i8] } %v131, 0
  %v270 = bitcast i8* %v22 to [8 x i8]*
  store [8 x i8] %v132, [8 x i8]* %v270, align 1
  %v133 = extractvalue { [8 x i8], [8 x i8] } %v131, 1
  %v271 = bitcast i8* %v23 to [8 x i8]*
  store [8 x i8] %v133, [8 x i8]* %v271, align 1
  %v134 = add i64 %v38, 16
  %v135 = zext i32 %v33 to i64
  %v136 = mul i64 %v135, 256
  br label %bb25
bb25:
  %v137 = phi float [ 0.0, %bb24 ], [ %v160, %bb31 ]
  %v138 = phi i64 [ 0, %bb24 ], [ %v161, %bb31 ]
  %v139 = icmp ult i64 %v138, 8
  %v140 = xor i1 %v139, 1
  br i1 %v140, label %bb32, label %bb26
bb26:
  br label %bb27
bb27:
  %v141 = phi float [ 0.0, %bb26 ], [ %v153, %bb29 ]
  %v142 = phi i64 [ 0, %bb26 ], [ %v154, %bb29 ]
  %v143 = icmp ult i64 %v142, 32
  %v144 = xor i1 %v143, 1
  br i1 %v144, label %bb30, label %bb28
bb28:
  %v145 = mul i64 %v138, 32
  %v146 = add i64 %v136, %v145
  %v147 = add i64 %v146, %v142
  %v148 = extractvalue { i8*, i64 } %v15, 1
  %v149 = icmp ult i64 %v147, %v148
  br i1 %v149, label %bb29, label %bb74
bb29:
  %v150 = extractvalue { i8*, i64 } %v15, 0
  %v272 = bitcast i8* %v150 to float*
  %v273 = getelementptr inbounds float, float* %v272, i64 %v147
  %v151 = bitcast float* %v273 to i8*
  %v274 = bitcast i8* %v151 to float*
  %v152 = load float, float* %v274, align 4
  %v153 = fadd contract float %v141, %v152
  %v154 = add i64 %v142, 1
  br label %bb27
bb30:
  %v155 = icmp ult i64 %v138, 8
  br i1 %v155, label %bb31, label %bb75
bb31:
  %v275 = bitcast i8* %v23 to [8 x i8]*
  %v156 = getelementptr inbounds [8 x i8], [8 x i8]* %v275, i32 0, i64 %v138
  %v157 = load i8, i8* %v156, align 1
  %v158 = uitofp i8 %v157 to float
  %v159 = fmul contract float %v158, %v141
  %v160 = fadd contract float %v137, %v159
  %v161 = add i64 %v138, 1
  br label %bb25
bb32:
  %v162 = fmul contract float %v70, %v137
  %v163 = fsub contract float %v32, %v162
  br label %bb33
bb33:
  %v164 = phi float [ %v163, %bb32 ], [ %v232, %bb46 ]
  %v165 = phi i64 [ 0, %bb32 ], [ %v206, %bb46 ]
  %v166 = phi i64 [ 0, %bb32 ], [ %v233, %bb46 ]
  %v167 = phi i64 [ 0, %bb32 ], [ %v234, %bb46 ]
  %v168 = icmp ult i64 %v167, 4
  %v169 = xor i1 %v168, 1
  br i1 %v169, label %bb47, label %bb34
bb34:
  %v170 = mul i64 %v167, 32
  %v171 = add i64 %v134, %v170
  %v172 = icmp ult i64 %v165, 8
  br i1 %v172, label %bb35, label %bb76
bb35:
  %v276 = bitcast i8* %v22 to [8 x i8]*
  %v173 = getelementptr inbounds [8 x i8], [8 x i8]* %v276, i32 0, i64 %v165
  %v174 = load i8, i8* %v173, align 1
  %v175 = uitofp i8 %v174 to float
  %v176 = add i64 %v165, 1
  br label %bb36
bb36:
  %v177 = phi float [ 0.0, %bb35 ], [ %v196, %bb39 ]
  %v178 = phi i64 [ 0, %bb35 ], [ %v197, %bb39 ]
  %v179 = icmp ult i64 %v178, 32
  %v180 = xor i1 %v179, 1
  br i1 %v180, label %bb40, label %bb37
bb37:
  %v181 = add i64 %v171, %v178
  %v182 = icmp ult i64 %v181, %v39
  br i1 %v182, label %bb38, label %bb77
bb38:
  %v183 = extractvalue { i8*, i64 } %v14, 0
  %v184 = getelementptr inbounds i8, i8* %v183, i64 %v181
  %v185 = load i8, i8* %v184, align 1
  %v186 = and i8 %v185, 15
  %v187 = uitofp i8 %v186 to float
  %v188 = add i64 %v136, %v166
  %v189 = add i64 %v188, %v178
  %v190 = extractvalue { i8*, i64 } %v15, 1
  %v191 = icmp ult i64 %v189, %v190
  br i1 %v191, label %bb39, label %bb78
bb39:
  %v192 = extractvalue { i8*, i64 } %v15, 0
  %v277 = bitcast i8* %v192 to float*
  %v278 = getelementptr inbounds float, float* %v277, i64 %v189
  %v193 = bitcast float* %v278 to i8*
  %v279 = bitcast i8* %v193 to float*
  %v194 = load float, float* %v279, align 4
  %v195 = fmul contract float %v187, %v194
  %v196 = fadd contract float %v177, %v195
  %v197 = add i64 %v178, 1
  br label %bb36
bb40:
  %v198 = fmul contract float %v69, %v175
  %v199 = fmul contract float %v198, %v177
  %v200 = fadd contract float %v164, %v199
  %v201 = add i64 %v166, 32
  %v202 = icmp ult i64 %v176, 8
  br i1 %v202, label %bb41, label %bb79
bb41:
  %v280 = bitcast i8* %v22 to [8 x i8]*
  %v203 = getelementptr inbounds [8 x i8], [8 x i8]* %v280, i32 0, i64 %v176
  %v204 = load i8, i8* %v203, align 1
  %v205 = uitofp i8 %v204 to float
  %v206 = add i64 %v176, 1
  br label %bb42
bb42:
  %v207 = phi float [ 0.0, %bb41 ], [ %v228, %bb45 ]
  %v208 = phi i64 [ 0, %bb41 ], [ %v229, %bb45 ]
  %v209 = icmp ult i64 %v208, 32
  %v210 = xor i1 %v209, 1
  br i1 %v210, label %bb46, label %bb43
bb43:
  %v211 = add i64 %v171, %v208
  %v212 = icmp ult i64 %v211, %v39
  br i1 %v212, label %bb44, label %bb80
bb44:
  %v213 = extractvalue { i8*, i64 } %v14, 0
  %v214 = getelementptr inbounds i8, i8* %v213, i64 %v211
  %v215 = load i8, i8* %v214, align 1
  %v216 = trunc i32 4 to i8
  %v217 = and i8 %v216, 7
  %v218 = lshr i8 %v215, %v217
  %v219 = uitofp i8 %v218 to float
  %v220 = add i64 %v136, %v201
  %v221 = add i64 %v220, %v208
  %v222 = extractvalue { i8*, i64 } %v15, 1
  %v223 = icmp ult i64 %v221, %v222
  br i1 %v223, label %bb45, label %bb81
bb45:
  %v224 = extractvalue { i8*, i64 } %v15, 0
  %v281 = bitcast i8* %v224 to float*
  %v282 = getelementptr inbounds float, float* %v281, i64 %v221
  %v225 = bitcast float* %v282 to i8*
  %v283 = bitcast i8* %v225 to float*
  %v226 = load float, float* %v283, align 4
  %v227 = fmul contract float %v219, %v226
  %v228 = fadd contract float %v207, %v227
  %v229 = add i64 %v208, 1
  br label %bb42
bb46:
  %v230 = fmul contract float %v69, %v205
  %v231 = fmul contract float %v230, %v207
  %v232 = fadd contract float %v200, %v231
  %v233 = add i64 %v201, 32
  %v234 = add i64 %v167, 1
  br label %bb33
bb47:
  %v235 = add i32 %v33, 1
  br label %bb4
bb48:
  %v236 = extractvalue { i8*, i64 } %v18, 1
  %v237 = icmp ult i64 %v25, %v236
  %v238 = xor i1 %v237, 1
  br i1 %v238, label %bb54, label %bb53
bb49:
  %v239 = extractvalue { i8, i8* } %v245, 1
  %v284 = bitcast i8* %v239 to float*
  store float %v32, float* %v284, align 4
  br label %bb51
bb50:
  br label %bb51
bb51:
  br label %bb52
bb52:
  ret void
bb53:
  %v240 = extractvalue { i8*, i64 } %v18, 0
  %v285 = bitcast i8* %v240 to float*
  %v286 = getelementptr inbounds float, float* %v285, i64 %v25
  %v241 = bitcast float* %v286 to i8*
  %v242 = insertvalue { i8, i8* } undef, i8 1, 0
  %v243 = insertvalue { i8, i8* } %v242, i8* %v241, 1
  br label %bb55
bb54:
  %v244 = insertvalue { i8, i8* } undef, i8 0, 0
  br label %bb55
bb55:
  %v245 = phi { i8, i8* } [ %v243, %bb53 ], [ %v244, %bb54 ]
  %v246 = extractvalue { i8, i8* } %v245, 0
  %v247 = zext i8 %v246 to i64
  %v248 = icmp eq i64 %v247, 1
  br i1 %v248, label %bb49, label %bb56
bb56:
  %v249 = icmp eq i64 %v247, 0
  br i1 %v249, label %bb50, label %bb57
bb57:
  unreachable
bb58:
  unreachable
bb59:
  unreachable
bb60:
  unreachable
bb61:
  unreachable
bb62:
  unreachable
bb63:
  unreachable
bb64:
  unreachable
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
bb68:
  unreachable
bb69:
  unreachable
bb70:
  unreachable
bb71:
  unreachable
bb72:
  unreachable
bb73:
  unreachable
bb74:
  unreachable
bb75:
  unreachable
bb76:
  unreachable
bb77:
  unreachable
bb78:
  unreachable
bb79:
  unreachable
bb80:
  unreachable
bb81:
  unreachable
}

define void @attention_paged_heads(i8* %v0, i64 %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, i8* %v16, i64 %v17) #0 {
entry:
  %v18 = insertvalue { i8*, i64 } undef, i8* %v0, 0
  %v19 = insertvalue { i8*, i64 } %v18, i64 %v1, 1
  %v20 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v21 = insertvalue { i8*, i64 } %v20, i64 %v3, 1
  %v22 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v23 = insertvalue { i8*, i64 } %v22, i64 %v5, 1
  %v24 = insertvalue { i8*, i64 } undef, i8* %v16, 0
  %v25 = insertvalue { i8*, i64 } %v24, i64 %v17, 1
  br label %bb0
bb0:
  %v26 = phi { i8*, i64 } [ %v19, %entry ]
  %v27 = phi { i8*, i64 } [ %v21, %entry ]
  %v28 = phi { i8*, i64 } [ %v23, %entry ]
  %v29 = phi i32 [ %v6, %entry ]
  %v30 = phi i32 [ %v7, %entry ]
  %v31 = phi i32 [ %v8, %entry ]
  %v32 = phi i32 [ %v9, %entry ]
  %v33 = phi float [ %v10, %entry ]
  %v34 = phi i32 [ %v11, %entry ]
  %v35 = phi i32 [ %v12, %entry ]
  %v36 = phi i32 [ %v13, %entry ]
  %v37 = phi i32 [ %v14, %entry ]
  %v38 = phi i32 [ %v15, %entry ]
  %v39 = phi { i8*, i64 } [ %v25, %entry ]
  %v197 = alloca {  }, align 1
  %v40 = bitcast {  }* %v197 to i8*
  %v41 = getelementptr i8, i8* %v40, i64 0
  %v42 = call i64 @cuda_device____internal__index_1d(i8* %v41) #0
  br label %bb1
bb1:
  %v43 = trunc i64 %v42 to i32
  %v44 = icmp uge i32 %v43, %v29
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb3, label %bb2
bb2:
  br label %bb41
bb3:
  %v46 = zext i32 %v31 to i64
  %v47 = zext i32 %v32 to i64
  %v48 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCsh6jSs53Rst6_12cuda_kernels(i32 %v30, i32 1) #0
  br label %bb4
bb4:
  %v49 = icmp eq i32 %v48, 0
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb5, label %bb44
bb5:
  %v51 = udiv i32 %v29, %v48
  %v52 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCsh6jSs53Rst6_12cuda_kernels(i32 %v51, i32 1) #0
  br label %bb6
bb6:
  %v53 = icmp eq i32 %v52, 0
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb7, label %bb45
bb7:
  %v55 = udiv i32 %v43, %v52
  %v56 = zext i32 %v55 to i64
  %v57 = zext i32 %v38 to i64
  %v58 = zext i32 %v36 to i64
  %v59 = zext i32 %v37 to i64
  %v60 = mul i64 %v57, 2
  %v61 = mul i64 %v58, %v60
  %v62 = zext i32 %v43 to i64
  %v63 = mul i64 %v62, %v46
  br label %bb8
bb8:
  %v64 = phi i64 [ 0, %bb7 ], [ %v70, %bb9 ]
  %v65 = icmp ult i64 %v64, %v46
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb10, label %bb9
bb9:
  %v67 = add i64 %v63, %v64
  %v68 = extractvalue { i8*, i64 } %v39, 0
  %v198 = bitcast i8* %v68 to float*
  %v199 = getelementptr inbounds float, float* %v198, i64 %v67
  %v69 = bitcast float* %v199 to i8*
  %v200 = bitcast i8* %v69 to float*
  store float 0.0, float* %v200, align 4
  %v70 = add i64 %v64, 1
  br label %bb8
bb10:
  br label %bb11
bb11:
  %v71 = phi float [ 0.0, %bb10 ], [ %v139, %bb33 ]
  %v72 = phi float [ 0.0, %bb10 ], [ %v189, %bb33 ]
  %v73 = phi i1 [ 0, %bb10 ], [ 1, %bb33 ]
  %v74 = phi i64 [ 0, %bb10 ], [ %v175, %bb33 ]
  %v75 = icmp ult i64 %v74, %v47
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb34, label %bb12
bb12:
  %v77 = icmp eq i64 %v58, 0
  %v78 = xor i1 %v77, 1
  br i1 %v78, label %bb13, label %bb46
bb13:
  %v79 = udiv i64 %v74, %v58
  %v80 = urem i64 %v74, %v58
  %v81 = zext i32 %v34 to i64
  %v82 = zext i32 %v35 to i64
  %v83 = mul i64 %v81, %v82
  %v84 = add i64 %v83, %v79
  %v85 = extractvalue { i8*, i64 } %v28, 1
  %v86 = icmp ult i64 %v84, %v85
  br i1 %v86, label %bb14, label %bb47
bb14:
  %v87 = extractvalue { i8*, i64 } %v28, 0
  %v201 = bitcast i8* %v87 to i32*
  %v202 = getelementptr inbounds i32, i32* %v201, i64 %v84
  %v88 = bitcast i32* %v202 to i8*
  %v203 = bitcast i8* %v88 to i32*
  %v89 = load i32, i32* %v203, align 4
  %v90 = zext i32 %v89 to i64
  %v91 = mul i64 %v90, %v59
  %v92 = mul i64 %v80, %v60
  %v93 = add i64 %v91, %v92
  %v94 = mul i64 %v56, %v46
  %v95 = mul i64 %v94, 2
  %v96 = add i64 %v93, %v95
  br label %bb15
bb15:
  %v97 = phi float [ 0.0, %bb14 ], [ %v129, %bb20 ]
  %v98 = phi i64 [ 0, %bb14 ], [ %v130, %bb20 ]
  %v99 = icmp ult i64 %v98, %v46
  %v100 = xor i1 %v99, 1
  br i1 %v100, label %bb21, label %bb16
bb16:
  %v101 = mul i64 %v98, 2
  %v102 = add i64 %v96, %v101
  %v103 = extractvalue { i8*, i64 } %v27, 1
  %v104 = icmp ult i64 %v102, %v103
  br i1 %v104, label %bb17, label %bb48
bb17:
  %v105 = extractvalue { i8*, i64 } %v27, 0
  %v106 = getelementptr inbounds i8, i8* %v105, i64 %v102
  %v107 = load i8, i8* %v106, align 1
  %v108 = zext i8 %v107 to i16
  %v109 = mul i64 %v98, 2
  %v110 = add i64 %v96, %v109
  %v111 = add i64 %v110, 1
  %v112 = icmp ult i64 %v111, %v103
  br i1 %v112, label %bb18, label %bb49
bb18:
  %v113 = extractvalue { i8*, i64 } %v27, 0
  %v114 = getelementptr inbounds i8, i8* %v113, i64 %v111
  %v115 = load i8, i8* %v114, align 1
  %v116 = zext i8 %v115 to i16
  %v117 = trunc i32 8 to i16
  %v118 = and i16 %v117, 15
  %v119 = shl i16 %v116, %v118
  %v120 = or i16 %v108, %v119
  %v121 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v120) #0
  br label %bb19
bb19:
  %v122 = add i64 %v63, %v98
  %v123 = extractvalue { i8*, i64 } %v26, 1
  %v124 = icmp ult i64 %v122, %v123
  br i1 %v124, label %bb20, label %bb50
bb20:
  %v125 = extractvalue { i8*, i64 } %v26, 0
  %v204 = bitcast i8* %v125 to float*
  %v205 = getelementptr inbounds float, float* %v204, i64 %v122
  %v126 = bitcast float* %v205 to i8*
  %v206 = bitcast i8* %v126 to float*
  %v127 = load float, float* %v206, align 4
  %v128 = fmul contract float %v127, %v121
  %v129 = fadd contract float %v97, %v128
  %v130 = add i64 %v98, 1
  br label %bb15
bb21:
  %v131 = fmul contract float %v97, %v33
  %v132 = xor i1 %v73, 1
  br i1 %v132, label %bb23, label %bb22
bb22:
  %v133 = fcmp ogt float %v131, %v71
  %v134 = xor i1 %v133, 1
  br i1 %v134, label %bb25, label %bb24
bb23:
  br label %bb27
bb24:
  %v135 = fsub contract float %v71, %v131
  %v136 = call float @__nv_expf(float %v135) #0
  br label %bb42
bb25:
  br label %bb26
bb26:
  %v137 = phi float [ %v71, %bb25 ], [ %v131, %bb42 ]
  %v138 = phi float [ 1.0, %bb25 ], [ %v136, %bb42 ]
  br label %bb27
bb27:
  %v139 = phi float [ %v131, %bb23 ], [ %v137, %bb26 ]
  %v140 = phi float [ 0.0, %bb23 ], [ %v138, %bb26 ]
  %v141 = fsub contract float %v131, %v139
  %v142 = call float @__nv_expf(float %v141) #0
  br label %bb43
bb28:
  %v143 = phi i64 [ %v174, %bb32 ], [ 0, %bb43 ]
  %v144 = icmp ult i64 %v143, %v46
  %v145 = xor i1 %v144, 1
  br i1 %v145, label %bb33, label %bb29
bb29:
  %v146 = mul i64 %v143, 2
  %v147 = add i64 %v192, %v146
  %v148 = extractvalue { i8*, i64 } %v27, 1
  %v149 = icmp ult i64 %v147, %v148
  br i1 %v149, label %bb30, label %bb51
bb30:
  %v150 = extractvalue { i8*, i64 } %v27, 0
  %v151 = getelementptr inbounds i8, i8* %v150, i64 %v147
  %v152 = load i8, i8* %v151, align 1
  %v153 = zext i8 %v152 to i16
  %v154 = mul i64 %v143, 2
  %v155 = add i64 %v192, %v154
  %v156 = add i64 %v155, 1
  %v157 = icmp ult i64 %v156, %v148
  br i1 %v157, label %bb31, label %bb52
bb31:
  %v158 = extractvalue { i8*, i64 } %v27, 0
  %v159 = getelementptr inbounds i8, i8* %v158, i64 %v156
  %v160 = load i8, i8* %v159, align 1
  %v161 = zext i8 %v160 to i16
  %v162 = trunc i32 8 to i16
  %v163 = and i16 %v162, 15
  %v164 = shl i16 %v161, %v163
  %v165 = or i16 %v153, %v164
  %v166 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v165) #0
  br label %bb32
bb32:
  %v167 = add i64 %v63, %v143
  %v168 = extractvalue { i8*, i64 } %v39, 0
  %v207 = bitcast i8* %v168 to float*
  %v208 = getelementptr inbounds float, float* %v207, i64 %v167
  %v169 = bitcast float* %v208 to i8*
  %v209 = bitcast i8* %v169 to float*
  %v170 = load float, float* %v209, align 4
  %v171 = fmul contract float %v170, %v140
  %v172 = fmul contract float %v142, %v166
  %v173 = fadd contract float %v171, %v172
  %v210 = bitcast i8* %v169 to float*
  store float %v173, float* %v210, align 4
  %v174 = add i64 %v143, 1
  br label %bb28
bb33:
  %v175 = add i64 %v74, 1
  br label %bb11
bb34:
  %v176 = fcmp ogt float %v72, 0.0
  %v177 = xor i1 %v176, 1
  br i1 %v177, label %bb36, label %bb35
bb35:
  %v178 = fdiv contract float 1.0, %v72
  br label %bb37
bb36:
  br label %bb40
bb37:
  %v179 = phi i64 [ 0, %bb35 ], [ %v187, %bb38 ]
  %v180 = icmp ult i64 %v179, %v46
  %v181 = xor i1 %v180, 1
  br i1 %v181, label %bb39, label %bb38
bb38:
  %v182 = add i64 %v63, %v179
  %v183 = extractvalue { i8*, i64 } %v39, 0
  %v211 = bitcast i8* %v183 to float*
  %v212 = getelementptr inbounds float, float* %v211, i64 %v182
  %v184 = bitcast float* %v212 to i8*
  %v213 = bitcast i8* %v184 to float*
  %v185 = load float, float* %v213, align 4
  %v186 = fmul contract float %v185, %v178
  %v214 = bitcast i8* %v184 to float*
  store float %v186, float* %v214, align 4
  %v187 = add i64 %v179, 1
  br label %bb37
bb39:
  br label %bb40
bb40:
  br label %bb41
bb41:
  ret void
bb42:
  br label %bb26
bb43:
  %v188 = fmul contract float %v72, %v140
  %v189 = fadd contract float %v188, %v142
  %v190 = add i64 %v91, %v61
  %v191 = add i64 %v190, %v92
  %v192 = add i64 %v191, %v95
  br label %bb28
bb44:
  unreachable
bb45:
  unreachable
bb46:
  unreachable
bb47:
  unreachable
bb48:
  unreachable
bb49:
  unreachable
bb50:
  unreachable
bb51:
  unreachable
bb52:
  unreachable
}

define void @scale_f32(float %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi float [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v37 = alloca {  }, align 1
  %v12 = bitcast {  }* %v37 to i8*
  %v13 = getelementptr i8, i8* %v12, i64 0
  %v14 = call i64 @cuda_device____internal__index_1d(i8* %v13) #0
  br label %bb1
bb1:
  %v15 = extractvalue { i8*, i64 } %v11, 1
  %v16 = icmp ult i64 %v14, %v15
  %v17 = xor i1 %v16, 1
  br i1 %v17, label %bb7, label %bb6
bb2:
  %v18 = extractvalue { i8, i8* } %v30, 1
  %v19 = extractvalue { i8*, i64 } %v10, 1
  %v20 = icmp ult i64 %v14, %v19
  br i1 %v20, label %bb3, label %bb11
bb3:
  %v21 = extractvalue { i8*, i64 } %v10, 0
  %v38 = bitcast i8* %v21 to float*
  %v39 = getelementptr inbounds float, float* %v38, i64 %v14
  %v22 = bitcast float* %v39 to i8*
  %v40 = bitcast i8* %v22 to float*
  %v23 = load float, float* %v40, align 4
  %v24 = fmul contract float %v23, %v9
  %v41 = bitcast i8* %v18 to float*
  store float %v24, float* %v41, align 4
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v25 = extractvalue { i8*, i64 } %v11, 0
  %v42 = bitcast i8* %v25 to float*
  %v43 = getelementptr inbounds float, float* %v42, i64 %v14
  %v26 = bitcast float* %v43 to i8*
  %v27 = insertvalue { i8, i8* } undef, i8 1, 0
  %v28 = insertvalue { i8, i8* } %v27, i8* %v26, 1
  br label %bb8
bb7:
  %v29 = insertvalue { i8, i8* } undef, i8 0, 0
  br label %bb8
bb8:
  %v30 = phi { i8, i8* } [ %v28, %bb6 ], [ %v29, %bb7 ]
  %v31 = extractvalue { i8, i8* } %v30, 0
  %v32 = zext i8 %v31 to i64
  %v33 = icmp eq i64 %v32, 1
  br i1 %v33, label %bb2, label %bb9
bb9:
  %v34 = icmp eq i64 %v32, 0
  br i1 %v34, label %bb4, label %bb10
bb10:
  unreachable
bb11:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define i64 @cuda_device____internal__index_1d(i8* %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi i8* [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v3 = zext i32 %v2 to i64
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v5 = zext i32 %v4 to i64
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v7 = zext i32 %v6 to i64
  %v8 = mul i64 %v5, %v7
  %v9 = add i64 %v8, %v3
  ret i64 %v9
}

define i16 @cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits(float %v0) #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi float [ %v0, %entry ]
  %v2 = bitcast float %v1 to i32
  %v3 = and i32 16, 31
  %v4 = lshr i32 %v2, %v3
  %v5 = and i32 %v4, 32768
  %v6 = and i32 23, 31
  %v7 = lshr i32 %v2, %v6
  %v8 = and i32 %v7, 255
  %v9 = bitcast i32 %v8 to i32
  %v10 = and i32 %v2, 8388607
  %v11 = icmp eq i32 %v9, 255
  br i1 %v11, label %bb1, label %bb5
bb1:
  %v12 = or i32 %v5, 31744
  %v13 = icmp eq i32 %v10, 0
  br i1 %v13, label %bb3, label %bb2
bb2:
  br label %bb4
bb3:
  br label %bb4
bb4:
  %v14 = phi i32 [ 512, %bb2 ], [ 0, %bb3 ]
  %v15 = or i32 %v12, %v14
  %v16 = trunc i32 %v15 to i16
  br label %bb19
bb5:
  %v17 = sub i32 %v9, 127
  %v18 = add i32 %v17, 15
  %v19 = icmp sge i32 %v18, 31
  %v20 = xor i1 %v19, 1
  br i1 %v20, label %bb7, label %bb6
bb6:
  %v21 = or i32 %v5, 31744
  %v22 = trunc i32 %v21 to i16
  br label %bb19
bb7:
  %v23 = icmp sle i32 %v18, 0
  %v24 = xor i1 %v23, 1
  br i1 %v24, label %bb9, label %bb8
bb8:
  %v25 = icmp slt i32 %v18, 4294967286
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb11, label %bb10
bb9:
  %v27 = bitcast i32 %v18 to i32
  %v28 = and i32 10, 31
  %v29 = shl i32 %v27, %v28
  %v30 = and i32 13, 31
  %v31 = lshr i32 %v10, %v30
  %v32 = or i32 %v29, %v31
  %v33 = and i32 %v10, 4096
  %v34 = icmp eq i32 %v33, 0
  br i1 %v34, label %bb16, label %bb15
bb10:
  %v35 = trunc i32 %v5 to i16
  br label %bb18
bb11:
  %v36 = or i32 %v10, 8388608
  %v37 = sub i32 1, %v18
  %v38 = and i32 %v37, 31
  %v39 = lshr i32 %v36, %v38
  %v40 = and i32 13, 31
  %v41 = lshr i32 %v39, %v40
  %v42 = and i32 %v39, 4096
  %v43 = icmp eq i32 %v42, 0
  br i1 %v43, label %bb13, label %bb12
bb12:
  br label %bb14
bb13:
  br label %bb14
bb14:
  %v44 = phi i32 [ 1, %bb12 ], [ 0, %bb13 ]
  %v45 = add i32 %v41, %v44
  %v46 = or i32 %v5, %v45
  %v47 = trunc i32 %v46 to i16
  br label %bb18
bb15:
  br label %bb17
bb16:
  br label %bb17
bb17:
  %v48 = phi i32 [ 1, %bb15 ], [ 0, %bb16 ]
  %v49 = add i32 %v32, %v48
  %v50 = or i32 %v5, %v49
  %v51 = trunc i32 %v50 to i16
  br label %bb19
bb18:
  %v52 = phi i16 [ %v35, %bb10 ], [ %v47, %bb14 ]
  br label %bb19
bb19:
  %v53 = phi i16 [ %v16, %bb4 ], [ %v22, %bb6 ], [ %v51, %bb17 ], [ %v52, %bb18 ]
  ret i16 %v53
}

define i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCsh6jSs53Rst6_12cuda_kernels(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v13 = alloca i32, align 4
  %v4 = bitcast i32* %v13 to i8*
  %v14 = alloca i32, align 4
  %v5 = bitcast i32* %v14 to i8*
  %v15 = bitcast i8* %v4 to i32*
  store i32 %v2, i32* %v15, align 4
  %v16 = bitcast i8* %v5 to i32*
  store i32 %v3, i32* %v16, align 4
  %v6 = getelementptr i8, i8* %v5, i64 0
  %v7 = getelementptr i8, i8* %v4, i64 0
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(i8* %v6, i8* %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v17 = bitcast i8* %v4 to i32*
  %v10 = load i32, i32* %v17, align 4
  br label %bb4
bb3:
  %v18 = bitcast i8* %v5 to i32*
  %v11 = load i32, i32* %v18, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
bb5:
  unreachable
bb6:
  unreachable
bb7:
  unreachable
bb8:
  unreachable
}

define float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v0) #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi i16 [ %v0, %entry ]
  %v2 = trunc i32 15 to i16
  %v3 = and i16 %v2, 15
  %v4 = lshr i16 %v1, %v3
  %v5 = and i16 %v4, 1
  %v6 = zext i16 %v5 to i32
  %v7 = trunc i32 10 to i16
  %v8 = and i16 %v7, 15
  %v9 = lshr i16 %v1, %v8
  %v10 = and i16 %v9, 31
  %v11 = zext i16 %v10 to i32
  %v12 = and i16 %v1, 1023
  %v13 = zext i16 %v12 to i32
  %v14 = icmp eq i32 %v11, 0
  br i1 %v14, label %bb1, label %bb8
bb1:
  %v15 = icmp eq i32 %v13, 0
  br i1 %v15, label %bb2, label %bb3
bb2:
  %v16 = and i32 31, 31
  %v17 = shl i32 %v6, %v16
  br label %bb7
bb3:
  br label %bb4
bb4:
  %v18 = phi i32 [ %v13, %bb3 ], [ %v23, %bb5 ]
  %v19 = phi i32 [ 113, %bb3 ], [ %v24, %bb5 ]
  %v20 = and i32 %v18, 1024
  %v21 = icmp eq i32 %v20, 0
  br i1 %v21, label %bb5, label %bb6
bb5:
  %v22 = and i32 1, 31
  %v23 = shl i32 %v18, %v22
  %v24 = sub i32 %v19, 1
  br label %bb4
bb6:
  %v25 = and i32 %v18, 1023
  %v26 = and i32 31, 31
  %v27 = shl i32 %v6, %v26
  %v28 = bitcast i32 %v19 to i32
  %v29 = and i32 23, 31
  %v30 = shl i32 %v28, %v29
  %v31 = or i32 %v27, %v30
  %v32 = and i32 13, 31
  %v33 = shl i32 %v25, %v32
  %v34 = or i32 %v31, %v33
  br label %bb7
bb7:
  %v35 = phi i32 [ %v17, %bb2 ], [ %v34, %bb6 ]
  br label %bb12
bb8:
  %v36 = icmp eq i32 %v11, 31
  br i1 %v36, label %bb9, label %bb10
bb9:
  %v37 = and i32 31, 31
  %v38 = shl i32 %v6, %v37
  %v39 = or i32 %v38, 2139095040
  %v40 = and i32 13, 31
  %v41 = shl i32 %v13, %v40
  %v42 = or i32 %v39, %v41
  br label %bb11
bb10:
  %v43 = and i32 31, 31
  %v44 = shl i32 %v6, %v43
  %v45 = add i32 %v11, 127
  %v46 = sub i32 %v45, 15
  %v47 = and i32 23, 31
  %v48 = shl i32 %v46, %v47
  %v49 = or i32 %v44, %v48
  %v50 = and i32 13, 31
  %v51 = shl i32 %v13, %v50
  %v52 = or i32 %v49, %v51
  br label %bb11
bb11:
  %v53 = phi i32 [ %v42, %bb9 ], [ %v52, %bb10 ]
  br label %bb12
bb12:
  %v54 = phi i32 [ %v35, %bb7 ], [ %v53, %bb11 ]
  %v55 = bitcast i32 %v54 to float
  ret float %v55
}

define { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v0, i8 %v1, i8 %v2, i8 %v3, i8 %v4, i8 %v5, i8 %v6, i8 %v7, i8 %v8, i8 %v9, i8 %v10, i8 %v11) #0 {
entry:
  br label %bb0
bb0:
  %v12 = phi i8 [ %v0, %entry ]
  %v13 = phi i8 [ %v1, %entry ]
  %v14 = phi i8 [ %v2, %entry ]
  %v15 = phi i8 [ %v3, %entry ]
  %v16 = phi i8 [ %v4, %entry ]
  %v17 = phi i8 [ %v5, %entry ]
  %v18 = phi i8 [ %v6, %entry ]
  %v19 = phi i8 [ %v7, %entry ]
  %v20 = phi i8 [ %v8, %entry ]
  %v21 = phi i8 [ %v9, %entry ]
  %v22 = phi i8 [ %v10, %entry ]
  %v23 = phi i8 [ %v11, %entry ]
  %v164 = alloca [4 x i32], align 4
  %v24 = bitcast [4 x i32]* %v164 to i8*
  %v165 = alloca [4 x i8], align 1
  %v25 = bitcast [4 x i8]* %v165 to i8*
  %v166 = alloca [4 x i8], align 1
  %v26 = bitcast [4 x i8]* %v166 to i8*
  %v167 = alloca [4 x i8], align 1
  %v27 = bitcast [4 x i8]* %v167 to i8*
  %v168 = alloca [4 x i8], align 1
  %v28 = bitcast [4 x i8]* %v168 to i8*
  %v169 = alloca [4 x i8], align 1
  %v29 = bitcast [4 x i8]* %v169 to i8*
  %v170 = alloca [4 x i8], align 1
  %v30 = bitcast [4 x i8]* %v170 to i8*
  %v171 = alloca [4 x i8], align 1
  %v31 = bitcast [4 x i8]* %v171 to i8*
  %v172 = alloca [8 x i8], align 1
  %v32 = bitcast [8 x i8]* %v172 to i8*
  %v173 = alloca [8 x i8], align 1
  %v33 = bitcast [8 x i8]* %v173 to i8*
  %v174 = bitcast i8* %v24 to [4 x i32]*
  %v175 = getelementptr inbounds [4 x i32], [4 x i32]* %v174, i32 0, i64 0
  %v34 = bitcast i32* %v175 to i8*
  %v176 = bitcast i8* %v34 to i32*
  store i32 0, i32* %v176, align 4
  %v177 = bitcast i8* %v24 to [4 x i32]*
  %v178 = getelementptr inbounds [4 x i32], [4 x i32]* %v177, i32 0, i64 1
  %v35 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v35 to i32*
  store i32 0, i32* %v179, align 4
  %v180 = bitcast i8* %v24 to [4 x i32]*
  %v181 = getelementptr inbounds [4 x i32], [4 x i32]* %v180, i32 0, i64 2
  %v36 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v36 to i32*
  store i32 0, i32* %v182, align 4
  %v183 = bitcast i8* %v24 to [4 x i32]*
  %v184 = getelementptr inbounds [4 x i32], [4 x i32]* %v183, i32 0, i64 3
  %v37 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v37 to i32*
  store i32 0, i32* %v185, align 4
  %v186 = bitcast i8* %v25 to [4 x i8]*
  %v38 = getelementptr inbounds [4 x i8], [4 x i8]* %v186, i32 0, i64 0
  store i8 %v12, i8* %v38, align 1
  %v187 = bitcast i8* %v25 to [4 x i8]*
  %v39 = getelementptr inbounds [4 x i8], [4 x i8]* %v187, i32 0, i64 1
  store i8 %v13, i8* %v39, align 1
  %v188 = bitcast i8* %v25 to [4 x i8]*
  %v40 = getelementptr inbounds [4 x i8], [4 x i8]* %v188, i32 0, i64 2
  store i8 %v14, i8* %v40, align 1
  %v189 = bitcast i8* %v25 to [4 x i8]*
  %v41 = getelementptr inbounds [4 x i8], [4 x i8]* %v189, i32 0, i64 3
  store i8 %v15, i8* %v41, align 1
  %v190 = bitcast i8* %v25 to [4 x i8]*
  %v42 = load [4 x i8], [4 x i8]* %v190, align 1
  %v191 = alloca [4 x i8], align 4
  %v43 = bitcast [4 x i8]* %v191 to i8*
  %v192 = bitcast i8* %v43 to [4 x i8]*
  store [4 x i8] %v42, [4 x i8]* %v192, align 4
  %v193 = bitcast i8* %v43 to i32*
  %v44 = load i32, i32* %v193, align 4
  %v194 = bitcast i8* %v24 to [4 x i32]*
  %v195 = getelementptr inbounds [4 x i32], [4 x i32]* %v194, i32 0, i64 0
  %v45 = bitcast i32* %v195 to i8*
  %v196 = bitcast i8* %v45 to i32*
  store i32 %v44, i32* %v196, align 4
  %v197 = bitcast i8* %v26 to [4 x i8]*
  %v46 = getelementptr inbounds [4 x i8], [4 x i8]* %v197, i32 0, i64 0
  store i8 %v16, i8* %v46, align 1
  %v198 = bitcast i8* %v26 to [4 x i8]*
  %v47 = getelementptr inbounds [4 x i8], [4 x i8]* %v198, i32 0, i64 1
  store i8 %v17, i8* %v47, align 1
  %v199 = bitcast i8* %v26 to [4 x i8]*
  %v48 = getelementptr inbounds [4 x i8], [4 x i8]* %v199, i32 0, i64 2
  store i8 %v18, i8* %v48, align 1
  %v200 = bitcast i8* %v26 to [4 x i8]*
  %v49 = getelementptr inbounds [4 x i8], [4 x i8]* %v200, i32 0, i64 3
  store i8 %v19, i8* %v49, align 1
  %v201 = bitcast i8* %v26 to [4 x i8]*
  %v50 = load [4 x i8], [4 x i8]* %v201, align 1
  %v202 = alloca [4 x i8], align 4
  %v51 = bitcast [4 x i8]* %v202 to i8*
  %v203 = bitcast i8* %v51 to [4 x i8]*
  store [4 x i8] %v50, [4 x i8]* %v203, align 4
  %v204 = bitcast i8* %v51 to i32*
  %v52 = load i32, i32* %v204, align 4
  %v205 = bitcast i8* %v24 to [4 x i32]*
  %v206 = getelementptr inbounds [4 x i32], [4 x i32]* %v205, i32 0, i64 1
  %v53 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v53 to i32*
  store i32 %v52, i32* %v207, align 4
  %v208 = bitcast i8* %v27 to [4 x i8]*
  %v54 = getelementptr inbounds [4 x i8], [4 x i8]* %v208, i32 0, i64 0
  store i8 %v20, i8* %v54, align 1
  %v209 = bitcast i8* %v27 to [4 x i8]*
  %v55 = getelementptr inbounds [4 x i8], [4 x i8]* %v209, i32 0, i64 1
  store i8 %v21, i8* %v55, align 1
  %v210 = bitcast i8* %v27 to [4 x i8]*
  %v56 = getelementptr inbounds [4 x i8], [4 x i8]* %v210, i32 0, i64 2
  store i8 %v22, i8* %v56, align 1
  %v211 = bitcast i8* %v27 to [4 x i8]*
  %v57 = getelementptr inbounds [4 x i8], [4 x i8]* %v211, i32 0, i64 3
  store i8 %v23, i8* %v57, align 1
  %v212 = bitcast i8* %v27 to [4 x i8]*
  %v58 = load [4 x i8], [4 x i8]* %v212, align 1
  %v213 = alloca [4 x i8], align 4
  %v59 = bitcast [4 x i8]* %v213 to i8*
  %v214 = bitcast i8* %v59 to [4 x i8]*
  store [4 x i8] %v58, [4 x i8]* %v214, align 4
  %v215 = bitcast i8* %v59 to i32*
  %v60 = load i32, i32* %v215, align 4
  %v216 = bitcast i8* %v24 to [4 x i32]*
  %v217 = getelementptr inbounds [4 x i32], [4 x i32]* %v216, i32 0, i64 2
  %v61 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v61 to i32*
  store i32 %v60, i32* %v218, align 4
  %v219 = bitcast i8* %v24 to [4 x i32]*
  %v220 = getelementptr inbounds [4 x i32], [4 x i32]* %v219, i32 0, i64 2
  %v62 = bitcast i32* %v220 to i8*
  %v221 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v221, align 4
  %v64 = and i32 4, 31
  %v65 = lshr i32 %v63, %v64
  %v66 = and i32 %v65, 252645135
  %v222 = bitcast i8* %v24 to [4 x i32]*
  %v223 = getelementptr inbounds [4 x i32], [4 x i32]* %v222, i32 0, i64 1
  %v67 = bitcast i32* %v223 to i8*
  %v224 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v224, align 4
  %v69 = and i32 6, 31
  %v70 = lshr i32 %v68, %v69
  %v71 = and i32 %v70, 50529027
  %v72 = and i32 4, 31
  %v73 = shl i32 %v71, %v72
  %v74 = or i32 %v66, %v73
  %v225 = bitcast i8* %v24 to [4 x i32]*
  %v226 = getelementptr inbounds [4 x i32], [4 x i32]* %v225, i32 0, i64 3
  %v75 = bitcast i32* %v226 to i8*
  %v227 = bitcast i8* %v75 to i32*
  store i32 %v74, i32* %v227, align 4
  %v228 = bitcast i8* %v24 to [4 x i32]*
  %v229 = getelementptr inbounds [4 x i32], [4 x i32]* %v228, i32 0, i64 1
  %v76 = bitcast i32* %v229 to i8*
  %v230 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v230, align 4
  %v78 = and i32 %v77, 1061109567
  %v231 = bitcast i8* %v24 to [4 x i32]*
  %v232 = getelementptr inbounds [4 x i32], [4 x i32]* %v231, i32 0, i64 2
  %v79 = bitcast i32* %v232 to i8*
  %v233 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v233, align 4
  %v81 = and i32 %v80, 252645135
  %v234 = bitcast i8* %v24 to [4 x i32]*
  %v235 = getelementptr inbounds [4 x i32], [4 x i32]* %v234, i32 0, i64 0
  %v82 = bitcast i32* %v235 to i8*
  %v236 = bitcast i8* %v82 to i32*
  %v83 = load i32, i32* %v236, align 4
  %v84 = and i32 6, 31
  %v85 = lshr i32 %v83, %v84
  %v86 = and i32 %v85, 50529027
  %v87 = and i32 4, 31
  %v88 = shl i32 %v86, %v87
  %v89 = or i32 %v81, %v88
  %v237 = bitcast i8* %v24 to [4 x i32]*
  %v238 = getelementptr inbounds [4 x i32], [4 x i32]* %v237, i32 0, i64 1
  %v90 = bitcast i32* %v238 to i8*
  %v239 = bitcast i8* %v90 to i32*
  store i32 %v89, i32* %v239, align 4
  %v240 = bitcast i8* %v24 to [4 x i32]*
  %v241 = getelementptr inbounds [4 x i32], [4 x i32]* %v240, i32 0, i64 2
  %v91 = bitcast i32* %v241 to i8*
  %v242 = bitcast i8* %v91 to i32*
  store i32 %v78, i32* %v242, align 4
  %v243 = bitcast i8* %v24 to [4 x i32]*
  %v244 = getelementptr inbounds [4 x i32], [4 x i32]* %v243, i32 0, i64 0
  %v92 = bitcast i32* %v244 to i8*
  %v245 = bitcast i8* %v92 to i32*
  %v93 = load i32, i32* %v245, align 4
  %v94 = and i32 %v93, 1061109567
  %v246 = bitcast i8* %v24 to [4 x i32]*
  %v247 = getelementptr inbounds [4 x i32], [4 x i32]* %v246, i32 0, i64 0
  %v95 = bitcast i32* %v247 to i8*
  %v248 = bitcast i8* %v95 to i32*
  store i32 %v94, i32* %v248, align 4
  %v249 = bitcast i8* %v24 to [4 x i32]*
  %v250 = getelementptr inbounds [4 x i32], [4 x i32]* %v249, i32 0, i64 0
  %v96 = bitcast i32* %v250 to i8*
  %v251 = bitcast i8* %v96 to i32*
  %v97 = load i32, i32* %v251, align 4
  %v252 = alloca i32, align 4
  %v98 = bitcast i32* %v252 to i8*
  %v253 = bitcast i8* %v98 to i32*
  store i32 %v97, i32* %v253, align 4
  %v254 = bitcast i8* %v98 to [4 x i8]*
  %v99 = load [4 x i8], [4 x i8]* %v254, align 4
  %v255 = bitcast i8* %v28 to [4 x i8]*
  store [4 x i8] %v99, [4 x i8]* %v255, align 1
  %v256 = bitcast i8* %v24 to [4 x i32]*
  %v257 = getelementptr inbounds [4 x i32], [4 x i32]* %v256, i32 0, i64 1
  %v100 = bitcast i32* %v257 to i8*
  %v258 = bitcast i8* %v100 to i32*
  %v101 = load i32, i32* %v258, align 4
  %v259 = alloca i32, align 4
  %v102 = bitcast i32* %v259 to i8*
  %v260 = bitcast i8* %v102 to i32*
  store i32 %v101, i32* %v260, align 4
  %v261 = bitcast i8* %v102 to [4 x i8]*
  %v103 = load [4 x i8], [4 x i8]* %v261, align 4
  %v262 = bitcast i8* %v29 to [4 x i8]*
  store [4 x i8] %v103, [4 x i8]* %v262, align 1
  %v263 = bitcast i8* %v24 to [4 x i32]*
  %v264 = getelementptr inbounds [4 x i32], [4 x i32]* %v263, i32 0, i64 2
  %v104 = bitcast i32* %v264 to i8*
  %v265 = bitcast i8* %v104 to i32*
  %v105 = load i32, i32* %v265, align 4
  %v266 = alloca i32, align 4
  %v106 = bitcast i32* %v266 to i8*
  %v267 = bitcast i8* %v106 to i32*
  store i32 %v105, i32* %v267, align 4
  %v268 = bitcast i8* %v106 to [4 x i8]*
  %v107 = load [4 x i8], [4 x i8]* %v268, align 4
  %v269 = bitcast i8* %v30 to [4 x i8]*
  store [4 x i8] %v107, [4 x i8]* %v269, align 1
  %v270 = bitcast i8* %v24 to [4 x i32]*
  %v271 = getelementptr inbounds [4 x i32], [4 x i32]* %v270, i32 0, i64 3
  %v108 = bitcast i32* %v271 to i8*
  %v272 = bitcast i8* %v108 to i32*
  %v109 = load i32, i32* %v272, align 4
  %v273 = alloca i32, align 4
  %v110 = bitcast i32* %v273 to i8*
  %v274 = bitcast i8* %v110 to i32*
  store i32 %v109, i32* %v274, align 4
  %v275 = bitcast i8* %v110 to [4 x i8]*
  %v111 = load [4 x i8], [4 x i8]* %v275, align 4
  %v276 = bitcast i8* %v31 to [4 x i8]*
  store [4 x i8] %v111, [4 x i8]* %v276, align 1
  %v277 = bitcast i8* %v28 to [4 x i8]*
  %v112 = getelementptr inbounds [4 x i8], [4 x i8]* %v277, i32 0, i64 0
  %v113 = load i8, i8* %v112, align 1
  %v278 = bitcast i8* %v28 to [4 x i8]*
  %v114 = getelementptr inbounds [4 x i8], [4 x i8]* %v278, i32 0, i64 1
  %v115 = load i8, i8* %v114, align 1
  %v279 = bitcast i8* %v28 to [4 x i8]*
  %v116 = getelementptr inbounds [4 x i8], [4 x i8]* %v279, i32 0, i64 2
  %v117 = load i8, i8* %v116, align 1
  %v280 = bitcast i8* %v28 to [4 x i8]*
  %v118 = getelementptr inbounds [4 x i8], [4 x i8]* %v280, i32 0, i64 3
  %v119 = load i8, i8* %v118, align 1
  %v281 = bitcast i8* %v29 to [4 x i8]*
  %v120 = getelementptr inbounds [4 x i8], [4 x i8]* %v281, i32 0, i64 0
  %v121 = load i8, i8* %v120, align 1
  %v282 = bitcast i8* %v29 to [4 x i8]*
  %v122 = getelementptr inbounds [4 x i8], [4 x i8]* %v282, i32 0, i64 1
  %v123 = load i8, i8* %v122, align 1
  %v283 = bitcast i8* %v29 to [4 x i8]*
  %v124 = getelementptr inbounds [4 x i8], [4 x i8]* %v283, i32 0, i64 2
  %v125 = load i8, i8* %v124, align 1
  %v284 = bitcast i8* %v29 to [4 x i8]*
  %v126 = getelementptr inbounds [4 x i8], [4 x i8]* %v284, i32 0, i64 3
  %v127 = load i8, i8* %v126, align 1
  %v285 = bitcast i8* %v32 to [8 x i8]*
  %v128 = getelementptr inbounds [8 x i8], [8 x i8]* %v285, i32 0, i64 0
  store i8 %v113, i8* %v128, align 1
  %v286 = bitcast i8* %v32 to [8 x i8]*
  %v129 = getelementptr inbounds [8 x i8], [8 x i8]* %v286, i32 0, i64 1
  store i8 %v115, i8* %v129, align 1
  %v287 = bitcast i8* %v32 to [8 x i8]*
  %v130 = getelementptr inbounds [8 x i8], [8 x i8]* %v287, i32 0, i64 2
  store i8 %v117, i8* %v130, align 1
  %v288 = bitcast i8* %v32 to [8 x i8]*
  %v131 = getelementptr inbounds [8 x i8], [8 x i8]* %v288, i32 0, i64 3
  store i8 %v119, i8* %v131, align 1
  %v289 = bitcast i8* %v32 to [8 x i8]*
  %v132 = getelementptr inbounds [8 x i8], [8 x i8]* %v289, i32 0, i64 4
  store i8 %v121, i8* %v132, align 1
  %v290 = bitcast i8* %v32 to [8 x i8]*
  %v133 = getelementptr inbounds [8 x i8], [8 x i8]* %v290, i32 0, i64 5
  store i8 %v123, i8* %v133, align 1
  %v291 = bitcast i8* %v32 to [8 x i8]*
  %v134 = getelementptr inbounds [8 x i8], [8 x i8]* %v291, i32 0, i64 6
  store i8 %v125, i8* %v134, align 1
  %v292 = bitcast i8* %v32 to [8 x i8]*
  %v135 = getelementptr inbounds [8 x i8], [8 x i8]* %v292, i32 0, i64 7
  store i8 %v127, i8* %v135, align 1
  %v293 = bitcast i8* %v30 to [4 x i8]*
  %v136 = getelementptr inbounds [4 x i8], [4 x i8]* %v293, i32 0, i64 0
  %v137 = load i8, i8* %v136, align 1
  %v294 = bitcast i8* %v30 to [4 x i8]*
  %v138 = getelementptr inbounds [4 x i8], [4 x i8]* %v294, i32 0, i64 1
  %v139 = load i8, i8* %v138, align 1
  %v295 = bitcast i8* %v30 to [4 x i8]*
  %v140 = getelementptr inbounds [4 x i8], [4 x i8]* %v295, i32 0, i64 2
  %v141 = load i8, i8* %v140, align 1
  %v296 = bitcast i8* %v30 to [4 x i8]*
  %v142 = getelementptr inbounds [4 x i8], [4 x i8]* %v296, i32 0, i64 3
  %v143 = load i8, i8* %v142, align 1
  %v297 = bitcast i8* %v31 to [4 x i8]*
  %v144 = getelementptr inbounds [4 x i8], [4 x i8]* %v297, i32 0, i64 0
  %v145 = load i8, i8* %v144, align 1
  %v298 = bitcast i8* %v31 to [4 x i8]*
  %v146 = getelementptr inbounds [4 x i8], [4 x i8]* %v298, i32 0, i64 1
  %v147 = load i8, i8* %v146, align 1
  %v299 = bitcast i8* %v31 to [4 x i8]*
  %v148 = getelementptr inbounds [4 x i8], [4 x i8]* %v299, i32 0, i64 2
  %v149 = load i8, i8* %v148, align 1
  %v300 = bitcast i8* %v31 to [4 x i8]*
  %v150 = getelementptr inbounds [4 x i8], [4 x i8]* %v300, i32 0, i64 3
  %v151 = load i8, i8* %v150, align 1
  %v301 = bitcast i8* %v33 to [8 x i8]*
  %v152 = getelementptr inbounds [8 x i8], [8 x i8]* %v301, i32 0, i64 0
  store i8 %v137, i8* %v152, align 1
  %v302 = bitcast i8* %v33 to [8 x i8]*
  %v153 = getelementptr inbounds [8 x i8], [8 x i8]* %v302, i32 0, i64 1
  store i8 %v139, i8* %v153, align 1
  %v303 = bitcast i8* %v33 to [8 x i8]*
  %v154 = getelementptr inbounds [8 x i8], [8 x i8]* %v303, i32 0, i64 2
  store i8 %v141, i8* %v154, align 1
  %v304 = bitcast i8* %v33 to [8 x i8]*
  %v155 = getelementptr inbounds [8 x i8], [8 x i8]* %v304, i32 0, i64 3
  store i8 %v143, i8* %v155, align 1
  %v305 = bitcast i8* %v33 to [8 x i8]*
  %v156 = getelementptr inbounds [8 x i8], [8 x i8]* %v305, i32 0, i64 4
  store i8 %v145, i8* %v156, align 1
  %v306 = bitcast i8* %v33 to [8 x i8]*
  %v157 = getelementptr inbounds [8 x i8], [8 x i8]* %v306, i32 0, i64 5
  store i8 %v147, i8* %v157, align 1
  %v307 = bitcast i8* %v33 to [8 x i8]*
  %v158 = getelementptr inbounds [8 x i8], [8 x i8]* %v307, i32 0, i64 6
  store i8 %v149, i8* %v158, align 1
  %v308 = bitcast i8* %v33 to [8 x i8]*
  %v159 = getelementptr inbounds [8 x i8], [8 x i8]* %v308, i32 0, i64 7
  store i8 %v151, i8* %v159, align 1
  %v309 = bitcast i8* %v32 to [8 x i8]*
  %v160 = load [8 x i8], [8 x i8]* %v309, align 1
  %v310 = bitcast i8* %v33 to [8 x i8]*
  %v161 = load [8 x i8], [8 x i8]* %v310, align 1
  %v162 = insertvalue { [8 x i8], [8 x i8] } undef, [8 x i8] %v160, 0
  %v163 = insertvalue { [8 x i8], [8 x i8] } %v162, [8 x i8] %v161, 1
  ret { [8 x i8], [8 x i8] } %v163
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(i8* %v0, i8* %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i8* [ %v1, %entry ]
  %v7 = bitcast i8* %v2 to i32*
  %v4 = load i32, i32* %v7, align 4
  %v8 = bitcast i8* %v3 to i32*
  %v5 = load i32, i32* %v8, align 4
  %v6 = icmp ult i32 %v4, %v5
  ret i1 %v6
}


@llvm.used = appending global [8 x i8*] [i8* bitcast (void (i8*, i64, i8*, i64, i32, i32, i32, float, i8*, i64)* @rope to i8*), i8* bitcast (void (i8*, i64, i8*, i64, i8*, i64, i32, i32, i32, i32, i32, i32, i32)* @kv_write_row to i8*), i8* bitcast (void (i8*, i64, i8*, i64, i8*, i64, i32, i32, i32, i32, float, i8*, i64)* @attention_heads to i8*), i8* bitcast (void (i8*, i64, i8*, i64, i8*, i64)* @silu_gate to i8*), i8* bitcast (void (i8*, i64, i8*, i64, float, i32, i32, i8*, i64)* @rmsnorm_group to i8*), i8* bitcast (void (i8*, i64, i8*, i64, i32, i32, i8*, i64)* @q4k_gemv_row to i8*), i8* bitcast (void (i8*, i64, i8*, i64, i8*, i64, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i8*, i64)* @attention_paged_heads to i8*), i8* bitcast (void (float, i8*, i64, i8*, i64)* @scale_f32 to i8*)], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{void (i8*, i64, i8*, i64, i32, i32, i32, float, i8*, i64)* @rope, !"kernel", i32 1}
!1 = !{void (i8*, i64, i8*, i64, i8*, i64, i32, i32, i32, i32, i32, i32, i32)* @kv_write_row, !"kernel", i32 1}
!2 = !{void (i8*, i64, i8*, i64, i8*, i64, i32, i32, i32, i32, float, i8*, i64)* @attention_heads, !"kernel", i32 1}
!3 = !{void (i8*, i64, i8*, i64, i8*, i64)* @silu_gate, !"kernel", i32 1}
!4 = !{void (i8*, i64, i8*, i64, float, i32, i32, i8*, i64)* @rmsnorm_group, !"kernel", i32 1}
!5 = !{void (i8*, i64, i8*, i64, i32, i32, i8*, i64)* @q4k_gemv_row, !"kernel", i32 1}
!6 = !{void (i8*, i64, i8*, i64, i8*, i64, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i8*, i64)* @attention_paged_heads, !"kernel", i32 1}
!7 = !{void (float, i8*, i64, i8*, i64)* @scale_f32, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6, !7}

!nvvmir.version = !{!8}
!8 = !{i32 2, i32 0, i32 3, i32 1}
