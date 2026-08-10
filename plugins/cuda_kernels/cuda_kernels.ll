; ModuleID = 'builtin.module'
source_filename = "cuda_kernels"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_23 = addrspace(3) global [4 x float] zeroinitializer, align 4
@__shared_mem_22 = addrspace(3) global [4 x float] zeroinitializer, align 4
@__shared_mem_21 = addrspace(3) global [2048 x float] zeroinitializer, align 4
@__shared_mem_20 = addrspace(3) global [2048 x float] zeroinitializer, align 4
@__shared_mem_19 = addrspace(3) global [2048 x float] zeroinitializer, align 4
@__shared_mem_18 = addrspace(3) global [2048 x float] zeroinitializer, align 4
@__shared_mem_17 = addrspace(3) global [4096 x float] zeroinitializer, align 4
@__shared_mem_16 = addrspace(3) global [4096 x float] zeroinitializer, align 4
@__shared_mem_15 = addrspace(3) global [2 x i32] zeroinitializer, align 4
@__shared_mem_14 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_13 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_12 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_11 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_10 = addrspace(3) global [8 x float] zeroinitializer, align 4
@__shared_mem_9 = addrspace(3) global [256 x i32] zeroinitializer, align 4
@__shared_mem_8 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_7 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_6 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_5 = addrspace(3) global [8 x float] zeroinitializer, align 4
@__shared_mem_4 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_3 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_2 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_1 = addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_0 = addrspace(3) global [128 x float] zeroinitializer, align 4
declare void @llvm.trap()

define ptx_kernel void @moe_scatter_assignments(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7) #0 {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v5, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v7, 1
  br label %bb0
bb0:
  %v16 = phi { ptr, i64 } [ %v9, %entry ]
  %v17 = phi { ptr, i64 } [ %v11, %entry ]
  %v18 = phi { ptr, i64 } [ %v13, %entry ]
  %v19 = phi { ptr, i64 } [ %v15, %entry ]
  %v20 = alloca {  }, align 1
  %v21 = bitcast ptr %v20 to ptr
  %v22 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v21) #0
  br label %bb1
bb1:
  %v23 = extractvalue { ptr, i64 } %v16, 1
  %v24 = icmp uge i64 %v22, %v23
  %v25 = xor i1 %v24, 1
  br i1 %v25, label %bb3, label %bb2
bb2:
  br label %bb8
bb3:
  %v26 = icmp ult i64 %v22, %v23
  br i1 %v26, label %bb4, label %bb9
bb4:
  %v27 = extractvalue { ptr, i64 } %v16, 0
  %v28 = getelementptr inbounds i32, ptr %v27, i64 %v22
  %v29 = load i32, ptr %v28, align 4
  %v30 = zext i32 %v29 to i64
  %v31 = extractvalue { ptr, i64 } %v18, 1
  %v32 = icmp ult i64 %v30, %v31
  br i1 %v32, label %bb5, label %bb10
bb5:
  %v33 = extractvalue { ptr, i64 } %v18, 0
  %v34 = getelementptr inbounds { { i32 } }, ptr %v33, i64 %v30
  %v35 = atomicrmw add ptr %v34, i32 1 syncscope("device") monotonic
  br label %bb6
bb6:
  %v36 = zext i32 %v35 to i64
  %v37 = extractvalue { ptr, i64 } %v17, 1
  %v38 = icmp ult i64 %v30, %v37
  br i1 %v38, label %bb7, label %bb11
bb7:
  %v39 = extractvalue { ptr, i64 } %v17, 0
  %v40 = getelementptr inbounds i32, ptr %v39, i64 %v30
  %v41 = load i32, ptr %v40, align 4
  %v42 = zext i32 %v41 to i64
  %v43 = add i64 %v42, %v36
  %v44 = extractvalue { ptr, i64 } %v19, 0
  %v45 = getelementptr inbounds i32, ptr %v44, i64 %v43
  %v46 = trunc i64 %v22 to i32
  store i32 %v46, ptr %v45, align 4
  br label %bb8
bb8:
  ret void
bb9:
  call void @llvm.trap() #0
  unreachable
bb10:
  call void @llvm.trap() #0
  unreachable
bb11:
  call void @llvm.trap() #0
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #0

define ptx_kernel void @q6k_gemv_warp4(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v22 = zext i32 %v21 to i64
  %v23 = zext i32 %v17 to i64
  %v24 = add i64 %v23, 3
  %v25 = udiv i64 %v24, 4
  %v26 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v27 = zext i32 %v26 to i64
  %v28 = zext i32 %v19 to i64
  %v29 = mul i64 %v25, %v28
  %v30 = icmp uge i64 %v27, %v29
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb4, label %bb3
bb3:
  br label %bb55
bb4:
  %v32 = icmp eq i64 %v25, 0
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb5, label %bb56
bb5:
  %v34 = udiv i64 %v27, %v25
  %v35 = urem i64 %v27, %v25
  %v36 = mul i64 %v35, 4
  %v37 = zext i32 %v18 to i64
  %v38 = mul i64 %v37, 210
  %v39 = mul i64 %v34, %v37
  %v40 = mul i64 %v39, 256
  %v41 = icmp ult i64 %v36, %v23
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb8, label %bb6
bb6:
  %v43 = mul i64 %v36, %v38
  %v44 = extractvalue { ptr, i64 } %v15, 0
  %v45 = extractvalue { ptr, i64 } %v15, 1
  %v46 = extractvalue { ptr, i64 } %v16, 0
  %v47 = extractvalue { ptr, i64 } %v16, 1
  %v48 = call float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v44, i64 %v45, i64 %v43, ptr %v46, i64 %v47, i64 %v40, i32 %v18, i64 %v22) #0
  br label %bb7
bb7:
  br label %bb9
bb8:
  br label %bb9
bb9:
  %v49 = phi float [ %v48, %bb7 ], [ 0.0, %bb8 ]
  %v50 = add i64 %v36, 1
  %v51 = icmp ult i64 %v50, %v23
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb12, label %bb10
bb10:
  %v53 = mul i64 %v50, %v38
  %v54 = extractvalue { ptr, i64 } %v15, 0
  %v55 = extractvalue { ptr, i64 } %v15, 1
  %v56 = extractvalue { ptr, i64 } %v16, 0
  %v57 = extractvalue { ptr, i64 } %v16, 1
  %v58 = call float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v54, i64 %v55, i64 %v53, ptr %v56, i64 %v57, i64 %v40, i32 %v18, i64 %v22) #0
  br label %bb11
bb11:
  br label %bb13
bb12:
  br label %bb13
bb13:
  %v59 = phi float [ %v58, %bb11 ], [ 0.0, %bb12 ]
  %v60 = add i64 %v36, 2
  %v61 = icmp ult i64 %v60, %v23
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb16, label %bb14
bb14:
  %v63 = mul i64 %v60, %v38
  %v64 = extractvalue { ptr, i64 } %v15, 0
  %v65 = extractvalue { ptr, i64 } %v15, 1
  %v66 = extractvalue { ptr, i64 } %v16, 0
  %v67 = extractvalue { ptr, i64 } %v16, 1
  %v68 = call float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v64, i64 %v65, i64 %v63, ptr %v66, i64 %v67, i64 %v40, i32 %v18, i64 %v22) #0
  br label %bb15
bb15:
  br label %bb17
bb16:
  br label %bb17
bb17:
  %v69 = phi float [ %v68, %bb15 ], [ 0.0, %bb16 ]
  %v70 = add i64 %v36, 3
  %v71 = icmp ult i64 %v70, %v23
  %v72 = xor i1 %v71, 1
  br i1 %v72, label %bb20, label %bb18
bb18:
  %v73 = mul i64 %v70, %v38
  %v74 = extractvalue { ptr, i64 } %v15, 0
  %v75 = extractvalue { ptr, i64 } %v15, 1
  %v76 = extractvalue { ptr, i64 } %v16, 0
  %v77 = extractvalue { ptr, i64 } %v16, 1
  %v78 = call float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v74, i64 %v75, i64 %v73, ptr %v76, i64 %v77, i64 %v40, i32 %v18, i64 %v22) #0
  br label %bb19
bb19:
  br label %bb21
bb20:
  br label %bb21
bb21:
  %v79 = phi float [ %v78, %bb19 ], [ 0.0, %bb20 ]
  %v80 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v22
  br label %bb22
bb22:
  store float %v49, ptr addrspace(3) %v80, align 4
  %v81 = add i64 32, %v22
  %v82 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v81
  br label %bb23
bb23:
  store float %v59, ptr addrspace(3) %v82, align 4
  %v83 = add i64 64, %v22
  %v84 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v83
  br label %bb24
bb24:
  store float %v69, ptr addrspace(3) %v84, align 4
  %v85 = add i64 96, %v22
  %v86 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v85
  br label %bb25
bb25:
  store float %v79, ptr addrspace(3) %v86, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb26
bb26:
  br label %bb27
bb27:
  %v88 = phi i64 [ 16, %bb26 ], [ %v122, %bb40 ]
  %v89 = icmp ugt i64 %v88, 0
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb41, label %bb28
bb28:
  %v91 = icmp ult i64 %v22, %v88
  %v92 = xor i1 %v91, 1
  br i1 %v92, label %bb38, label %bb29
bb29:
  %v93 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v94 = add i64 %v22, %v88
  %v95 = getelementptr inbounds float, ptr addrspace(3) %v93, i64 %v94
  br label %bb30
bb30:
  %v96 = load float, ptr addrspace(3) %v95, align 4
  %v97 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v22
  br label %bb31
bb31:
  %v98 = load float, ptr addrspace(3) %v97, align 4
  %v99 = fadd contract float %v98, %v96
  store float %v99, ptr addrspace(3) %v97, align 4
  %v100 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v101 = add i64 %v81, %v88
  %v102 = getelementptr inbounds float, ptr addrspace(3) %v100, i64 %v101
  br label %bb32
bb32:
  %v103 = load float, ptr addrspace(3) %v102, align 4
  %v104 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v81
  br label %bb33
bb33:
  %v105 = load float, ptr addrspace(3) %v104, align 4
  %v106 = fadd contract float %v105, %v103
  store float %v106, ptr addrspace(3) %v104, align 4
  %v107 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v108 = add i64 %v83, %v88
  %v109 = getelementptr inbounds float, ptr addrspace(3) %v107, i64 %v108
  br label %bb34
bb34:
  %v110 = load float, ptr addrspace(3) %v109, align 4
  %v111 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v83
  br label %bb35
bb35:
  %v112 = load float, ptr addrspace(3) %v111, align 4
  %v113 = fadd contract float %v112, %v110
  store float %v113, ptr addrspace(3) %v111, align 4
  %v114 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v115 = add i64 %v85, %v88
  %v116 = getelementptr inbounds float, ptr addrspace(3) %v114, i64 %v115
  br label %bb36
bb36:
  %v117 = load float, ptr addrspace(3) %v116, align 4
  %v118 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v85
  br label %bb37
bb37:
  %v119 = load float, ptr addrspace(3) %v118, align 4
  %v120 = fadd contract float %v119, %v117
  store float %v120, ptr addrspace(3) %v118, align 4
  br label %bb39
bb38:
  br label %bb39
bb39:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb40
bb40:
  %v122 = udiv i64 %v88, 2
  br label %bb27
bb41:
  %v123 = icmp eq i64 %v22, 0
  br i1 %v123, label %bb42, label %bb54
bb42:
  %v124 = mul i64 %v34, %v23
  %v125 = add i64 %v124, %v36
  %v126 = xor i1 %v41, 1
  br i1 %v126, label %bb45, label %bb43
bb43:
  %v127 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v128 = getelementptr inbounds float, ptr addrspace(3) %v127, i64 0
  br label %bb44
bb44:
  %v129 = load float, ptr addrspace(3) %v128, align 4
  %v130 = extractvalue { ptr, i64 } %v20, 0
  %v131 = getelementptr inbounds float, ptr %v130, i64 %v125
  store float %v129, ptr %v131, align 4
  br label %bb45
bb45:
  %v132 = xor i1 %v51, 1
  br i1 %v132, label %bb48, label %bb46
bb46:
  %v133 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v134 = getelementptr inbounds float, ptr addrspace(3) %v133, i64 32
  br label %bb47
bb47:
  %v135 = load float, ptr addrspace(3) %v134, align 4
  %v136 = add i64 %v125, 1
  %v137 = extractvalue { ptr, i64 } %v20, 0
  %v138 = getelementptr inbounds float, ptr %v137, i64 %v136
  store float %v135, ptr %v138, align 4
  br label %bb48
bb48:
  %v139 = xor i1 %v61, 1
  br i1 %v139, label %bb51, label %bb49
bb49:
  %v140 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v141 = getelementptr inbounds float, ptr addrspace(3) %v140, i64 64
  br label %bb50
bb50:
  %v142 = load float, ptr addrspace(3) %v141, align 4
  %v143 = add i64 %v125, 2
  %v144 = extractvalue { ptr, i64 } %v20, 0
  %v145 = getelementptr inbounds float, ptr %v144, i64 %v143
  store float %v142, ptr %v145, align 4
  br label %bb51
bb51:
  %v146 = xor i1 %v71, 1
  br i1 %v146, label %bb54, label %bb52
bb52:
  %v147 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v148 = getelementptr inbounds float, ptr addrspace(3) %v147, i64 96
  br label %bb53
bb53:
  %v149 = load float, ptr addrspace(3) %v148, align 4
  %v150 = add i64 %v125, 3
  %v151 = extractvalue { ptr, i64 } %v20, 0
  %v152 = getelementptr inbounds float, ptr %v151, i64 %v150
  store float %v149, ptr %v152, align 4
  br label %bb54
bb54:
  br label %bb55
bb55:
  ret void
bb56:
  call void @llvm.trap() #0
  unreachable
}

declare float @__nv_expf(float)

define ptx_kernel void @attention_paged_heads(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, ptr %v16, i64 %v17) #0 {
entry:
  %v18 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v1, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v3, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v5, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v17, 1
  br label %bb0
bb0:
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi { ptr, i64 } [ %v23, %entry ]
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
  %v39 = phi { ptr, i64 } [ %v25, %entry ]
  %v40 = alloca {  }, align 1
  %v41 = bitcast ptr %v40 to ptr
  %v42 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v41) #0
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
  %v48 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v30, i32 1) #0
  br label %bb4
bb4:
  %v49 = icmp eq i32 %v48, 0
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb5, label %bb44
bb5:
  %v51 = udiv i32 %v29, %v48
  %v52 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v51, i32 1) #0
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
  %v68 = extractvalue { ptr, i64 } %v39, 0
  %v69 = getelementptr inbounds float, ptr %v68, i64 %v67
  store float 0.0, ptr %v69, align 4
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
  %v85 = extractvalue { ptr, i64 } %v28, 1
  %v86 = icmp ult i64 %v84, %v85
  br i1 %v86, label %bb14, label %bb47
bb14:
  %v87 = extractvalue { ptr, i64 } %v28, 0
  %v88 = getelementptr inbounds i32, ptr %v87, i64 %v84
  %v89 = load i32, ptr %v88, align 4
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
  %v103 = extractvalue { ptr, i64 } %v27, 1
  %v104 = icmp ult i64 %v102, %v103
  br i1 %v104, label %bb17, label %bb48
bb17:
  %v105 = extractvalue { ptr, i64 } %v27, 0
  %v106 = getelementptr inbounds i8, ptr %v105, i64 %v102
  %v107 = load i8, ptr %v106, align 1
  %v108 = zext i8 %v107 to i16
  %v109 = mul i64 %v98, 2
  %v110 = add i64 %v96, %v109
  %v111 = add i64 %v110, 1
  %v112 = icmp ult i64 %v111, %v103
  br i1 %v112, label %bb18, label %bb49
bb18:
  %v113 = extractvalue { ptr, i64 } %v27, 0
  %v114 = getelementptr inbounds i8, ptr %v113, i64 %v111
  %v115 = load i8, ptr %v114, align 1
  %v116 = zext i8 %v115 to i16
  %v117 = trunc i32 8 to i16
  %v118 = and i16 %v117, 15
  %v119 = shl i16 %v116, %v118
  %v120 = or i16 %v108, %v119
  %v121 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v120) #0
  br label %bb19
bb19:
  %v122 = add i64 %v63, %v98
  %v123 = extractvalue { ptr, i64 } %v26, 1
  %v124 = icmp ult i64 %v122, %v123
  br i1 %v124, label %bb20, label %bb50
bb20:
  %v125 = extractvalue { ptr, i64 } %v26, 0
  %v126 = getelementptr inbounds float, ptr %v125, i64 %v122
  %v127 = load float, ptr %v126, align 4
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
  %v148 = extractvalue { ptr, i64 } %v27, 1
  %v149 = icmp ult i64 %v147, %v148
  br i1 %v149, label %bb30, label %bb51
bb30:
  %v150 = extractvalue { ptr, i64 } %v27, 0
  %v151 = getelementptr inbounds i8, ptr %v150, i64 %v147
  %v152 = load i8, ptr %v151, align 1
  %v153 = zext i8 %v152 to i16
  %v154 = mul i64 %v143, 2
  %v155 = add i64 %v192, %v154
  %v156 = add i64 %v155, 1
  %v157 = icmp ult i64 %v156, %v148
  br i1 %v157, label %bb31, label %bb52
bb31:
  %v158 = extractvalue { ptr, i64 } %v27, 0
  %v159 = getelementptr inbounds i8, ptr %v158, i64 %v156
  %v160 = load i8, ptr %v159, align 1
  %v161 = zext i8 %v160 to i16
  %v162 = trunc i32 8 to i16
  %v163 = and i16 %v162, 15
  %v164 = shl i16 %v161, %v163
  %v165 = or i16 %v153, %v164
  %v166 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v165) #0
  br label %bb32
bb32:
  %v167 = add i64 %v63, %v143
  %v168 = extractvalue { ptr, i64 } %v39, 0
  %v169 = getelementptr inbounds float, ptr %v168, i64 %v167
  %v170 = load float, ptr %v169, align 4
  %v171 = fmul contract float %v170, %v140
  %v172 = fmul contract float %v142, %v166
  %v173 = fadd contract float %v171, %v172
  store float %v173, ptr %v169, align 4
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
  %v183 = extractvalue { ptr, i64 } %v39, 0
  %v184 = getelementptr inbounds float, ptr %v183, i64 %v182
  %v185 = load float, ptr %v184, align 4
  %v186 = fmul contract float %v185, %v178
  store float %v186, ptr %v184, align 4
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
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @shortconv_mix(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, ptr %v8, i64 %v9) #0 {
entry:
  %v10 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v1, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v3, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v5, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v9, 1
  br label %bb0
bb0:
  %v18 = phi { ptr, i64 } [ %v11, %entry ]
  %v19 = phi { ptr, i64 } [ %v13, %entry ]
  %v20 = phi { ptr, i64 } [ %v15, %entry ]
  %v21 = phi i32 [ %v6, %entry ]
  %v22 = phi i32 [ %v7, %entry ]
  %v23 = phi { ptr, i64 } [ %v17, %entry ]
  %v24 = alloca {  }, align 1
  %v25 = bitcast ptr %v24 to ptr
  %v26 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v25) #0
  br label %bb1
bb1:
  %v27 = zext i32 %v21 to i64
  %v28 = icmp uge i64 %v26, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb2, label %bb3
bb2:
  %v30 = icmp eq i32 %v22, 0
  br i1 %v30, label %bb3, label %bb4
bb3:
  br label %bb22
bb4:
  %v31 = zext i32 %v22 to i64
  %v32 = sub i64 %v31, 1
  %v33 = extractvalue { ptr, i64 } %v18, 1
  %v34 = icmp ult i64 %v26, %v33
  br i1 %v34, label %bb5, label %bb30
bb5:
  %v35 = extractvalue { ptr, i64 } %v18, 0
  %v36 = getelementptr inbounds float, ptr %v35, i64 %v26
  %v37 = load float, ptr %v36, align 4
  %v38 = add i64 %v27, %v26
  %v39 = icmp ult i64 %v38, %v33
  br i1 %v39, label %bb6, label %bb31
bb6:
  %v40 = extractvalue { ptr, i64 } %v18, 0
  %v41 = getelementptr inbounds float, ptr %v40, i64 %v38
  %v42 = load float, ptr %v41, align 4
  %v43 = mul i64 2, %v27
  %v44 = add i64 %v43, %v26
  %v45 = icmp ult i64 %v44, %v33
  br i1 %v45, label %bb7, label %bb32
bb7:
  %v46 = extractvalue { ptr, i64 } %v18, 0
  %v47 = getelementptr inbounds float, ptr %v46, i64 %v44
  %v48 = load float, ptr %v47, align 4
  %v49 = fmul contract float %v37, %v48
  %v50 = mul i64 %v26, %v31
  %v51 = add i64 %v50, %v32
  %v52 = extractvalue { ptr, i64 } %v19, 1
  %v53 = icmp ult i64 %v51, %v52
  br i1 %v53, label %bb8, label %bb33
bb8:
  %v54 = extractvalue { ptr, i64 } %v19, 0
  %v55 = getelementptr inbounds float, ptr %v54, i64 %v51
  %v56 = load float, ptr %v55, align 4
  %v57 = fmul contract float %v49, %v56
  br label %bb9
bb9:
  %v58 = phi float [ %v57, %bb8 ], [ %v73, %bb11 ]
  %v59 = phi i64 [ 0, %bb8 ], [ %v74, %bb11 ]
  %v60 = icmp ult i64 %v59, %v32
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb12, label %bb10
bb10:
  %v62 = mul i64 %v59, %v27
  %v63 = add i64 %v62, %v26
  %v64 = extractvalue { ptr, i64 } %v20, 0
  %v65 = getelementptr inbounds float, ptr %v64, i64 %v63
  %v66 = load float, ptr %v65, align 4
  %v67 = add i64 %v50, %v59
  %v68 = icmp ult i64 %v67, %v52
  br i1 %v68, label %bb11, label %bb34
bb11:
  %v69 = extractvalue { ptr, i64 } %v19, 0
  %v70 = getelementptr inbounds float, ptr %v69, i64 %v67
  %v71 = load float, ptr %v70, align 4
  %v72 = fmul contract float %v66, %v71
  %v73 = fadd contract float %v58, %v72
  %v74 = add i64 %v59, 1
  br label %bb9
bb12:
  %v75 = icmp eq i64 %v26, 18446744073709551615
  br i1 %v75, label %bb26, label %bb23
bb13:
  %v76 = extractvalue { ptr } %v107, 0
  %v77 = fmul contract float %v42, %v58
  store float %v77, ptr %v76, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v78 = phi i64 [ 0, %bb15 ], [ %v95, %bb20 ]
  %v79 = icmp ult i64 %v78, %v32
  %v80 = xor i1 %v79, 1
  br i1 %v80, label %bb21, label %bb17
bb17:
  %v81 = add i64 %v78, 1
  %v82 = icmp ult i64 %v81, %v32
  %v83 = xor i1 %v82, 1
  br i1 %v83, label %bb19, label %bb18
bb18:
  %v84 = add i64 %v78, 1
  %v85 = mul i64 %v84, %v27
  %v86 = add i64 %v85, %v26
  %v87 = extractvalue { ptr, i64 } %v20, 0
  %v88 = getelementptr inbounds float, ptr %v87, i64 %v86
  %v89 = load float, ptr %v88, align 4
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v90 = phi float [ %v89, %bb18 ], [ %v49, %bb19 ]
  %v91 = mul i64 %v78, %v27
  %v92 = add i64 %v91, %v26
  %v93 = extractvalue { ptr, i64 } %v20, 0
  %v94 = getelementptr inbounds float, ptr %v93, i64 %v92
  store float %v90, ptr %v94, align 4
  %v95 = add i64 %v78, 1
  br label %bb16
bb21:
  br label %bb22
bb22:
  ret void
bb23:
  %v96 = extractvalue { ptr, i64 } %v23, 1
  %v97 = icmp ult i64 %v26, %v96
  %v98 = xor i1 %v97, 1
  br i1 %v98, label %bb25, label %bb24
bb24:
  %v99 = extractvalue { ptr, i64 } %v23, 0
  %v100 = getelementptr inbounds float, ptr %v99, i64 %v26
  %v101 = insertvalue { ptr } undef, ptr %v100, 0
  %v102 = extractvalue { ptr } %v101, 0
  br label %bb27
bb25:
  br label %bb26
bb26:
  %v103 = inttoptr i64 0 to ptr
  %v104 = insertvalue { ptr } undef, ptr %v103, 0
  %v105 = extractvalue { ptr } %v104, 0
  br label %bb27
bb27:
  %v106 = phi ptr [ %v102, %bb24 ], [ %v105, %bb26 ]
  %v107 = insertvalue { ptr } undef, ptr %v106, 0
  %v108 = extractvalue { ptr } %v107, 0
  %v109 = ptrtoint ptr %v108 to i64
  %v110 = sub i64 %v109, 0
  %v111 = icmp ule i64 %v110, 0
  %v112 = add i64 %v110, 0
  %v113 = select i1 %v111, i64 %v112, i64 1
  %v114 = icmp eq i64 %v113, 1
  br i1 %v114, label %bb13, label %bb28
bb28:
  %v115 = icmp eq i64 %v113, 0
  br i1 %v115, label %bb14, label %bb29
bb29:
  unreachable
bb30:
  call void @llvm.trap() #0
  unreachable
bb31:
  call void @llvm.trap() #0
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_q8_0_rows(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = bitcast ptr %v21 to ptr
  %v24 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v23) #0
  br label %bb1
bb1:
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v18 to i64
  %v27 = mul i64 %v25, %v26
  %v28 = icmp uge i64 %v24, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb3, label %bb2
bb2:
  br label %bb13
bb3:
  %v30 = icmp eq i64 %v26, 0
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb4, label %bb21
bb4:
  %v32 = udiv i64 %v24, %v26
  %v33 = urem i64 %v24, %v26
  %v34 = zext i32 %v19 to i64
  %v35 = mul i64 %v34, 34
  %v36 = udiv i64 %v33, 32
  %v37 = urem i64 %v33, 32
  %v38 = extractvalue { ptr, i64 } %v16, 1
  %v39 = icmp ult i64 %v32, %v38
  br i1 %v39, label %bb5, label %bb22
bb5:
  %v40 = extractvalue { ptr, i64 } %v16, 0
  %v41 = getelementptr inbounds i32, ptr %v40, i64 %v32
  %v42 = load i32, ptr %v41, align 4
  %v43 = zext i32 %v42 to i64
  %v44 = mul i64 %v43, %v35
  %v45 = mul i64 %v36, 34
  %v46 = add i64 %v44, %v45
  %v47 = extractvalue { ptr, i64 } %v15, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb6, label %bb23
bb6:
  %v49 = extractvalue { ptr, i64 } %v15, 0
  %v50 = getelementptr inbounds i8, ptr %v49, i64 %v46
  %v51 = load i8, ptr %v50, align 1
  %v52 = add i64 %v46, 1
  %v53 = icmp ult i64 %v52, %v47
  br i1 %v53, label %bb7, label %bb24
bb7:
  %v54 = extractvalue { ptr, i64 } %v15, 0
  %v55 = getelementptr inbounds i8, ptr %v54, i64 %v52
  %v56 = load i8, ptr %v55, align 1
  %v57 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v51, ptr %v57, align 1
  %v58 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v56, ptr %v58, align 1
  %v59 = load [2 x i8], ptr %v22, align 1
  %v60 = alloca [2 x i8], align 2
  store [2 x i8] %v59, ptr %v60, align 2
  %v61 = load i16, ptr %v60, align 2
  %v62 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v61) #0
  br label %bb8
bb8:
  %v63 = icmp eq i64 %v24, 18446744073709551615
  br i1 %v63, label %bb17, label %bb14
bb9:
  %v64 = extractvalue { ptr } %v85, 0
  %v65 = add i64 %v46, 2
  %v66 = add i64 %v65, %v37
  %v67 = icmp ult i64 %v66, %v47
  br i1 %v67, label %bb10, label %bb25
bb10:
  %v68 = extractvalue { ptr, i64 } %v15, 0
  %v69 = getelementptr inbounds i8, ptr %v68, i64 %v66
  %v70 = load i8, ptr %v69, align 1
  %v71 = bitcast i8 %v70 to i8
  %v72 = sitofp i8 %v71 to float
  %v73 = fmul contract float %v62, %v72
  store float %v73, ptr %v64, align 4
  br label %bb12
bb11:
  br label %bb12
bb12:
  br label %bb13
bb13:
  ret void
bb14:
  %v74 = extractvalue { ptr, i64 } %v20, 1
  %v75 = icmp ult i64 %v24, %v74
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb16, label %bb15
bb15:
  %v77 = extractvalue { ptr, i64 } %v20, 0
  %v78 = getelementptr inbounds float, ptr %v77, i64 %v24
  %v79 = insertvalue { ptr } undef, ptr %v78, 0
  %v80 = extractvalue { ptr } %v79, 0
  br label %bb18
bb16:
  br label %bb17
bb17:
  %v81 = inttoptr i64 0 to ptr
  %v82 = insertvalue { ptr } undef, ptr %v81, 0
  %v83 = extractvalue { ptr } %v82, 0
  br label %bb18
bb18:
  %v84 = phi ptr [ %v80, %bb15 ], [ %v83, %bb17 ]
  %v85 = insertvalue { ptr } undef, ptr %v84, 0
  %v86 = extractvalue { ptr } %v85, 0
  %v87 = ptrtoint ptr %v86 to i64
  %v88 = sub i64 %v87, 0
  %v89 = icmp ule i64 %v88, 0
  %v90 = add i64 %v88, 0
  %v91 = select i1 %v89, i64 %v90, i64 1
  %v92 = icmp eq i64 %v91, 1
  br i1 %v92, label %bb9, label %bb19
bb19:
  %v93 = icmp eq i64 %v91, 0
  br i1 %v93, label %bb11, label %bb20
bb20:
  unreachable
bb21:
  call void @llvm.trap() #0
  unreachable
bb22:
  call void @llvm.trap() #0
  unreachable
bb23:
  call void @llvm.trap() #0
  unreachable
bb24:
  call void @llvm.trap() #0
  unreachable
bb25:
  call void @llvm.trap() #0
  unreachable
}

declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32) #0

define ptx_kernel void @q6k_q8_gemv_warp4(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, ptr %v9, i64 %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v5, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v10, 1
  br label %bb0
bb0:
  %v19 = phi { ptr, i64 } [ %v12, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = phi i32 [ %v6, %entry ]
  %v23 = phi i32 [ %v7, %entry ]
  %v24 = phi i32 [ %v8, %entry ]
  %v25 = phi { ptr, i64 } [ %v18, %entry ]
  %v26 = alloca [4 x i8], align 1
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v28 = zext i32 %v27 to i64
  %v29 = zext i32 %v22 to i64
  %v30 = add i64 %v29, 3
  %v31 = udiv i64 %v30, 4
  %v32 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v33 = zext i32 %v32 to i64
  %v34 = zext i32 %v24 to i64
  %v35 = mul i64 %v31, %v34
  %v36 = icmp uge i64 %v33, %v35
  %v37 = xor i1 %v36, 1
  br i1 %v37, label %bb4, label %bb3
bb3:
  br label %bb40
bb4:
  %v38 = icmp eq i64 %v31, 0
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb5, label %bb45
bb5:
  %v40 = udiv i64 %v33, %v31
  %v41 = urem i64 %v33, %v31
  %v42 = mul i64 %v41, 4
  %v43 = zext i32 %v23 to i64
  %v44 = mul i64 %v43, 210
  %v45 = mul i64 %v40, %v43
  %v46 = mul i64 %v45, 256
  br label %bb6
bb6:
  %v47 = phi float [ 0.0, %bb5 ], [ %v55, %bb22 ]
  %v48 = phi float [ 0.0, %bb5 ], [ %v56, %bb22 ]
  %v49 = phi float [ 0.0, %bb5 ], [ %v57, %bb22 ]
  %v50 = phi float [ 0.0, %bb5 ], [ %v58, %bb22 ]
  %v51 = phi i64 [ 0, %bb5 ], [ %v148, %bb22 ]
  %v52 = icmp ult i64 %v51, %v43
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb23, label %bb7
bb7:
  %v54 = mul i64 %v28, 8
  br label %bb8
bb8:
  %v55 = phi float [ %v47, %bb7 ], [ %v113, %bb21 ]
  %v56 = phi float [ %v48, %bb7 ], [ %v124, %bb21 ]
  %v57 = phi float [ %v49, %bb7 ], [ %v135, %bb21 ]
  %v58 = phi float [ %v50, %bb7 ], [ %v146, %bb21 ]
  %v59 = phi i64 [ 0, %bb7 ], [ %v147, %bb21 ]
  %v60 = icmp ult i64 %v59, 2
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb22, label %bb9
bb9:
  %v62 = mul i64 %v59, 4
  %v63 = add i64 %v54, %v62
  %v64 = mul i64 %v51, 256
  %v65 = add i64 %v46, %v64
  %v66 = add i64 %v65, %v63
  %v67 = extractvalue { ptr, i64 } %v20, 1
  %v68 = icmp ult i64 %v66, %v67
  %v69 = extractvalue { ptr, i64 } %v20, 0
  %v70 = getelementptr inbounds i8, ptr %v69, i64 %v66
  %v71 = load i8, ptr %v70, align 1
  %v72 = bitcast i8 %v71 to i8
  %v73 = add i64 %v66, 1
  %v74 = icmp ult i64 %v73, %v67
  %v75 = extractvalue { ptr, i64 } %v20, 0
  %v76 = getelementptr inbounds i8, ptr %v75, i64 %v73
  %v77 = load i8, ptr %v76, align 1
  %v78 = bitcast i8 %v77 to i8
  %v79 = add i64 %v66, 2
  %v80 = icmp ult i64 %v79, %v67
  %v81 = extractvalue { ptr, i64 } %v20, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = bitcast i8 %v83 to i8
  %v85 = add i64 %v66, 3
  %v86 = icmp ult i64 %v85, %v67
  %v87 = extractvalue { ptr, i64 } %v20, 0
  %v88 = getelementptr inbounds i8, ptr %v87, i64 %v85
  %v89 = load i8, ptr %v88, align 1
  %v90 = bitcast i8 %v89 to i8
  %v91 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 0
  store i8 %v72, ptr %v91, align 1
  %v92 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 1
  store i8 %v78, ptr %v92, align 1
  %v93 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 2
  store i8 %v84, ptr %v93, align 1
  %v94 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 3
  store i8 %v90, ptr %v94, align 1
  %v95 = load [4 x i8], ptr %v26, align 1
  %v96 = alloca [4 x i8], align 4
  store [4 x i8] %v95, ptr %v96, align 4
  %v97 = load i32, ptr %v96, align 4
  %v98 = udiv i64 %v66, 32
  %v99 = extractvalue { ptr, i64 } %v21, 1
  %v100 = icmp ult i64 %v98, %v99
  %v101 = extractvalue { ptr, i64 } %v21, 0
  %v102 = getelementptr inbounds float, ptr %v101, i64 %v98
  %v103 = load float, ptr %v102, align 4
  %v104 = icmp ult i64 %v42, %v29
  %v105 = xor i1 %v104, 1
  br i1 %v105, label %bb12, label %bb10
bb10:
  %v106 = mul i64 %v42, %v44
  %v107 = mul i64 %v51, 210
  %v108 = add i64 %v106, %v107
  %v109 = extractvalue { ptr, i64 } %v19, 0
  %v110 = extractvalue { ptr, i64 } %v19, 1
  %v111 = call float @cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk(ptr %v109, i64 %v110, i64 %v108, i64 %v63, i32 %v97, float %v103) #0
  br label %bb11
bb11:
  %v112 = fadd contract float %v55, %v111
  br label %bb12
bb12:
  %v113 = phi float [ %v55, %bb9 ], [ %v112, %bb11 ]
  %v114 = add i64 %v42, 1
  %v115 = icmp ult i64 %v114, %v29
  %v116 = xor i1 %v115, 1
  br i1 %v116, label %bb15, label %bb13
bb13:
  %v117 = mul i64 %v114, %v44
  %v118 = mul i64 %v51, 210
  %v119 = add i64 %v117, %v118
  %v120 = extractvalue { ptr, i64 } %v19, 0
  %v121 = extractvalue { ptr, i64 } %v19, 1
  %v122 = call float @cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk(ptr %v120, i64 %v121, i64 %v119, i64 %v63, i32 %v97, float %v103) #0
  br label %bb14
bb14:
  %v123 = fadd contract float %v56, %v122
  br label %bb15
bb15:
  %v124 = phi float [ %v56, %bb12 ], [ %v123, %bb14 ]
  %v125 = add i64 %v42, 2
  %v126 = icmp ult i64 %v125, %v29
  %v127 = xor i1 %v126, 1
  br i1 %v127, label %bb18, label %bb16
bb16:
  %v128 = mul i64 %v125, %v44
  %v129 = mul i64 %v51, 210
  %v130 = add i64 %v128, %v129
  %v131 = extractvalue { ptr, i64 } %v19, 0
  %v132 = extractvalue { ptr, i64 } %v19, 1
  %v133 = call float @cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk(ptr %v131, i64 %v132, i64 %v130, i64 %v63, i32 %v97, float %v103) #0
  br label %bb17
bb17:
  %v134 = fadd contract float %v57, %v133
  br label %bb18
bb18:
  %v135 = phi float [ %v57, %bb15 ], [ %v134, %bb17 ]
  %v136 = add i64 %v42, 3
  %v137 = icmp ult i64 %v136, %v29
  %v138 = xor i1 %v137, 1
  br i1 %v138, label %bb21, label %bb19
bb19:
  %v139 = mul i64 %v136, %v44
  %v140 = mul i64 %v51, 210
  %v141 = add i64 %v139, %v140
  %v142 = extractvalue { ptr, i64 } %v19, 0
  %v143 = extractvalue { ptr, i64 } %v19, 1
  %v144 = call float @cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk(ptr %v142, i64 %v143, i64 %v141, i64 %v63, i32 %v97, float %v103) #0
  br label %bb20
bb20:
  %v145 = fadd contract float %v58, %v144
  br label %bb21
bb21:
  %v146 = phi float [ %v58, %bb18 ], [ %v145, %bb20 ]
  %v147 = add i64 %v59, 1
  br label %bb8
bb22:
  %v148 = add i64 %v51, 1
  br label %bb6
bb23:
  br label %bb24
bb24:
  %v149 = phi float [ %v47, %bb23 ], [ %v182, %bb44 ]
  %v150 = phi float [ %v48, %bb23 ], [ %v184, %bb44 ]
  %v151 = phi float [ %v49, %bb23 ], [ %v186, %bb44 ]
  %v152 = phi float [ %v50, %bb23 ], [ %v188, %bb44 ]
  %v153 = phi i32 [ 16, %bb23 ], [ %v189, %bb44 ]
  %v154 = icmp ugt i32 %v153, 0
  %v155 = xor i1 %v154, 1
  br i1 %v155, label %bb26, label %bb25
bb25:
  %v156 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v149, i32 %v153, i32 31) #0
  br label %bb41
bb26:
  %v157 = icmp eq i64 %v28, 0
  br i1 %v157, label %bb27, label %bb39
bb27:
  %v158 = mul i64 %v40, %v29
  %v159 = add i64 %v158, %v42
  %v160 = icmp ult i64 %v42, %v29
  %v161 = xor i1 %v160, 1
  br i1 %v161, label %bb29, label %bb28
bb28:
  %v162 = extractvalue { ptr, i64 } %v25, 0
  %v163 = getelementptr inbounds float, ptr %v162, i64 %v159
  store float %v149, ptr %v163, align 4
  br label %bb29
bb29:
  %v164 = add i64 %v42, 1
  %v165 = icmp ult i64 %v164, %v29
  %v166 = xor i1 %v165, 1
  br i1 %v166, label %bb31, label %bb30
bb30:
  %v167 = add i64 %v159, 1
  %v168 = extractvalue { ptr, i64 } %v25, 0
  %v169 = getelementptr inbounds float, ptr %v168, i64 %v167
  store float %v150, ptr %v169, align 4
  br label %bb32
bb31:
  br label %bb32
bb32:
  %v170 = add i64 %v42, 2
  %v171 = icmp ult i64 %v170, %v29
  %v172 = xor i1 %v171, 1
  br i1 %v172, label %bb34, label %bb33
bb33:
  %v173 = add i64 %v159, 2
  %v174 = extractvalue { ptr, i64 } %v25, 0
  %v175 = getelementptr inbounds float, ptr %v174, i64 %v173
  store float %v151, ptr %v175, align 4
  br label %bb35
bb34:
  br label %bb35
bb35:
  %v176 = add i64 %v42, 3
  %v177 = icmp ult i64 %v176, %v29
  %v178 = xor i1 %v177, 1
  br i1 %v178, label %bb37, label %bb36
bb36:
  %v179 = add i64 %v159, 3
  %v180 = extractvalue { ptr, i64 } %v25, 0
  %v181 = getelementptr inbounds float, ptr %v180, i64 %v179
  store float %v152, ptr %v181, align 4
  br label %bb38
bb37:
  br label %bb38
bb38:
  br label %bb39
bb39:
  br label %bb40
bb40:
  ret void
bb41:
  %v182 = fadd contract float %v149, %v156
  %v183 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v150, i32 %v153, i32 31) #0
  br label %bb42
bb42:
  %v184 = fadd contract float %v150, %v183
  %v185 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v151, i32 %v153, i32 31) #0
  br label %bb43
bb43:
  %v186 = fadd contract float %v151, %v185
  %v187 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v152, i32 %v153, i32 31) #0
  br label %bb44
bb44:
  %v188 = fadd contract float %v152, %v187
  %v189 = udiv i32 %v153, 2
  br label %bb24
bb45:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q4k_gemm_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v22 = zext i32 %v21 to i64
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v19 to i64
  %v27 = mul i64 %v25, %v26
  %v28 = icmp uge i64 %v24, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb4, label %bb3
bb3:
  br label %bb24
bb4:
  %v30 = icmp eq i64 %v25, 0
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb5, label %bb25
bb5:
  %v32 = urem i64 %v24, %v25
  %v33 = udiv i64 %v24, %v25
  %v34 = zext i32 %v18 to i64
  %v35 = mul i64 %v34, 144
  br label %bb6
bb6:
  %v36 = phi float [ 0.0, %bb5 ], [ %v51, %bb8 ]
  %v37 = phi i64 [ %v22, %bb5 ], [ %v52, %bb8 ]
  %v38 = icmp ult i64 %v37, %v34
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb9, label %bb7
bb7:
  %v40 = mul i64 %v32, %v35
  %v41 = mul i64 %v37, 144
  %v42 = add i64 %v40, %v41
  %v43 = mul i64 %v33, %v34
  %v44 = add i64 %v43, %v37
  %v45 = mul i64 %v44, 256
  %v46 = extractvalue { ptr, i64 } %v15, 0
  %v47 = extractvalue { ptr, i64 } %v15, 1
  %v48 = extractvalue { ptr, i64 } %v16, 0
  %v49 = extractvalue { ptr, i64 } %v16, 1
  %v50 = call float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v46, i64 %v47, i64 %v42, ptr %v48, i64 %v49, i64 %v45, i32 1) #0
  br label %bb8
bb8:
  %v51 = fadd contract float %v36, %v50
  %v52 = add i64 %v37, 32
  br label %bb6
bb9:
  %v53 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v22
  br label %bb10
bb10:
  store float %v36, ptr addrspace(3) %v53, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb11
bb11:
  br label %bb12
bb12:
  %v55 = phi i64 [ 16, %bb11 ], [ %v68, %bb19 ]
  %v56 = icmp ugt i64 %v55, 0
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb20, label %bb13
bb13:
  %v58 = icmp ult i64 %v22, %v55
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb17, label %bb14
bb14:
  %v60 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v61 = add i64 %v22, %v55
  %v62 = getelementptr inbounds float, ptr addrspace(3) %v60, i64 %v61
  br label %bb15
bb15:
  %v63 = load float, ptr addrspace(3) %v62, align 4
  %v64 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v22
  br label %bb16
bb16:
  %v65 = load float, ptr addrspace(3) %v64, align 4
  %v66 = fadd contract float %v65, %v63
  store float %v66, ptr addrspace(3) %v64, align 4
  br label %bb18
bb17:
  br label %bb18
bb18:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb19
bb19:
  %v68 = udiv i64 %v55, 2
  br label %bb12
bb20:
  %v69 = icmp eq i64 %v22, 0
  br i1 %v69, label %bb21, label %bb23
bb21:
  %v70 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v71 = getelementptr inbounds float, ptr addrspace(3) %v70, i64 0
  br label %bb22
bb22:
  %v72 = load float, ptr addrspace(3) %v71, align 4
  %v73 = extractvalue { ptr, i64 } %v20, 0
  %v74 = getelementptr inbounds float, ptr %v73, i64 %v24
  store float %v72, ptr %v74, align 4
  br label %bb23
bb23:
  br label %bb24
bb24:
  ret void
bb25:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @add_f32(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = alloca {  }, align 1
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v16) #0
  br label %bb1
bb1:
  %v18 = icmp eq i64 %v17, 18446744073709551615
  br i1 %v18, label %bb10, label %bb7
bb2:
  %v19 = extractvalue { ptr } %v42, 0
  %v20 = extractvalue { ptr, i64 } %v12, 1
  %v21 = icmp ult i64 %v17, %v20
  br i1 %v21, label %bb3, label %bb14
bb3:
  %v22 = extractvalue { ptr, i64 } %v12, 0
  %v23 = getelementptr inbounds float, ptr %v22, i64 %v17
  %v24 = load float, ptr %v23, align 4
  %v25 = extractvalue { ptr, i64 } %v13, 1
  %v26 = icmp ult i64 %v17, %v25
  br i1 %v26, label %bb4, label %bb15
bb4:
  %v27 = extractvalue { ptr, i64 } %v13, 0
  %v28 = getelementptr inbounds float, ptr %v27, i64 %v17
  %v29 = load float, ptr %v28, align 4
  %v30 = fadd contract float %v24, %v29
  store float %v30, ptr %v19, align 4
  br label %bb6
bb5:
  br label %bb6
bb6:
  ret void
bb7:
  %v31 = extractvalue { ptr, i64 } %v14, 1
  %v32 = icmp ult i64 %v17, %v31
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb9, label %bb8
bb8:
  %v34 = extractvalue { ptr, i64 } %v14, 0
  %v35 = getelementptr inbounds float, ptr %v34, i64 %v17
  %v36 = insertvalue { ptr } undef, ptr %v35, 0
  %v37 = extractvalue { ptr } %v36, 0
  br label %bb11
bb9:
  br label %bb10
bb10:
  %v38 = inttoptr i64 0 to ptr
  %v39 = insertvalue { ptr } undef, ptr %v38, 0
  %v40 = extractvalue { ptr } %v39, 0
  br label %bb11
bb11:
  %v41 = phi ptr [ %v37, %bb8 ], [ %v40, %bb10 ]
  %v42 = insertvalue { ptr } undef, ptr %v41, 0
  %v43 = extractvalue { ptr } %v42, 0
  %v44 = ptrtoint ptr %v43 to i64
  %v45 = sub i64 %v44, 0
  %v46 = icmp ule i64 %v45, 0
  %v47 = add i64 %v45, 0
  %v48 = select i1 %v46, i64 %v47, i64 1
  %v49 = icmp eq i64 %v48, 1
  br i1 %v49, label %bb2, label %bb12
bb12:
  %v50 = icmp eq i64 %v48, 0
  br i1 %v50, label %bb5, label %bb13
bb13:
  unreachable
bb14:
  call void @llvm.trap() #0
  unreachable
bb15:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_q4k_row(ptr %v0, i64 %v1, i32 %v2, i32 %v3, i32 %v4, ptr %v5, i64 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v5, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v6, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi i32 [ %v2, %entry ]
  %v13 = phi i32 [ %v3, %entry ]
  %v14 = phi i32 [ %v4, %entry ]
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = alloca {  }, align 1
  %v17 = alloca [2 x i8], align 1
  %v18 = alloca [2 x i8], align 1
  %v19 = alloca [8 x i8], align 1
  %v20 = alloca [8 x i8], align 1
  %v21 = bitcast ptr %v16 to ptr
  %v22 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v21) #0
  br label %bb1
bb1:
  %v23 = zext i32 %v13 to i64
  %v24 = icmp uge i64 %v22, %v23
  %v25 = xor i1 %v24, 1
  br i1 %v25, label %bb3, label %bb2
bb2:
  br label %bb32
bb3:
  %v26 = mul i32 %v14, 144
  %v27 = zext i32 %v26 to i64
  %v28 = zext i32 %v12 to i64
  %v29 = mul i64 %v28, %v27
  %v30 = udiv i64 %v22, 256
  %v31 = urem i64 %v22, 256
  %v32 = mul i64 %v30, 144
  %v33 = add i64 %v29, %v32
  %v34 = extractvalue { ptr, i64 } %v11, 1
  %v35 = icmp ult i64 %v33, %v34
  br i1 %v35, label %bb4, label %bb40
bb4:
  %v36 = extractvalue { ptr, i64 } %v11, 0
  %v37 = getelementptr inbounds i8, ptr %v36, i64 %v33
  %v38 = load i8, ptr %v37, align 1
  %v39 = add i64 %v33, 1
  %v40 = icmp ult i64 %v39, %v34
  br i1 %v40, label %bb5, label %bb41
bb5:
  %v41 = extractvalue { ptr, i64 } %v11, 0
  %v42 = getelementptr inbounds i8, ptr %v41, i64 %v39
  %v43 = load i8, ptr %v42, align 1
  %v44 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 0
  store i8 %v38, ptr %v44, align 1
  %v45 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 1
  store i8 %v43, ptr %v45, align 1
  %v46 = load [2 x i8], ptr %v17, align 1
  %v47 = alloca [2 x i8], align 2
  store [2 x i8] %v46, ptr %v47, align 2
  %v48 = load i16, ptr %v47, align 2
  %v49 = add i64 %v33, 2
  %v50 = icmp ult i64 %v49, %v34
  br i1 %v50, label %bb6, label %bb42
bb6:
  %v51 = extractvalue { ptr, i64 } %v11, 0
  %v52 = getelementptr inbounds i8, ptr %v51, i64 %v49
  %v53 = load i8, ptr %v52, align 1
  %v54 = add i64 %v33, 3
  %v55 = icmp ult i64 %v54, %v34
  br i1 %v55, label %bb7, label %bb43
bb7:
  %v56 = extractvalue { ptr, i64 } %v11, 0
  %v57 = getelementptr inbounds i8, ptr %v56, i64 %v54
  %v58 = load i8, ptr %v57, align 1
  %v59 = getelementptr inbounds [2 x i8], ptr %v18, i32 0, i64 0
  store i8 %v53, ptr %v59, align 1
  %v60 = getelementptr inbounds [2 x i8], ptr %v18, i32 0, i64 1
  store i8 %v58, ptr %v60, align 1
  %v61 = load [2 x i8], ptr %v18, align 1
  %v62 = alloca [2 x i8], align 2
  store [2 x i8] %v61, ptr %v62, align 2
  %v63 = load i16, ptr %v62, align 2
  %v64 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v48) #0
  br label %bb8
bb8:
  %v65 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v63) #0
  br label %bb9
bb9:
  %v66 = add i64 %v33, 4
  %v67 = icmp ult i64 %v66, %v34
  br i1 %v67, label %bb10, label %bb44
bb10:
  %v68 = extractvalue { ptr, i64 } %v11, 0
  %v69 = getelementptr inbounds i8, ptr %v68, i64 %v66
  %v70 = load i8, ptr %v69, align 1
  %v71 = add i64 %v33, 5
  %v72 = icmp ult i64 %v71, %v34
  br i1 %v72, label %bb11, label %bb45
bb11:
  %v73 = extractvalue { ptr, i64 } %v11, 0
  %v74 = getelementptr inbounds i8, ptr %v73, i64 %v71
  %v75 = load i8, ptr %v74, align 1
  %v76 = add i64 %v33, 6
  %v77 = icmp ult i64 %v76, %v34
  br i1 %v77, label %bb12, label %bb46
bb12:
  %v78 = extractvalue { ptr, i64 } %v11, 0
  %v79 = getelementptr inbounds i8, ptr %v78, i64 %v76
  %v80 = load i8, ptr %v79, align 1
  %v81 = add i64 %v33, 7
  %v82 = icmp ult i64 %v81, %v34
  br i1 %v82, label %bb13, label %bb47
bb13:
  %v83 = extractvalue { ptr, i64 } %v11, 0
  %v84 = getelementptr inbounds i8, ptr %v83, i64 %v81
  %v85 = load i8, ptr %v84, align 1
  %v86 = add i64 %v33, 8
  %v87 = icmp ult i64 %v86, %v34
  br i1 %v87, label %bb14, label %bb48
bb14:
  %v88 = extractvalue { ptr, i64 } %v11, 0
  %v89 = getelementptr inbounds i8, ptr %v88, i64 %v86
  %v90 = load i8, ptr %v89, align 1
  %v91 = add i64 %v33, 9
  %v92 = icmp ult i64 %v91, %v34
  br i1 %v92, label %bb15, label %bb49
bb15:
  %v93 = extractvalue { ptr, i64 } %v11, 0
  %v94 = getelementptr inbounds i8, ptr %v93, i64 %v91
  %v95 = load i8, ptr %v94, align 1
  %v96 = add i64 %v33, 10
  %v97 = icmp ult i64 %v96, %v34
  br i1 %v97, label %bb16, label %bb50
bb16:
  %v98 = extractvalue { ptr, i64 } %v11, 0
  %v99 = getelementptr inbounds i8, ptr %v98, i64 %v96
  %v100 = load i8, ptr %v99, align 1
  %v101 = add i64 %v33, 11
  %v102 = icmp ult i64 %v101, %v34
  br i1 %v102, label %bb17, label %bb51
bb17:
  %v103 = extractvalue { ptr, i64 } %v11, 0
  %v104 = getelementptr inbounds i8, ptr %v103, i64 %v101
  %v105 = load i8, ptr %v104, align 1
  %v106 = add i64 %v33, 12
  %v107 = icmp ult i64 %v106, %v34
  br i1 %v107, label %bb18, label %bb52
bb18:
  %v108 = extractvalue { ptr, i64 } %v11, 0
  %v109 = getelementptr inbounds i8, ptr %v108, i64 %v106
  %v110 = load i8, ptr %v109, align 1
  %v111 = add i64 %v33, 13
  %v112 = icmp ult i64 %v111, %v34
  br i1 %v112, label %bb19, label %bb53
bb19:
  %v113 = extractvalue { ptr, i64 } %v11, 0
  %v114 = getelementptr inbounds i8, ptr %v113, i64 %v111
  %v115 = load i8, ptr %v114, align 1
  %v116 = add i64 %v33, 14
  %v117 = icmp ult i64 %v116, %v34
  br i1 %v117, label %bb20, label %bb54
bb20:
  %v118 = extractvalue { ptr, i64 } %v11, 0
  %v119 = getelementptr inbounds i8, ptr %v118, i64 %v116
  %v120 = load i8, ptr %v119, align 1
  %v121 = add i64 %v33, 15
  %v122 = icmp ult i64 %v121, %v34
  br i1 %v122, label %bb21, label %bb55
bb21:
  %v123 = extractvalue { ptr, i64 } %v11, 0
  %v124 = getelementptr inbounds i8, ptr %v123, i64 %v121
  %v125 = load i8, ptr %v124, align 1
  %v126 = call { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v70, i8 %v75, i8 %v80, i8 %v85, i8 %v90, i8 %v95, i8 %v100, i8 %v105, i8 %v110, i8 %v115, i8 %v120, i8 %v125) #0
  br label %bb22
bb22:
  %v127 = extractvalue { [8 x i8], [8 x i8] } %v126, 0
  store [8 x i8] %v127, ptr %v19, align 1
  %v128 = extractvalue { [8 x i8], [8 x i8] } %v126, 1
  store [8 x i8] %v128, ptr %v20, align 1
  %v129 = udiv i64 %v31, 32
  %v130 = urem i64 %v31, 32
  %v131 = icmp ult i64 %v129, 8
  br i1 %v131, label %bb23, label %bb56
bb23:
  %v132 = getelementptr inbounds [8 x i8], ptr %v19, i32 0, i64 %v129
  %v133 = load i8, ptr %v132, align 1
  %v134 = uitofp i8 %v133 to float
  %v135 = getelementptr inbounds [8 x i8], ptr %v20, i32 0, i64 %v129
  %v136 = load i8, ptr %v135, align 1
  %v137 = uitofp i8 %v136 to float
  %v138 = add i64 %v33, 16
  %v139 = udiv i64 %v31, 64
  %v140 = urem i64 %v31, 64
  %v141 = mul i64 %v139, 32
  %v142 = add i64 %v138, %v141
  %v143 = icmp ult i64 %v140, 32
  %v144 = xor i1 %v143, 1
  br i1 %v144, label %bb26, label %bb24
bb24:
  %v145 = add i64 %v142, %v130
  %v146 = icmp ult i64 %v145, %v34
  br i1 %v146, label %bb25, label %bb57
bb25:
  %v147 = extractvalue { ptr, i64 } %v11, 0
  %v148 = getelementptr inbounds i8, ptr %v147, i64 %v145
  %v149 = load i8, ptr %v148, align 1
  %v150 = and i8 %v149, 15
  %v151 = uitofp i8 %v150 to float
  br label %bb28
bb26:
  %v152 = add i64 %v142, %v130
  %v153 = icmp ult i64 %v152, %v34
  br i1 %v153, label %bb27, label %bb58
bb27:
  %v154 = extractvalue { ptr, i64 } %v11, 0
  %v155 = getelementptr inbounds i8, ptr %v154, i64 %v152
  %v156 = load i8, ptr %v155, align 1
  %v157 = trunc i32 4 to i8
  %v158 = and i8 %v157, 7
  %v159 = lshr i8 %v156, %v158
  %v160 = uitofp i8 %v159 to float
  br label %bb28
bb28:
  %v161 = phi float [ %v151, %bb25 ], [ %v160, %bb27 ]
  %v162 = icmp eq i64 %v22, 18446744073709551615
  br i1 %v162, label %bb36, label %bb33
bb29:
  %v163 = extractvalue { ptr } %v179, 0
  %v164 = fmul contract float %v64, %v134
  %v165 = fmul contract float %v164, %v161
  %v166 = fmul contract float %v65, %v137
  %v167 = fsub contract float %v165, %v166
  store float %v167, ptr %v163, align 4
  br label %bb31
bb30:
  br label %bb31
bb31:
  br label %bb32
bb32:
  ret void
bb33:
  %v168 = extractvalue { ptr, i64 } %v15, 1
  %v169 = icmp ult i64 %v22, %v168
  %v170 = xor i1 %v169, 1
  br i1 %v170, label %bb35, label %bb34
bb34:
  %v171 = extractvalue { ptr, i64 } %v15, 0
  %v172 = getelementptr inbounds float, ptr %v171, i64 %v22
  %v173 = insertvalue { ptr } undef, ptr %v172, 0
  %v174 = extractvalue { ptr } %v173, 0
  br label %bb37
bb35:
  br label %bb36
bb36:
  %v175 = inttoptr i64 0 to ptr
  %v176 = insertvalue { ptr } undef, ptr %v175, 0
  %v177 = extractvalue { ptr } %v176, 0
  br label %bb37
bb37:
  %v178 = phi ptr [ %v174, %bb34 ], [ %v177, %bb36 ]
  %v179 = insertvalue { ptr } undef, ptr %v178, 0
  %v180 = extractvalue { ptr } %v179, 0
  %v181 = ptrtoint ptr %v180 to i64
  %v182 = sub i64 %v181, 0
  %v183 = icmp ule i64 %v182, 0
  %v184 = add i64 %v182, 0
  %v185 = select i1 %v183, i64 %v184, i64 1
  %v186 = icmp eq i64 %v185, 1
  br i1 %v186, label %bb29, label %bb38
bb38:
  %v187 = icmp eq i64 %v185, 0
  br i1 %v187, label %bb30, label %bb39
bb39:
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
bb57:
  call void @llvm.trap() #0
  unreachable
bb58:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q8_0_gemm_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca [2 x i8], align 1
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v23 = zext i32 %v22 to i64
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v25 = zext i32 %v24 to i64
  %v26 = zext i32 %v19 to i64
  %v27 = zext i32 %v17 to i64
  %v28 = mul i64 %v26, %v27
  %v29 = icmp uge i64 %v25, %v28
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb4, label %bb3
bb3:
  br label %bb31
bb4:
  %v31 = icmp eq i64 %v27, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb5, label %bb32
bb5:
  %v33 = urem i64 %v25, %v27
  %v34 = udiv i64 %v25, %v27
  %v35 = zext i32 %v18 to i64
  %v36 = mul i64 %v35, 34
  br label %bb6
bb6:
  %v37 = phi float [ 0.0, %bb5 ], [ %v63, %bb15 ]
  %v38 = phi i64 [ %v23, %bb5 ], [ %v85, %bb15 ]
  %v39 = icmp ult i64 %v38, %v35
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb16, label %bb7
bb7:
  %v41 = mul i64 %v33, %v36
  %v42 = mul i64 %v38, 34
  %v43 = add i64 %v41, %v42
  %v44 = extractvalue { ptr, i64 } %v15, 1
  %v45 = icmp ult i64 %v43, %v44
  br i1 %v45, label %bb8, label %bb33
bb8:
  %v46 = extractvalue { ptr, i64 } %v15, 0
  %v47 = getelementptr inbounds i8, ptr %v46, i64 %v43
  %v48 = load i8, ptr %v47, align 1
  %v49 = add i64 %v43, 1
  %v50 = icmp ult i64 %v49, %v44
  br i1 %v50, label %bb9, label %bb34
bb9:
  %v51 = extractvalue { ptr, i64 } %v15, 0
  %v52 = getelementptr inbounds i8, ptr %v51, i64 %v49
  %v53 = load i8, ptr %v52, align 1
  %v54 = getelementptr inbounds [2 x i8], ptr %v21, i32 0, i64 0
  store i8 %v48, ptr %v54, align 1
  %v55 = getelementptr inbounds [2 x i8], ptr %v21, i32 0, i64 1
  store i8 %v53, ptr %v55, align 1
  %v56 = load [2 x i8], ptr %v21, align 1
  %v57 = alloca [2 x i8], align 2
  store [2 x i8] %v56, ptr %v57, align 2
  %v58 = load i16, ptr %v57, align 2
  %v59 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v58) #0
  br label %bb10
bb10:
  %v60 = mul i64 %v34, %v35
  %v61 = add i64 %v60, %v38
  %v62 = mul i64 %v61, 32
  br label %bb11
bb11:
  %v63 = phi float [ %v37, %bb10 ], [ %v83, %bb14 ]
  %v64 = phi i64 [ 0, %bb10 ], [ %v84, %bb14 ]
  %v65 = icmp ult i64 %v64, 32
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb15, label %bb12
bb12:
  %v67 = add i64 %v43, 2
  %v68 = add i64 %v67, %v64
  %v69 = icmp ult i64 %v68, %v44
  br i1 %v69, label %bb13, label %bb35
bb13:
  %v70 = extractvalue { ptr, i64 } %v15, 0
  %v71 = getelementptr inbounds i8, ptr %v70, i64 %v68
  %v72 = load i8, ptr %v71, align 1
  %v73 = bitcast i8 %v72 to i8
  %v74 = sitofp i8 %v73 to float
  %v75 = fmul contract float %v59, %v74
  %v76 = add i64 %v62, %v64
  %v77 = extractvalue { ptr, i64 } %v16, 1
  %v78 = icmp ult i64 %v76, %v77
  br i1 %v78, label %bb14, label %bb36
bb14:
  %v79 = extractvalue { ptr, i64 } %v16, 0
  %v80 = getelementptr inbounds float, ptr %v79, i64 %v76
  %v81 = load float, ptr %v80, align 4
  %v82 = fmul contract float %v75, %v81
  %v83 = fadd contract float %v63, %v82
  %v84 = add i64 %v64, 1
  br label %bb11
bb15:
  %v85 = add i64 %v38, 32
  br label %bb6
bb16:
  %v86 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_2, i64 %v23
  br label %bb17
bb17:
  store float %v37, ptr addrspace(3) %v86, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb18
bb18:
  br label %bb19
bb19:
  %v88 = phi i64 [ 16, %bb18 ], [ %v101, %bb26 ]
  %v89 = icmp ugt i64 %v88, 0
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb27, label %bb20
bb20:
  %v91 = icmp ult i64 %v23, %v88
  %v92 = xor i1 %v91, 1
  br i1 %v92, label %bb24, label %bb21
bb21:
  %v93 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v94 = add i64 %v23, %v88
  %v95 = getelementptr inbounds float, ptr addrspace(3) %v93, i64 %v94
  br label %bb22
bb22:
  %v96 = load float, ptr addrspace(3) %v95, align 4
  %v97 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_2, i64 %v23
  br label %bb23
bb23:
  %v98 = load float, ptr addrspace(3) %v97, align 4
  %v99 = fadd contract float %v98, %v96
  store float %v99, ptr addrspace(3) %v97, align 4
  br label %bb25
bb24:
  br label %bb25
bb25:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb26
bb26:
  %v101 = udiv i64 %v88, 2
  br label %bb19
bb27:
  %v102 = icmp eq i64 %v23, 0
  br i1 %v102, label %bb28, label %bb30
bb28:
  %v103 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v104 = getelementptr inbounds float, ptr addrspace(3) %v103, i64 0
  br label %bb29
bb29:
  %v105 = load float, ptr addrspace(3) %v104, align 4
  %v106 = extractvalue { ptr, i64 } %v20, 0
  %v107 = getelementptr inbounds float, ptr %v106, i64 %v25
  store float %v105, ptr %v107, align 4
  br label %bb30
bb30:
  br label %bb31
bb31:
  ret void
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q6k_gemm_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca [2 x i8], align 1
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v23 = zext i32 %v22 to i64
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v25 = zext i32 %v24 to i64
  %v26 = zext i32 %v17 to i64
  %v27 = zext i32 %v19 to i64
  %v28 = mul i64 %v26, %v27
  %v29 = icmp uge i64 %v25, %v28
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb4, label %bb3
bb3:
  br label %bb40
bb4:
  %v31 = icmp eq i64 %v26, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb5, label %bb41
bb5:
  %v33 = urem i64 %v25, %v26
  %v34 = udiv i64 %v25, %v26
  %v35 = zext i32 %v18 to i64
  %v36 = mul i64 %v35, 210
  br label %bb6
bb6:
  %v37 = phi float [ 0.0, %bb5 ], [ %v64, %bb24 ]
  %v38 = phi i64 [ 0, %bb5 ], [ %v209, %bb24 ]
  %v39 = icmp ult i64 %v38, %v35
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb25, label %bb7
bb7:
  %v41 = mul i64 %v33, %v36
  %v42 = mul i64 %v38, 210
  %v43 = add i64 %v41, %v42
  %v44 = add i64 %v43, 208
  %v45 = extractvalue { ptr, i64 } %v15, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb8, label %bb42
bb8:
  %v47 = extractvalue { ptr, i64 } %v15, 0
  %v48 = getelementptr inbounds i8, ptr %v47, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v50 = add i64 %v43, 209
  %v51 = icmp ult i64 %v50, %v45
  br i1 %v51, label %bb9, label %bb43
bb9:
  %v52 = extractvalue { ptr, i64 } %v15, 0
  %v53 = getelementptr inbounds i8, ptr %v52, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v55 = getelementptr inbounds [2 x i8], ptr %v21, i32 0, i64 0
  store i8 %v49, ptr %v55, align 1
  %v56 = getelementptr inbounds [2 x i8], ptr %v21, i32 0, i64 1
  store i8 %v54, ptr %v56, align 1
  %v57 = load [2 x i8], ptr %v21, align 1
  %v58 = alloca [2 x i8], align 2
  store [2 x i8] %v57, ptr %v58, align 2
  %v59 = load i16, ptr %v58, align 2
  %v60 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v59) #0
  br label %bb10
bb10:
  %v61 = mul i64 %v34, %v35
  %v62 = add i64 %v61, %v38
  %v63 = mul i64 %v62, 256
  br label %bb11
bb11:
  %v64 = phi float [ %v37, %bb10 ], [ %v207, %bb23 ]
  %v65 = phi i64 [ 0, %bb10 ], [ %v208, %bb23 ]
  %v66 = icmp ult i64 %v65, 2
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb24, label %bb12
bb12:
  %v68 = mul i64 %v65, 64
  %v69 = add i64 %v43, %v68
  %v70 = add i64 %v43, 128
  %v71 = mul i64 %v65, 32
  %v72 = add i64 %v70, %v71
  %v73 = add i64 %v43, 192
  %v74 = mul i64 %v65, 8
  %v75 = add i64 %v73, %v74
  %v76 = mul i64 %v65, 128
  %v77 = add i64 %v63, %v76
  %v78 = udiv i64 %v23, 16
  %v79 = add i64 %v69, %v23
  %v80 = icmp ult i64 %v79, %v45
  br i1 %v80, label %bb13, label %bb44
bb13:
  %v81 = extractvalue { ptr, i64 } %v15, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = and i8 %v83, 15
  %v85 = zext i8 %v84 to i32
  %v86 = add i64 %v72, %v23
  %v87 = icmp ult i64 %v86, %v45
  br i1 %v87, label %bb14, label %bb45
bb14:
  %v88 = extractvalue { ptr, i64 } %v15, 0
  %v89 = getelementptr inbounds i8, ptr %v88, i64 %v86
  %v90 = load i8, ptr %v89, align 1
  %v91 = and i8 %v90, 3
  %v92 = zext i8 %v91 to i32
  %v93 = and i32 4, 31
  %v94 = shl i32 %v92, %v93
  %v95 = or i32 %v85, %v94
  %v96 = sub i32 %v95, 32
  %v97 = add i64 %v79, 32
  %v98 = icmp ult i64 %v97, %v45
  br i1 %v98, label %bb15, label %bb46
bb15:
  %v99 = extractvalue { ptr, i64 } %v15, 0
  %v100 = getelementptr inbounds i8, ptr %v99, i64 %v97
  %v101 = load i8, ptr %v100, align 1
  %v102 = and i8 %v101, 15
  %v103 = zext i8 %v102 to i32
  %v104 = trunc i32 2 to i8
  %v105 = and i8 %v104, 7
  %v106 = lshr i8 %v90, %v105
  %v107 = and i8 %v106, 3
  %v108 = zext i8 %v107 to i32
  %v109 = and i32 4, 31
  %v110 = shl i32 %v108, %v109
  %v111 = or i32 %v103, %v110
  %v112 = sub i32 %v111, 32
  %v113 = trunc i32 4 to i8
  %v114 = and i8 %v113, 7
  %v115 = lshr i8 %v83, %v114
  %v116 = zext i8 %v115 to i32
  %v117 = trunc i32 4 to i8
  %v118 = and i8 %v117, 7
  %v119 = lshr i8 %v90, %v118
  %v120 = and i8 %v119, 3
  %v121 = zext i8 %v120 to i32
  %v122 = and i32 4, 31
  %v123 = shl i32 %v121, %v122
  %v124 = or i32 %v116, %v123
  %v125 = sub i32 %v124, 32
  %v126 = trunc i32 4 to i8
  %v127 = and i8 %v126, 7
  %v128 = lshr i8 %v101, %v127
  %v129 = zext i8 %v128 to i32
  %v130 = trunc i32 6 to i8
  %v131 = and i8 %v130, 7
  %v132 = lshr i8 %v90, %v131
  %v133 = and i8 %v132, 3
  %v134 = zext i8 %v133 to i32
  %v135 = and i32 4, 31
  %v136 = shl i32 %v134, %v135
  %v137 = or i32 %v129, %v136
  %v138 = sub i32 %v137, 32
  %v139 = add i64 %v75, %v78
  %v140 = icmp ult i64 %v139, %v45
  br i1 %v140, label %bb16, label %bb47
bb16:
  %v141 = extractvalue { ptr, i64 } %v15, 0
  %v142 = getelementptr inbounds i8, ptr %v141, i64 %v139
  %v143 = load i8, ptr %v142, align 1
  %v144 = bitcast i8 %v143 to i8
  %v145 = sitofp i8 %v144 to float
  %v146 = add i64 %v139, 2
  %v147 = icmp ult i64 %v146, %v45
  br i1 %v147, label %bb17, label %bb48
bb17:
  %v148 = extractvalue { ptr, i64 } %v15, 0
  %v149 = getelementptr inbounds i8, ptr %v148, i64 %v146
  %v150 = load i8, ptr %v149, align 1
  %v151 = bitcast i8 %v150 to i8
  %v152 = sitofp i8 %v151 to float
  %v153 = add i64 %v139, 4
  %v154 = icmp ult i64 %v153, %v45
  br i1 %v154, label %bb18, label %bb49
bb18:
  %v155 = extractvalue { ptr, i64 } %v15, 0
  %v156 = getelementptr inbounds i8, ptr %v155, i64 %v153
  %v157 = load i8, ptr %v156, align 1
  %v158 = bitcast i8 %v157 to i8
  %v159 = sitofp i8 %v158 to float
  %v160 = add i64 %v139, 6
  %v161 = icmp ult i64 %v160, %v45
  br i1 %v161, label %bb19, label %bb50
bb19:
  %v162 = extractvalue { ptr, i64 } %v15, 0
  %v163 = getelementptr inbounds i8, ptr %v162, i64 %v160
  %v164 = load i8, ptr %v163, align 1
  %v165 = bitcast i8 %v164 to i8
  %v166 = sitofp i8 %v165 to float
  %v167 = fmul contract float %v60, %v145
  %v168 = sitofp i32 %v96 to float
  %v169 = fmul contract float %v167, %v168
  %v170 = add i64 %v77, %v23
  %v171 = extractvalue { ptr, i64 } %v16, 1
  %v172 = icmp ult i64 %v170, %v171
  br i1 %v172, label %bb20, label %bb51
bb20:
  %v173 = extractvalue { ptr, i64 } %v16, 0
  %v174 = getelementptr inbounds float, ptr %v173, i64 %v170
  %v175 = load float, ptr %v174, align 4
  %v176 = fmul contract float %v169, %v175
  %v177 = fadd contract float %v64, %v176
  %v178 = fmul contract float %v60, %v152
  %v179 = sitofp i32 %v112 to float
  %v180 = fmul contract float %v178, %v179
  %v181 = add i64 %v170, 32
  %v182 = icmp ult i64 %v181, %v171
  br i1 %v182, label %bb21, label %bb52
bb21:
  %v183 = extractvalue { ptr, i64 } %v16, 0
  %v184 = getelementptr inbounds float, ptr %v183, i64 %v181
  %v185 = load float, ptr %v184, align 4
  %v186 = fmul contract float %v180, %v185
  %v187 = fadd contract float %v177, %v186
  %v188 = fmul contract float %v60, %v159
  %v189 = sitofp i32 %v125 to float
  %v190 = fmul contract float %v188, %v189
  %v191 = add i64 %v170, 64
  %v192 = icmp ult i64 %v191, %v171
  br i1 %v192, label %bb22, label %bb53
bb22:
  %v193 = extractvalue { ptr, i64 } %v16, 0
  %v194 = getelementptr inbounds float, ptr %v193, i64 %v191
  %v195 = load float, ptr %v194, align 4
  %v196 = fmul contract float %v190, %v195
  %v197 = fadd contract float %v187, %v196
  %v198 = fmul contract float %v60, %v166
  %v199 = sitofp i32 %v138 to float
  %v200 = fmul contract float %v198, %v199
  %v201 = add i64 %v170, 96
  %v202 = icmp ult i64 %v201, %v171
  br i1 %v202, label %bb23, label %bb54
bb23:
  %v203 = extractvalue { ptr, i64 } %v16, 0
  %v204 = getelementptr inbounds float, ptr %v203, i64 %v201
  %v205 = load float, ptr %v204, align 4
  %v206 = fmul contract float %v200, %v205
  %v207 = fadd contract float %v197, %v206
  %v208 = add i64 %v65, 1
  br label %bb11
bb24:
  %v209 = add i64 %v38, 1
  br label %bb6
bb25:
  %v210 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_3, i64 %v23
  br label %bb26
bb26:
  store float %v37, ptr addrspace(3) %v210, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb27
bb27:
  br label %bb28
bb28:
  %v212 = phi i64 [ 16, %bb27 ], [ %v225, %bb35 ]
  %v213 = icmp ugt i64 %v212, 0
  %v214 = xor i1 %v213, 1
  br i1 %v214, label %bb36, label %bb29
bb29:
  %v215 = icmp ult i64 %v23, %v212
  %v216 = xor i1 %v215, 1
  br i1 %v216, label %bb33, label %bb30
bb30:
  %v217 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v218 = add i64 %v23, %v212
  %v219 = getelementptr inbounds float, ptr addrspace(3) %v217, i64 %v218
  br label %bb31
bb31:
  %v220 = load float, ptr addrspace(3) %v219, align 4
  %v221 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_3, i64 %v23
  br label %bb32
bb32:
  %v222 = load float, ptr addrspace(3) %v221, align 4
  %v223 = fadd contract float %v222, %v220
  store float %v223, ptr addrspace(3) %v221, align 4
  br label %bb34
bb33:
  br label %bb34
bb34:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb35
bb35:
  %v225 = udiv i64 %v212, 2
  br label %bb28
bb36:
  %v226 = icmp eq i64 %v23, 0
  br i1 %v226, label %bb37, label %bb39
bb37:
  %v227 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v228 = getelementptr inbounds float, ptr addrspace(3) %v227, i64 0
  br label %bb38
bb38:
  %v229 = load float, ptr addrspace(3) %v228, align 4
  %v230 = extractvalue { ptr, i64 } %v20, 0
  %v231 = getelementptr inbounds float, ptr %v230, i64 %v25
  store float %v229, ptr %v231, align 4
  br label %bb39
bb39:
  br label %bb40
bb40:
  ret void
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
}

declare float @llvm.fabs.f32(float)
declare float @llvm.round.f32(float)
declare i8 @llvm.fptosi.sat.i8.f32(float)

define ptx_kernel void @quantize_q8_32(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v16 = zext i32 %v15 to i64
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v18 = zext i32 %v17 to i64
  %v19 = mul i64 %v18, 32
  %v20 = add i64 %v19, %v16
  %v21 = extractvalue { ptr, i64 } %v12, 1
  %v22 = icmp uge i64 %v20, %v21
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb4, label %bb3
bb3:
  br label %bb25
bb4:
  %v24 = icmp ult i64 %v20, %v21
  br i1 %v24, label %bb5, label %bb29
bb5:
  %v25 = extractvalue { ptr, i64 } %v12, 0
  %v26 = getelementptr inbounds float, ptr %v25, i64 %v20
  %v27 = load float, ptr %v26, align 4
  %v28 = call float @llvm.fabs.f32(float %v27) #0
  br label %bb26
bb6:
  store float %v28, ptr addrspace(3) %v63, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb7
bb7:
  br label %bb8
bb8:
  %v30 = phi i64 [ 16, %bb7 ], [ %v47, %bb16 ]
  %v31 = icmp ugt i64 %v30, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb17, label %bb9
bb9:
  %v33 = icmp ult i64 %v16, %v30
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb14, label %bb10
bb10:
  %v35 = bitcast ptr addrspace(3) @__shared_mem_4 to ptr addrspace(3)
  %v36 = getelementptr inbounds float, ptr addrspace(3) %v35, i64 %v16
  br label %bb11
bb11:
  %v37 = load float, ptr addrspace(3) %v36, align 4
  %v38 = bitcast ptr addrspace(3) @__shared_mem_4 to ptr addrspace(3)
  %v39 = add i64 %v16, %v30
  %v40 = getelementptr inbounds float, ptr addrspace(3) %v38, i64 %v39
  br label %bb12
bb12:
  %v41 = load float, ptr addrspace(3) %v40, align 4
  %v42 = fcmp uno float %v37, %v37
  %v43 = fcmp oge float %v41, %v37
  %v44 = select i1 %v43, float %v41, float %v37
  %v45 = select i1 %v42, float %v41, float %v44
  br label %bb27
bb13:
  store float %v45, ptr addrspace(3) %v64, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb16
bb16:
  %v47 = udiv i64 %v30, 2
  br label %bb8
bb17:
  %v48 = bitcast ptr addrspace(3) @__shared_mem_4 to ptr addrspace(3)
  %v49 = getelementptr inbounds float, ptr addrspace(3) %v48, i64 0
  br label %bb18
bb18:
  %v50 = load float, ptr addrspace(3) %v49, align 4
  %v51 = fdiv contract float %v50, 127.0
  %v52 = fcmp ogt float %v51, 0.0
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb21, label %bb19
bb19:
  %v54 = fdiv contract float %v27, %v51
  %v55 = call float @llvm.round.f32(float %v54) #0
  br label %bb28
bb20:
  %v56 = call i8 @llvm.fptosi.sat.i8.f32(float %v65) #0
  br label %bb22
bb21:
  br label %bb22
bb22:
  %v57 = phi i8 [ %v56, %bb20 ], [ 0, %bb21 ]
  %v58 = extractvalue { ptr, i64 } %v13, 0
  %v59 = getelementptr inbounds i8, ptr %v58, i64 %v20
  store i8 %v57, ptr %v59, align 1
  %v60 = icmp eq i64 %v16, 0
  br i1 %v60, label %bb23, label %bb24
bb23:
  %v61 = extractvalue { ptr, i64 } %v14, 0
  %v62 = getelementptr inbounds float, ptr %v61, i64 %v18
  store float %v51, ptr %v62, align 4
  br label %bb24
bb24:
  br label %bb25
bb25:
  ret void
bb26:
  %v63 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_4, i64 %v16
  br label %bb6
bb27:
  %v64 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_4, i64 %v16
  br label %bb13
bb28:
  %v65 = call float @core__f32___impl_f32___clamp(float %v55, float -127.0, float 127.0) #0
  br label %bb20
bb29:
  call void @llvm.trap() #0
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define ptx_kernel void @q4k_q8_gemv_multiwarp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr %v12, i64 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v13, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi { ptr, i64 } [ %v23, %entry ]
  %v33 = alloca { [0 x i32], { { i32 } } }, align 4
  %v34 = alloca { [0 x i32], { { i32 } } }, align 4
  %v35 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v36 = zext i32 %v35 to i64
  %v37 = and i64 %v36, 31
  %v38 = zext i32 5 to i64
  %v39 = and i64 %v38, 63
  %v40 = lshr i64 %v36, %v39
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v42 = zext i32 %v41 to i64
  %v43 = zext i32 5 to i64
  %v44 = and i64 %v43, 63
  %v45 = lshr i64 %v42, %v44
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb3
bb3:
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v29 to i64
  %v49 = zext i32 %v31 to i64
  %v50 = mul i64 %v48, %v49
  %v51 = icmp uge i64 %v47, %v50
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb5, label %bb4
bb4:
  br label %bb6
bb5:
  %v53 = icmp uge i64 %v40, %v45
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb7, label %bb6
bb6:
  br label %bb37
bb7:
  %v55 = icmp eq i64 %v48, 0
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb8, label %bb40
bb8:
  %v57 = urem i64 %v47, %v48
  %v58 = udiv i64 %v47, %v48
  %v59 = zext i32 %v30 to i64
  %v60 = mul i64 %v59, 144
  %v61 = mul i64 %v58, %v59
  %v62 = mul i64 %v61, 256
  br label %bb9
bb9:
  %v63 = phi float [ 0.0, %bb8 ], [ %v70, %bb14 ]
  %v64 = phi i64 [ %v40, %bb8 ], [ %v121, %bb14 ]
  %v65 = icmp ult i64 %v64, %v59
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb15, label %bb10
bb10:
  %v67 = mul i64 %v57, %v60
  %v68 = mul i64 %v64, 144
  %v69 = add i64 %v67, %v68
  br label %bb11
bb11:
  %v70 = phi float [ %v63, %bb10 ], [ %v119, %bb13 ]
  %v71 = phi i64 [ 0, %bb10 ], [ %v120, %bb13 ]
  %v72 = icmp ult i64 %v71, 2
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb14, label %bb12
bb12:
  %v74 = mul i64 %v71, 128
  %v75 = mul i64 %v37, 4
  %v76 = add i64 %v74, %v75
  %v77 = mul i64 %v64, 256
  %v78 = add i64 %v62, %v77
  %v79 = add i64 %v78, %v76
  %v80 = extractvalue { ptr, i64 } %v25, 0
  %v81 = getelementptr inbounds i8, ptr %v80, i64 %v79
  store { [0 x i32], { { i32 } } } undef, ptr %v33, align 4
  %v82 = bitcast ptr %v81 to ptr
  %v83 = bitcast ptr %v33 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %v83, ptr %v82, i64 4, i1 0) #0
  %v85 = load { [0 x i32], { { i32 } } }, ptr %v33, align 4
  store { [0 x i32], { { i32 } } } %v85, ptr %v34, align 4
  %v86 = getelementptr inbounds i8, ptr %v34, i32 0
  %v87 = bitcast ptr %v86 to ptr
  %v88 = load i32, ptr %v87, align 4
  %v89 = trunc i32 %v88 to i8
  %v90 = bitcast i8 %v89 to i8
  %v91 = and i32 8, 31
  %v92 = lshr i32 %v88, %v91
  %v93 = trunc i32 %v92 to i8
  %v94 = bitcast i8 %v93 to i8
  %v95 = and i32 16, 31
  %v96 = lshr i32 %v88, %v95
  %v97 = trunc i32 %v96 to i8
  %v98 = bitcast i8 %v97 to i8
  %v99 = and i32 24, 31
  %v100 = lshr i32 %v88, %v99
  %v101 = trunc i32 %v100 to i8
  %v102 = bitcast i8 %v101 to i8
  %v103 = sext i8 %v90 to i32
  %v104 = sext i8 %v94 to i32
  %v105 = add i32 %v103, %v104
  %v106 = sext i8 %v98 to i32
  %v107 = add i32 %v105, %v106
  %v108 = sext i8 %v102 to i32
  %v109 = add i32 %v107, %v108
  %v110 = udiv i64 %v79, 32
  %v111 = extractvalue { ptr, i64 } %v26, 1
  %v112 = icmp ult i64 %v110, %v111
  %v113 = extractvalue { ptr, i64 } %v26, 0
  %v114 = getelementptr inbounds float, ptr %v113, i64 %v110
  %v115 = load float, ptr %v114, align 4
  %v116 = extractvalue { ptr, i64 } %v24, 0
  %v117 = extractvalue { ptr, i64 } %v24, 1
  %v118 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v116, i64 %v117, i64 %v69, i64 %v76, i32 %v88, i32 %v109, float %v115) #0
  br label %bb13
bb13:
  %v119 = fadd contract float %v70, %v118
  %v120 = add i64 %v71, 1
  br label %bb11
bb14:
  %v121 = add i64 %v64, %v45
  br label %bb9
bb15:
  br label %bb16
bb16:
  %v122 = phi float [ %v63, %bb15 ], [ %v154, %bb38 ]
  %v123 = phi i32 [ 16, %bb15 ], [ %v155, %bb38 ]
  %v124 = icmp ugt i32 %v123, 0
  %v125 = xor i1 %v124, 1
  br i1 %v125, label %bb18, label %bb17
bb17:
  %v126 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v122, i32 %v123, i32 31) #0
  br label %bb38
bb18:
  %v127 = icmp eq i64 %v37, 0
  %v128 = icmp eq i64 %v37, 0
  br i1 %v128, label %bb19, label %bb21
bb19:
  %v129 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_5, i64 %v40
  br label %bb20
bb20:
  store float %v122, ptr addrspace(3) %v129, align 4
  br label %bb21
bb21:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb22
bb22:
  %v131 = icmp eq i64 %v40, 0
  br i1 %v131, label %bb23, label %bb36
bb23:
  %v132 = icmp ult i64 %v37, %v45
  %v133 = xor i1 %v132, 1
  br i1 %v133, label %bb26, label %bb24
bb24:
  %v134 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v135 = getelementptr inbounds float, ptr addrspace(3) %v134, i64 %v37
  br label %bb25
bb25:
  %v136 = load float, ptr addrspace(3) %v135, align 4
  br label %bb27
bb26:
  br label %bb27
bb27:
  %v137 = phi float [ %v136, %bb25 ], [ 0.0, %bb26 ]
  br label %bb28
bb28:
  %v138 = phi float [ %v137, %bb27 ], [ %v156, %bb39 ]
  %v139 = phi i32 [ 16, %bb27 ], [ %v157, %bb39 ]
  %v140 = icmp ugt i32 %v139, 0
  %v141 = xor i1 %v140, 1
  br i1 %v141, label %bb30, label %bb29
bb29:
  %v142 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v138, i32 %v139, i32 31) #0
  br label %bb39
bb30:
  %v143 = xor i1 %v127, 1
  br i1 %v143, label %bb35, label %bb31
bb31:
  %v144 = icmp eq i32 %v28, 0
  br i1 %v144, label %bb33, label %bb32
bb32:
  %v145 = extractvalue { ptr, i64 } %v27, 1
  %v146 = icmp ult i64 %v47, %v145
  %v147 = extractvalue { ptr, i64 } %v27, 0
  %v148 = getelementptr inbounds float, ptr %v147, i64 %v47
  %v149 = load float, ptr %v148, align 4
  br label %bb34
bb33:
  br label %bb34
bb34:
  %v150 = phi float [ %v149, %bb32 ], [ 0.0, %bb33 ]
  %v151 = extractvalue { ptr, i64 } %v32, 0
  %v152 = getelementptr inbounds float, ptr %v151, i64 %v47
  %v153 = fadd contract float %v138, %v150
  store float %v153, ptr %v152, align 4
  br label %bb35
bb35:
  br label %bb36
bb36:
  br label %bb37
bb37:
  ret void
bb38:
  %v154 = fadd contract float %v122, %v126
  %v155 = udiv i32 %v123, 2
  br label %bb16
bb39:
  %v156 = fadd contract float %v138, %v142
  %v157 = udiv i32 %v139, 2
  br label %bb28
bb40:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_f32_rows(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, ptr %v6, i64 %v7) #0 {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v7, 1
  br label %bb0
bb0:
  %v14 = phi { ptr, i64 } [ %v9, %entry ]
  %v15 = phi { ptr, i64 } [ %v11, %entry ]
  %v16 = phi i32 [ %v4, %entry ]
  %v17 = phi i32 [ %v5, %entry ]
  %v18 = phi { ptr, i64 } [ %v13, %entry ]
  %v19 = alloca {  }, align 1
  %v20 = bitcast ptr %v19 to ptr
  %v21 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v20) #0
  br label %bb1
bb1:
  %v22 = zext i32 %v16 to i64
  %v23 = zext i32 %v17 to i64
  %v24 = mul i64 %v22, %v23
  %v25 = icmp uge i64 %v21, %v24
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb3, label %bb2
bb2:
  br label %bb10
bb3:
  %v27 = icmp eq i64 %v23, 0
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb4, label %bb18
bb4:
  %v29 = udiv i64 %v21, %v23
  %v30 = urem i64 %v21, %v23
  %v31 = icmp eq i64 %v21, 18446744073709551615
  br i1 %v31, label %bb14, label %bb11
bb5:
  %v32 = extractvalue { ptr } %v57, 0
  %v33 = extractvalue { ptr, i64 } %v15, 1
  %v34 = icmp ult i64 %v29, %v33
  br i1 %v34, label %bb6, label %bb19
bb6:
  %v35 = extractvalue { ptr, i64 } %v15, 0
  %v36 = getelementptr inbounds i32, ptr %v35, i64 %v29
  %v37 = load i32, ptr %v36, align 4
  %v38 = zext i32 %v37 to i64
  %v39 = mul i64 %v38, %v23
  %v40 = add i64 %v39, %v30
  %v41 = extractvalue { ptr, i64 } %v14, 1
  %v42 = icmp ult i64 %v40, %v41
  br i1 %v42, label %bb7, label %bb20
bb7:
  %v43 = extractvalue { ptr, i64 } %v14, 0
  %v44 = getelementptr inbounds float, ptr %v43, i64 %v40
  %v45 = load float, ptr %v44, align 4
  store float %v45, ptr %v32, align 4
  br label %bb9
bb8:
  br label %bb9
bb9:
  br label %bb10
bb10:
  ret void
bb11:
  %v46 = extractvalue { ptr, i64 } %v18, 1
  %v47 = icmp ult i64 %v21, %v46
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb13, label %bb12
bb12:
  %v49 = extractvalue { ptr, i64 } %v18, 0
  %v50 = getelementptr inbounds float, ptr %v49, i64 %v21
  %v51 = insertvalue { ptr } undef, ptr %v50, 0
  %v52 = extractvalue { ptr } %v51, 0
  br label %bb15
bb13:
  br label %bb14
bb14:
  %v53 = inttoptr i64 0 to ptr
  %v54 = insertvalue { ptr } undef, ptr %v53, 0
  %v55 = extractvalue { ptr } %v54, 0
  br label %bb15
bb15:
  %v56 = phi ptr [ %v52, %bb12 ], [ %v55, %bb14 ]
  %v57 = insertvalue { ptr } undef, ptr %v56, 0
  %v58 = extractvalue { ptr } %v57, 0
  %v59 = ptrtoint ptr %v58 to i64
  %v60 = sub i64 %v59, 0
  %v61 = icmp ule i64 %v60, 0
  %v62 = add i64 %v60, 0
  %v63 = select i1 %v61, i64 %v62, i64 1
  %v64 = icmp eq i64 %v63, 1
  br i1 %v64, label %bb5, label %bb16
bb16:
  %v65 = icmp eq i64 %v63, 0
  br i1 %v65, label %bb8, label %bb17
bb17:
  unreachable
bb18:
  call void @llvm.trap() #0
  unreachable
bb19:
  call void @llvm.trap() #0
  unreachable
bb20:
  call void @llvm.trap() #0
  unreachable
}

declare i32 @llvm.fptoui.sat.i32.f32(float)

define ptx_kernel void @weighted_embedding_q6k_topk(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, ptr %v8, i64 %v9) #0 {
entry:
  %v10 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v1, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v3, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v9, 1
  br label %bb0
bb0:
  %v16 = phi { ptr, i64 } [ %v11, %entry ]
  %v17 = phi { ptr, i64 } [ %v13, %entry ]
  %v18 = phi i32 [ %v4, %entry ]
  %v19 = phi i32 [ %v5, %entry ]
  %v20 = phi i32 [ %v6, %entry ]
  %v21 = phi i32 [ %v7, %entry ]
  %v22 = phi { ptr, i64 } [ %v15, %entry ]
  %v23 = alloca {  }, align 1
  %v24 = alloca [2 x i8], align 1
  %v25 = bitcast ptr %v23 to ptr
  %v26 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v25) #0
  br label %bb1
bb1:
  %v27 = zext i32 %v18 to i64
  %v28 = zext i32 %v20 to i64
  %v29 = mul i64 %v27, %v28
  %v30 = icmp uge i64 %v26, %v29
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb3, label %bb2
bb2:
  br label %bb42
bb3:
  %v32 = icmp eq i64 %v28, 0
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb4, label %bb52
bb4:
  %v34 = udiv i64 %v26, %v28
  %v35 = urem i64 %v26, %v28
  %v36 = zext i32 %v19 to i64
  %v37 = mul i64 %v34, %v36
  %v38 = mul i64 %v37, 2
  br label %bb5
bb5:
  %v39 = phi float [ -340282346638528860000000000000000000000.0, %bb4 ], [ %v53, %bb10 ]
  %v40 = phi i64 [ 0, %bb4 ], [ %v54, %bb10 ]
  %v41 = icmp ult i64 %v40, %v36
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb11, label %bb6
bb6:
  %v43 = mul i64 %v40, 2
  %v44 = add i64 %v38, %v43
  %v45 = add i64 %v44, 1
  %v46 = extractvalue { ptr, i64 } %v17, 1
  %v47 = icmp ult i64 %v45, %v46
  br i1 %v47, label %bb7, label %bb53
bb7:
  %v48 = extractvalue { ptr, i64 } %v17, 0
  %v49 = getelementptr inbounds float, ptr %v48, i64 %v45
  %v50 = load float, ptr %v49, align 4
  %v51 = fcmp ogt float %v50, %v39
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb9, label %bb8
bb8:
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v53 = phi float [ %v50, %bb8 ], [ %v39, %bb9 ]
  %v54 = add i64 %v40, 1
  br label %bb5
bb11:
  br label %bb12
bb12:
  %v55 = phi i64 [ 0, %bb11 ], [ %v188, %bb43 ]
  %v56 = phi float [ 0.0, %bb11 ], [ %v187, %bb43 ]
  %v57 = icmp ult i64 %v55, %v36
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb15, label %bb13
bb13:
  %v59 = mul i64 %v55, 2
  %v60 = add i64 %v38, %v59
  %v61 = add i64 %v60, 1
  %v62 = extractvalue { ptr, i64 } %v17, 1
  %v63 = icmp ult i64 %v61, %v62
  br i1 %v63, label %bb14, label %bb54
bb14:
  %v64 = extractvalue { ptr, i64 } %v17, 0
  %v65 = getelementptr inbounds float, ptr %v64, i64 %v61
  %v66 = load float, ptr %v65, align 4
  %v67 = fsub contract float %v66, %v39
  %v68 = call float @__nv_expf(float %v67) #0
  br label %bb43
bb15:
  %v69 = udiv i64 %v28, 256
  %v70 = mul i64 %v69, 210
  %v71 = udiv i64 %v35, 256
  %v72 = urem i64 %v35, 256
  br label %bb16
bb16:
  %v73 = phi i64 [ 0, %bb15 ], [ %v184, %bb37 ]
  %v74 = phi float [ 0.0, %bb15 ], [ %v183, %bb37 ]
  %v75 = icmp ult i64 %v73, %v36
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb38, label %bb17
bb17:
  %v77 = mul i64 %v73, 2
  %v78 = add i64 %v38, %v77
  %v79 = extractvalue { ptr, i64 } %v17, 1
  %v80 = icmp ult i64 %v78, %v79
  br i1 %v80, label %bb18, label %bb55
bb18:
  %v81 = extractvalue { ptr, i64 } %v17, 0
  %v82 = getelementptr inbounds float, ptr %v81, i64 %v78
  %v83 = load float, ptr %v82, align 4
  %v84 = call i32 @llvm.fptoui.sat.i32.f32(float %v83) #0
  %v85 = icmp ult i32 %v84, %v21
  %v86 = xor i1 %v85, 1
  br i1 %v86, label %bb37, label %bb19
bb19:
  %v87 = fcmp ogt float %v56, 0.0
  %v88 = xor i1 %v87, 1
  br i1 %v88, label %bb36, label %bb20
bb20:
  %v89 = zext i32 %v84 to i64
  %v90 = mul i64 %v89, %v70
  %v91 = mul i64 %v71, 210
  %v92 = add i64 %v90, %v91
  %v93 = add i64 %v92, 208
  %v94 = extractvalue { ptr, i64 } %v16, 1
  %v95 = icmp ult i64 %v93, %v94
  br i1 %v95, label %bb21, label %bb56
bb21:
  %v96 = extractvalue { ptr, i64 } %v16, 0
  %v97 = getelementptr inbounds i8, ptr %v96, i64 %v93
  %v98 = load i8, ptr %v97, align 1
  %v99 = add i64 %v92, 209
  %v100 = icmp ult i64 %v99, %v94
  br i1 %v100, label %bb22, label %bb57
bb22:
  %v101 = extractvalue { ptr, i64 } %v16, 0
  %v102 = getelementptr inbounds i8, ptr %v101, i64 %v99
  %v103 = load i8, ptr %v102, align 1
  %v104 = getelementptr inbounds [2 x i8], ptr %v24, i32 0, i64 0
  store i8 %v98, ptr %v104, align 1
  %v105 = getelementptr inbounds [2 x i8], ptr %v24, i32 0, i64 1
  store i8 %v103, ptr %v105, align 1
  %v106 = load [2 x i8], ptr %v24, align 1
  %v107 = alloca [2 x i8], align 2
  store [2 x i8] %v106, ptr %v107, align 2
  %v108 = load i16, ptr %v107, align 2
  %v109 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v108) #0
  br label %bb23
bb23:
  %v110 = udiv i64 %v72, 128
  %v111 = urem i64 %v72, 128
  %v112 = urem i64 %v111, 32
  %v113 = udiv i64 %v111, 32
  %v114 = mul i64 %v110, 64
  %v115 = add i64 %v92, %v114
  %v116 = add i64 %v92, 128
  %v117 = mul i64 %v110, 32
  %v118 = add i64 %v116, %v117
  %v119 = add i64 %v92, 192
  %v120 = mul i64 %v110, 8
  %v121 = add i64 %v119, %v120
  %v122 = trunc i64 %v113 to i32
  %v123 = mul i32 %v122, 2
  %v124 = icmp eq i64 %v113, 0
  br i1 %v124, label %bb25, label %bb24
bb24:
  %v125 = icmp eq i64 %v113, 2
  br i1 %v125, label %bb25, label %bb26
bb25:
  br label %bb27
bb26:
  %v126 = add i64 %v112, 32
  br label %bb27
bb27:
  %v127 = phi i64 [ %v112, %bb25 ], [ %v126, %bb26 ]
  %v128 = icmp ult i64 %v113, 2
  %v129 = xor i1 %v128, 1
  br i1 %v129, label %bb30, label %bb28
bb28:
  %v130 = add i64 %v115, %v127
  %v131 = icmp ult i64 %v130, %v94
  br i1 %v131, label %bb29, label %bb58
bb29:
  %v132 = extractvalue { ptr, i64 } %v16, 0
  %v133 = getelementptr inbounds i8, ptr %v132, i64 %v130
  %v134 = load i8, ptr %v133, align 1
  %v135 = and i8 %v134, 15
  %v136 = zext i8 %v135 to i32
  br label %bb32
bb30:
  %v137 = add i64 %v115, %v127
  %v138 = icmp ult i64 %v137, %v94
  br i1 %v138, label %bb31, label %bb59
bb31:
  %v139 = extractvalue { ptr, i64 } %v16, 0
  %v140 = getelementptr inbounds i8, ptr %v139, i64 %v137
  %v141 = load i8, ptr %v140, align 1
  %v142 = trunc i32 4 to i8
  %v143 = and i8 %v142, 7
  %v144 = lshr i8 %v141, %v143
  %v145 = zext i8 %v144 to i32
  br label %bb32
bb32:
  %v146 = phi i32 [ %v136, %bb29 ], [ %v145, %bb31 ]
  %v147 = add i64 %v118, %v112
  %v148 = icmp ult i64 %v147, %v94
  br i1 %v148, label %bb33, label %bb60
bb33:
  %v149 = extractvalue { ptr, i64 } %v16, 0
  %v150 = getelementptr inbounds i8, ptr %v149, i64 %v147
  %v151 = load i8, ptr %v150, align 1
  %v152 = trunc i32 %v123 to i8
  %v153 = and i8 %v152, 7
  %v154 = lshr i8 %v151, %v153
  %v155 = and i8 %v154, 3
  %v156 = zext i8 %v155 to i32
  %v157 = and i32 4, 31
  %v158 = shl i32 %v156, %v157
  %v159 = or i32 %v146, %v158
  %v160 = sub i32 %v159, 32
  %v161 = udiv i64 %v112, 16
  %v162 = add i64 %v121, %v161
  %v163 = mul i64 %v113, 2
  %v164 = add i64 %v162, %v163
  %v165 = icmp ult i64 %v164, %v94
  br i1 %v165, label %bb34, label %bb61
bb34:
  %v166 = extractvalue { ptr, i64 } %v16, 0
  %v167 = getelementptr inbounds i8, ptr %v166, i64 %v164
  %v168 = load i8, ptr %v167, align 1
  %v169 = bitcast i8 %v168 to i8
  %v170 = sitofp i8 %v169 to float
  %v171 = fmul contract float %v109, %v170
  %v172 = sitofp i32 %v160 to float
  %v173 = fmul contract float %v171, %v172
  %v174 = mul i64 %v73, 2
  %v175 = add i64 %v38, %v174
  %v176 = add i64 %v175, 1
  %v177 = icmp ult i64 %v176, %v79
  br i1 %v177, label %bb35, label %bb62
bb35:
  %v178 = extractvalue { ptr, i64 } %v17, 0
  %v179 = getelementptr inbounds float, ptr %v178, i64 %v176
  %v180 = load float, ptr %v179, align 4
  %v181 = fsub contract float %v180, %v39
  %v182 = call float @__nv_expf(float %v181) #0
  br label %bb44
bb36:
  br label %bb37
bb37:
  %v183 = phi float [ %v74, %bb18 ], [ %v74, %bb36 ], [ %v191, %bb44 ]
  %v184 = add i64 %v73, 1
  br label %bb16
bb38:
  %v185 = icmp eq i64 %v26, 18446744073709551615
  br i1 %v185, label %bb48, label %bb45
bb39:
  %v186 = extractvalue { ptr } %v203, 0
  store float %v74, ptr %v186, align 4
  br label %bb41
bb40:
  br label %bb41
bb41:
  br label %bb42
bb42:
  ret void
bb43:
  %v187 = fadd contract float %v56, %v68
  %v188 = add i64 %v55, 1
  br label %bb12
bb44:
  %v189 = fdiv contract float %v182, %v56
  %v190 = fmul contract float %v189, %v173
  %v191 = fadd contract float %v74, %v190
  br label %bb37
bb45:
  %v192 = extractvalue { ptr, i64 } %v22, 1
  %v193 = icmp ult i64 %v26, %v192
  %v194 = xor i1 %v193, 1
  br i1 %v194, label %bb47, label %bb46
bb46:
  %v195 = extractvalue { ptr, i64 } %v22, 0
  %v196 = getelementptr inbounds float, ptr %v195, i64 %v26
  %v197 = insertvalue { ptr } undef, ptr %v196, 0
  %v198 = extractvalue { ptr } %v197, 0
  br label %bb49
bb47:
  br label %bb48
bb48:
  %v199 = inttoptr i64 0 to ptr
  %v200 = insertvalue { ptr } undef, ptr %v199, 0
  %v201 = extractvalue { ptr } %v200, 0
  br label %bb49
bb49:
  %v202 = phi ptr [ %v198, %bb46 ], [ %v201, %bb48 ]
  %v203 = insertvalue { ptr } undef, ptr %v202, 0
  %v204 = extractvalue { ptr } %v203, 0
  %v205 = ptrtoint ptr %v204 to i64
  %v206 = sub i64 %v205, 0
  %v207 = icmp ule i64 %v206, 0
  %v208 = add i64 %v206, 0
  %v209 = select i1 %v207, i64 %v208, i64 1
  %v210 = icmp eq i64 %v209, 1
  br i1 %v210, label %bb39, label %bb50
bb50:
  %v211 = icmp eq i64 %v209, 0
  br i1 %v211, label %bb40, label %bb51
bb51:
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
bb57:
  call void @llvm.trap() #0
  unreachable
bb58:
  call void @llvm.trap() #0
  unreachable
bb59:
  call void @llvm.trap() #0
  unreachable
bb60:
  call void @llvm.trap() #0
  unreachable
bb61:
  call void @llvm.trap() #0
  unreachable
bb62:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_q6k_row(ptr %v0, i64 %v1, i32 %v2, i32 %v3, i32 %v4, ptr %v5, i64 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v5, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v6, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi i32 [ %v2, %entry ]
  %v13 = phi i32 [ %v3, %entry ]
  %v14 = phi i32 [ %v4, %entry ]
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = alloca {  }, align 1
  %v17 = alloca [2 x i8], align 1
  %v18 = bitcast ptr %v16 to ptr
  %v19 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v18) #0
  br label %bb1
bb1:
  %v20 = zext i32 %v13 to i64
  %v21 = icmp uge i64 %v19, %v20
  %v22 = xor i1 %v21, 1
  br i1 %v22, label %bb3, label %bb2
bb2:
  br label %bb29
bb3:
  %v23 = mul i32 %v14, 210
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v12 to i64
  %v26 = mul i64 %v25, %v24
  %v27 = udiv i64 %v19, 256
  %v28 = urem i64 %v19, 256
  %v29 = mul i64 %v27, 210
  %v30 = add i64 %v26, %v29
  %v31 = add i64 %v30, 208
  %v32 = extractvalue { ptr, i64 } %v11, 1
  %v33 = icmp ult i64 %v31, %v32
  br i1 %v33, label %bb4, label %bb37
bb4:
  %v34 = extractvalue { ptr, i64 } %v11, 0
  %v35 = getelementptr inbounds i8, ptr %v34, i64 %v31
  %v36 = load i8, ptr %v35, align 1
  %v37 = add i64 %v30, 209
  %v38 = icmp ult i64 %v37, %v32
  br i1 %v38, label %bb5, label %bb38
bb5:
  %v39 = extractvalue { ptr, i64 } %v11, 0
  %v40 = getelementptr inbounds i8, ptr %v39, i64 %v37
  %v41 = load i8, ptr %v40, align 1
  %v42 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 0
  store i8 %v36, ptr %v42, align 1
  %v43 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 1
  store i8 %v41, ptr %v43, align 1
  %v44 = load [2 x i8], ptr %v17, align 1
  %v45 = alloca [2 x i8], align 2
  store [2 x i8] %v44, ptr %v45, align 2
  %v46 = load i16, ptr %v45, align 2
  %v47 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v46) #0
  br label %bb6
bb6:
  %v48 = udiv i64 %v28, 128
  %v49 = urem i64 %v28, 128
  %v50 = mul i64 %v48, 64
  %v51 = add i64 %v30, %v50
  %v52 = add i64 %v30, 128
  %v53 = mul i64 %v48, 32
  %v54 = add i64 %v52, %v53
  %v55 = add i64 %v30, 192
  %v56 = mul i64 %v48, 8
  %v57 = add i64 %v55, %v56
  %v58 = urem i64 %v49, 32
  %v59 = udiv i64 %v49, 32
  %v60 = udiv i64 %v58, 16
  %v61 = icmp eq i64 %v59, 0
  br i1 %v61, label %bb12, label %bb7
bb7:
  %v62 = icmp eq i64 %v59, 1
  br i1 %v62, label %bb11, label %bb8
bb8:
  %v63 = icmp eq i64 %v59, 2
  br i1 %v63, label %bb10, label %bb9
bb9:
  br label %bb13
bb10:
  br label %bb13
bb11:
  br label %bb13
bb12:
  br label %bb13
bb13:
  %v64 = phi i32 [ 6, %bb9 ], [ 4, %bb10 ], [ 2, %bb11 ], [ 0, %bb12 ]
  %v65 = icmp eq i64 %v59, 0
  %v66 = icmp eq i64 %v59, 0
  br i1 %v66, label %bb15, label %bb14
bb14:
  %v67 = icmp eq i64 %v59, 2
  br i1 %v67, label %bb15, label %bb16
bb15:
  br label %bb17
bb16:
  %v68 = add i64 %v58, 32
  br label %bb17
bb17:
  %v69 = phi i64 [ %v58, %bb15 ], [ %v68, %bb16 ]
  %v70 = xor i1 %v65, 1
  br i1 %v70, label %bb18, label %bb19
bb18:
  %v71 = icmp eq i64 %v59, 1
  br i1 %v71, label %bb19, label %bb21
bb19:
  %v72 = add i64 %v51, %v69
  %v73 = icmp ult i64 %v72, %v32
  br i1 %v73, label %bb20, label %bb39
bb20:
  %v74 = extractvalue { ptr, i64 } %v11, 0
  %v75 = getelementptr inbounds i8, ptr %v74, i64 %v72
  %v76 = load i8, ptr %v75, align 1
  %v77 = and i8 %v76, 15
  %v78 = zext i8 %v77 to i32
  br label %bb23
bb21:
  %v79 = add i64 %v51, %v69
  %v80 = icmp ult i64 %v79, %v32
  br i1 %v80, label %bb22, label %bb40
bb22:
  %v81 = extractvalue { ptr, i64 } %v11, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = trunc i32 4 to i8
  %v85 = and i8 %v84, 7
  %v86 = lshr i8 %v83, %v85
  %v87 = zext i8 %v86 to i32
  br label %bb23
bb23:
  %v88 = phi i32 [ %v78, %bb20 ], [ %v87, %bb22 ]
  %v89 = add i64 %v54, %v58
  %v90 = icmp ult i64 %v89, %v32
  br i1 %v90, label %bb24, label %bb41
bb24:
  %v91 = extractvalue { ptr, i64 } %v11, 0
  %v92 = getelementptr inbounds i8, ptr %v91, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = trunc i32 %v64 to i8
  %v95 = and i8 %v94, 7
  %v96 = lshr i8 %v93, %v95
  %v97 = and i8 %v96, 3
  %v98 = zext i8 %v97 to i32
  %v99 = and i32 4, 31
  %v100 = shl i32 %v98, %v99
  %v101 = or i32 %v88, %v100
  %v102 = sub i32 %v101, 32
  %v103 = add i64 %v57, %v60
  %v104 = mul i64 %v59, 2
  %v105 = add i64 %v103, %v104
  %v106 = icmp ult i64 %v105, %v32
  br i1 %v106, label %bb25, label %bb42
bb25:
  %v107 = extractvalue { ptr, i64 } %v11, 0
  %v108 = getelementptr inbounds i8, ptr %v107, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v110 = bitcast i8 %v109 to i8
  %v111 = sitofp i8 %v110 to float
  %v112 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v112, label %bb33, label %bb30
bb26:
  %v113 = extractvalue { ptr } %v128, 0
  %v114 = fmul contract float %v47, %v111
  %v115 = sitofp i32 %v102 to float
  %v116 = fmul contract float %v114, %v115
  store float %v116, ptr %v113, align 4
  br label %bb28
bb27:
  br label %bb28
bb28:
  br label %bb29
bb29:
  ret void
bb30:
  %v117 = extractvalue { ptr, i64 } %v15, 1
  %v118 = icmp ult i64 %v19, %v117
  %v119 = xor i1 %v118, 1
  br i1 %v119, label %bb32, label %bb31
bb31:
  %v120 = extractvalue { ptr, i64 } %v15, 0
  %v121 = getelementptr inbounds float, ptr %v120, i64 %v19
  %v122 = insertvalue { ptr } undef, ptr %v121, 0
  %v123 = extractvalue { ptr } %v122, 0
  br label %bb34
bb32:
  br label %bb33
bb33:
  %v124 = inttoptr i64 0 to ptr
  %v125 = insertvalue { ptr } undef, ptr %v124, 0
  %v126 = extractvalue { ptr } %v125, 0
  br label %bb34
bb34:
  %v127 = phi ptr [ %v123, %bb31 ], [ %v126, %bb33 ]
  %v128 = insertvalue { ptr } undef, ptr %v127, 0
  %v129 = extractvalue { ptr } %v128, 0
  %v130 = ptrtoint ptr %v129 to i64
  %v131 = sub i64 %v130, 0
  %v132 = icmp ule i64 %v131, 0
  %v133 = add i64 %v131, 0
  %v134 = select i1 %v132, i64 %v133, i64 1
  %v135 = icmp eq i64 %v134, 1
  br i1 %v135, label %bb26, label %bb35
bb35:
  %v136 = icmp eq i64 %v134, 0
  br i1 %v136, label %bb27, label %bb36
bb36:
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_q5_0_project_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr %v12, i64 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v13, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi { ptr, i64 } [ %v23, %entry ]
  %v33 = alloca [2 x i8], align 1
  %v34 = alloca [4 x i8], align 1
  %v35 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v36 = zext i32 %v35 to i64
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v38 = zext i32 %v37 to i64
  %v39 = zext i32 %v28 to i64
  %v40 = zext i32 %v29 to i64
  %v41 = mul i64 %v39, %v40
  %v42 = zext i32 %v30 to i64
  %v43 = mul i64 %v41, %v42
  %v44 = icmp uge i64 %v38, %v43
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb4, label %bb3
bb3:
  br label %bb38
bb4:
  %v46 = icmp eq i64 %v42, 0
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb5, label %bb39
bb5:
  %v48 = urem i64 %v38, %v42
  %v49 = udiv i64 %v38, %v42
  %v50 = extractvalue { ptr, i64 } %v27, 1
  %v51 = icmp ult i64 %v49, %v50
  br i1 %v51, label %bb6, label %bb40
bb6:
  %v52 = extractvalue { ptr, i64 } %v27, 0
  %v53 = getelementptr inbounds i32, ptr %v52, i64 %v49
  %v54 = load i32, ptr %v53, align 4
  %v55 = zext i32 %v54 to i64
  %v56 = extractvalue { ptr, i64 } %v26, 1
  %v57 = icmp ult i64 %v55, %v56
  br i1 %v57, label %bb7, label %bb41
bb7:
  %v58 = extractvalue { ptr, i64 } %v26, 0
  %v59 = getelementptr inbounds i32, ptr %v58, i64 %v55
  %v60 = load i32, ptr %v59, align 4
  %v61 = zext i32 %v60 to i64
  %v62 = zext i32 %v31 to i64
  %v63 = udiv i64 %v62, 32
  %v64 = mul i64 %v63, 22
  br label %bb8
bb8:
  %v65 = phi float [ 0.0, %bb7 ], [ %v120, %bb22 ]
  %v66 = phi i64 [ %v36, %bb7 ], [ %v176, %bb22 ]
  %v67 = icmp ult i64 %v66, %v63
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb23, label %bb9
bb9:
  %v69 = mul i64 %v61, %v42
  %v70 = add i64 %v69, %v48
  %v71 = mul i64 %v70, %v64
  %v72 = mul i64 %v66, 22
  %v73 = add i64 %v71, %v72
  %v74 = extractvalue { ptr, i64 } %v24, 1
  %v75 = icmp ult i64 %v73, %v74
  br i1 %v75, label %bb10, label %bb42
bb10:
  %v76 = extractvalue { ptr, i64 } %v24, 0
  %v77 = getelementptr inbounds i8, ptr %v76, i64 %v73
  %v78 = load i8, ptr %v77, align 1
  %v79 = add i64 %v73, 1
  %v80 = icmp ult i64 %v79, %v74
  br i1 %v80, label %bb11, label %bb43
bb11:
  %v81 = extractvalue { ptr, i64 } %v24, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = getelementptr inbounds [2 x i8], ptr %v33, i32 0, i64 0
  store i8 %v78, ptr %v84, align 1
  %v85 = getelementptr inbounds [2 x i8], ptr %v33, i32 0, i64 1
  store i8 %v83, ptr %v85, align 1
  %v86 = load [2 x i8], ptr %v33, align 1
  %v87 = alloca [2 x i8], align 2
  store [2 x i8] %v86, ptr %v87, align 2
  %v88 = load i16, ptr %v87, align 2
  %v89 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v88) #0
  br label %bb12
bb12:
  %v90 = add i64 %v73, 2
  %v91 = icmp ult i64 %v90, %v74
  br i1 %v91, label %bb13, label %bb44
bb13:
  %v92 = extractvalue { ptr, i64 } %v24, 0
  %v93 = getelementptr inbounds i8, ptr %v92, i64 %v90
  %v94 = load i8, ptr %v93, align 1
  %v95 = add i64 %v73, 3
  %v96 = icmp ult i64 %v95, %v74
  br i1 %v96, label %bb14, label %bb45
bb14:
  %v97 = extractvalue { ptr, i64 } %v24, 0
  %v98 = getelementptr inbounds i8, ptr %v97, i64 %v95
  %v99 = load i8, ptr %v98, align 1
  %v100 = add i64 %v73, 4
  %v101 = icmp ult i64 %v100, %v74
  br i1 %v101, label %bb15, label %bb46
bb15:
  %v102 = extractvalue { ptr, i64 } %v24, 0
  %v103 = getelementptr inbounds i8, ptr %v102, i64 %v100
  %v104 = load i8, ptr %v103, align 1
  %v105 = add i64 %v73, 5
  %v106 = icmp ult i64 %v105, %v74
  br i1 %v106, label %bb16, label %bb47
bb16:
  %v107 = extractvalue { ptr, i64 } %v24, 0
  %v108 = getelementptr inbounds i8, ptr %v107, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v110 = getelementptr inbounds [4 x i8], ptr %v34, i32 0, i64 0
  store i8 %v94, ptr %v110, align 1
  %v111 = getelementptr inbounds [4 x i8], ptr %v34, i32 0, i64 1
  store i8 %v99, ptr %v111, align 1
  %v112 = getelementptr inbounds [4 x i8], ptr %v34, i32 0, i64 2
  store i8 %v104, ptr %v112, align 1
  %v113 = getelementptr inbounds [4 x i8], ptr %v34, i32 0, i64 3
  store i8 %v109, ptr %v113, align 1
  %v114 = load [4 x i8], ptr %v34, align 1
  %v115 = alloca [4 x i8], align 4
  store [4 x i8] %v114, ptr %v115, align 4
  %v116 = load i32, ptr %v115, align 4
  %v117 = mul i64 %v55, %v63
  %v118 = add i64 %v117, %v66
  %v119 = mul i64 %v118, 32
  br label %bb17
bb17:
  %v120 = phi float [ %v65, %bb16 ], [ %v174, %bb21 ]
  %v121 = phi i64 [ 0, %bb16 ], [ %v175, %bb21 ]
  %v122 = icmp ult i64 %v121, 16
  %v123 = xor i1 %v122, 1
  br i1 %v123, label %bb22, label %bb18
bb18:
  %v124 = add i64 %v73, 6
  %v125 = add i64 %v124, %v121
  %v126 = icmp ult i64 %v125, %v74
  br i1 %v126, label %bb19, label %bb48
bb19:
  %v127 = extractvalue { ptr, i64 } %v24, 0
  %v128 = getelementptr inbounds i8, ptr %v127, i64 %v125
  %v129 = load i8, ptr %v128, align 1
  %v130 = trunc i64 %v121 to i32
  %v131 = and i32 %v130, 31
  %v132 = lshr i32 %v116, %v131
  %v133 = and i32 %v132, 1
  %v134 = bitcast i32 %v133 to i32
  %v135 = and i32 4, 31
  %v136 = shl i32 %v134, %v135
  %v137 = and i8 %v129, 15
  %v138 = zext i8 %v137 to i32
  %v139 = or i32 %v136, %v138
  %v140 = sub i32 %v139, 16
  %v141 = add i64 %v121, 16
  %v142 = trunc i64 %v141 to i32
  %v143 = and i32 %v142, 31
  %v144 = lshr i32 %v116, %v143
  %v145 = and i32 %v144, 1
  %v146 = bitcast i32 %v145 to i32
  %v147 = and i32 4, 31
  %v148 = shl i32 %v146, %v147
  %v149 = trunc i32 4 to i8
  %v150 = and i8 %v149, 7
  %v151 = lshr i8 %v129, %v150
  %v152 = zext i8 %v151 to i32
  %v153 = or i32 %v148, %v152
  %v154 = sub i32 %v153, 16
  %v155 = sitofp i32 %v140 to float
  %v156 = fmul contract float %v89, %v155
  %v157 = add i64 %v119, %v121
  %v158 = extractvalue { ptr, i64 } %v25, 1
  %v159 = icmp ult i64 %v157, %v158
  br i1 %v159, label %bb20, label %bb49
bb20:
  %v160 = extractvalue { ptr, i64 } %v25, 0
  %v161 = getelementptr inbounds float, ptr %v160, i64 %v157
  %v162 = load float, ptr %v161, align 4
  %v163 = fmul contract float %v156, %v162
  %v164 = sitofp i32 %v154 to float
  %v165 = fmul contract float %v89, %v164
  %v166 = add i64 %v119, %v121
  %v167 = add i64 %v166, 16
  %v168 = icmp ult i64 %v167, %v158
  br i1 %v168, label %bb21, label %bb50
bb21:
  %v169 = extractvalue { ptr, i64 } %v25, 0
  %v170 = getelementptr inbounds float, ptr %v169, i64 %v167
  %v171 = load float, ptr %v170, align 4
  %v172 = fmul contract float %v165, %v171
  %v173 = fadd contract float %v163, %v172
  %v174 = fadd contract float %v120, %v173
  %v175 = add i64 %v121, 1
  br label %bb17
bb22:
  %v176 = add i64 %v66, 32
  br label %bb8
bb23:
  %v177 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_6, i64 %v36
  br label %bb24
bb24:
  store float %v65, ptr addrspace(3) %v177, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb25
bb25:
  br label %bb26
bb26:
  %v179 = phi i64 [ 16, %bb25 ], [ %v192, %bb33 ]
  %v180 = icmp ugt i64 %v179, 0
  %v181 = xor i1 %v180, 1
  br i1 %v181, label %bb34, label %bb27
bb27:
  %v182 = icmp ult i64 %v36, %v179
  %v183 = xor i1 %v182, 1
  br i1 %v183, label %bb31, label %bb28
bb28:
  %v184 = bitcast ptr addrspace(3) @__shared_mem_6 to ptr addrspace(3)
  %v185 = add i64 %v36, %v179
  %v186 = getelementptr inbounds float, ptr addrspace(3) %v184, i64 %v185
  br label %bb29
bb29:
  %v187 = load float, ptr addrspace(3) %v186, align 4
  %v188 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_6, i64 %v36
  br label %bb30
bb30:
  %v189 = load float, ptr addrspace(3) %v188, align 4
  %v190 = fadd contract float %v189, %v187
  store float %v190, ptr addrspace(3) %v188, align 4
  br label %bb32
bb31:
  br label %bb32
bb32:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb33
bb33:
  %v192 = udiv i64 %v179, 2
  br label %bb26
bb34:
  %v193 = icmp eq i64 %v36, 0
  br i1 %v193, label %bb35, label %bb37
bb35:
  %v194 = bitcast ptr addrspace(3) @__shared_mem_6 to ptr addrspace(3)
  %v195 = getelementptr inbounds float, ptr addrspace(3) %v194, i64 0
  br label %bb36
bb36:
  %v196 = load float, ptr addrspace(3) %v195, align 4
  %v197 = mul i64 %v55, %v42
  %v198 = add i64 %v197, %v48
  %v199 = extractvalue { ptr, i64 } %v32, 0
  %v200 = getelementptr inbounds float, ptr %v199, i64 %v198
  store float %v196, ptr %v200, align 4
  br label %bb37
bb37:
  br label %bb38
bb38:
  ret void
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q4k_gemv_row(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = alloca [2 x i8], align 1
  %v24 = alloca [8 x i8], align 1
  %v25 = alloca [8 x i8], align 1
  %v26 = bitcast ptr %v21 to ptr
  %v27 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v26) #0
  br label %bb1
bb1:
  %v28 = zext i32 %v17 to i64
  %v29 = zext i32 %v19 to i64
  %v30 = mul i64 %v28, %v29
  %v31 = icmp uge i64 %v27, %v30
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb3, label %bb2
bb2:
  br label %bb53
bb3:
  %v33 = icmp eq i64 %v28, 0
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb4, label %bb61
bb4:
  %v35 = urem i64 %v27, %v28
  %v36 = udiv i64 %v27, %v28
  %v37 = mul i32 %v18, 144
  %v38 = zext i32 %v37 to i64
  %v39 = mul i64 %v35, %v38
  br label %bb5
bb5:
  %v40 = phi float [ 0.0, %bb4 ], [ %v176, %bb48 ]
  %v41 = phi i32 [ 0, %bb4 ], [ %v247, %bb48 ]
  %v42 = icmp ult i32 %v41, %v18
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb49, label %bb6
bb6:
  %v44 = zext i32 %v41 to i64
  %v45 = mul i64 %v44, 144
  %v46 = add i64 %v39, %v45
  %v47 = extractvalue { ptr, i64 } %v15, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb7, label %bb62
bb7:
  %v49 = extractvalue { ptr, i64 } %v15, 0
  %v50 = getelementptr inbounds i8, ptr %v49, i64 %v46
  %v51 = load i8, ptr %v50, align 1
  %v52 = add i64 %v46, 1
  %v53 = icmp ult i64 %v52, %v47
  br i1 %v53, label %bb8, label %bb63
bb8:
  %v54 = extractvalue { ptr, i64 } %v15, 0
  %v55 = getelementptr inbounds i8, ptr %v54, i64 %v52
  %v56 = load i8, ptr %v55, align 1
  %v57 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v51, ptr %v57, align 1
  %v58 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v56, ptr %v58, align 1
  %v59 = load [2 x i8], ptr %v22, align 1
  %v60 = alloca [2 x i8], align 2
  store [2 x i8] %v59, ptr %v60, align 2
  %v61 = load i16, ptr %v60, align 2
  %v62 = add i64 %v46, 2
  %v63 = icmp ult i64 %v62, %v47
  br i1 %v63, label %bb9, label %bb64
bb9:
  %v64 = extractvalue { ptr, i64 } %v15, 0
  %v65 = getelementptr inbounds i8, ptr %v64, i64 %v62
  %v66 = load i8, ptr %v65, align 1
  %v67 = add i64 %v46, 3
  %v68 = icmp ult i64 %v67, %v47
  br i1 %v68, label %bb10, label %bb65
bb10:
  %v69 = extractvalue { ptr, i64 } %v15, 0
  %v70 = getelementptr inbounds i8, ptr %v69, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v72 = getelementptr inbounds [2 x i8], ptr %v23, i32 0, i64 0
  store i8 %v66, ptr %v72, align 1
  %v73 = getelementptr inbounds [2 x i8], ptr %v23, i32 0, i64 1
  store i8 %v71, ptr %v73, align 1
  %v74 = load [2 x i8], ptr %v23, align 1
  %v75 = alloca [2 x i8], align 2
  store [2 x i8] %v74, ptr %v75, align 2
  %v76 = load i16, ptr %v75, align 2
  %v77 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v61) #0
  br label %bb11
bb11:
  %v78 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v76) #0
  br label %bb12
bb12:
  %v79 = add i64 %v46, 4
  %v80 = icmp ult i64 %v79, %v47
  br i1 %v80, label %bb13, label %bb66
bb13:
  %v81 = extractvalue { ptr, i64 } %v15, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = add i64 %v46, 5
  %v85 = icmp ult i64 %v84, %v47
  br i1 %v85, label %bb14, label %bb67
bb14:
  %v86 = extractvalue { ptr, i64 } %v15, 0
  %v87 = getelementptr inbounds i8, ptr %v86, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = add i64 %v46, 6
  %v90 = icmp ult i64 %v89, %v47
  br i1 %v90, label %bb15, label %bb68
bb15:
  %v91 = extractvalue { ptr, i64 } %v15, 0
  %v92 = getelementptr inbounds i8, ptr %v91, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = add i64 %v46, 7
  %v95 = icmp ult i64 %v94, %v47
  br i1 %v95, label %bb16, label %bb69
bb16:
  %v96 = extractvalue { ptr, i64 } %v15, 0
  %v97 = getelementptr inbounds i8, ptr %v96, i64 %v94
  %v98 = load i8, ptr %v97, align 1
  %v99 = add i64 %v46, 8
  %v100 = icmp ult i64 %v99, %v47
  br i1 %v100, label %bb17, label %bb70
bb17:
  %v101 = extractvalue { ptr, i64 } %v15, 0
  %v102 = getelementptr inbounds i8, ptr %v101, i64 %v99
  %v103 = load i8, ptr %v102, align 1
  %v104 = add i64 %v46, 9
  %v105 = icmp ult i64 %v104, %v47
  br i1 %v105, label %bb18, label %bb71
bb18:
  %v106 = extractvalue { ptr, i64 } %v15, 0
  %v107 = getelementptr inbounds i8, ptr %v106, i64 %v104
  %v108 = load i8, ptr %v107, align 1
  %v109 = add i64 %v46, 10
  %v110 = icmp ult i64 %v109, %v47
  br i1 %v110, label %bb19, label %bb72
bb19:
  %v111 = extractvalue { ptr, i64 } %v15, 0
  %v112 = getelementptr inbounds i8, ptr %v111, i64 %v109
  %v113 = load i8, ptr %v112, align 1
  %v114 = add i64 %v46, 11
  %v115 = icmp ult i64 %v114, %v47
  br i1 %v115, label %bb20, label %bb73
bb20:
  %v116 = extractvalue { ptr, i64 } %v15, 0
  %v117 = getelementptr inbounds i8, ptr %v116, i64 %v114
  %v118 = load i8, ptr %v117, align 1
  %v119 = add i64 %v46, 12
  %v120 = icmp ult i64 %v119, %v47
  br i1 %v120, label %bb21, label %bb74
bb21:
  %v121 = extractvalue { ptr, i64 } %v15, 0
  %v122 = getelementptr inbounds i8, ptr %v121, i64 %v119
  %v123 = load i8, ptr %v122, align 1
  %v124 = add i64 %v46, 13
  %v125 = icmp ult i64 %v124, %v47
  br i1 %v125, label %bb22, label %bb75
bb22:
  %v126 = extractvalue { ptr, i64 } %v15, 0
  %v127 = getelementptr inbounds i8, ptr %v126, i64 %v124
  %v128 = load i8, ptr %v127, align 1
  %v129 = add i64 %v46, 14
  %v130 = icmp ult i64 %v129, %v47
  br i1 %v130, label %bb23, label %bb76
bb23:
  %v131 = extractvalue { ptr, i64 } %v15, 0
  %v132 = getelementptr inbounds i8, ptr %v131, i64 %v129
  %v133 = load i8, ptr %v132, align 1
  %v134 = add i64 %v46, 15
  %v135 = icmp ult i64 %v134, %v47
  br i1 %v135, label %bb24, label %bb77
bb24:
  %v136 = extractvalue { ptr, i64 } %v15, 0
  %v137 = getelementptr inbounds i8, ptr %v136, i64 %v134
  %v138 = load i8, ptr %v137, align 1
  %v139 = call { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v83, i8 %v88, i8 %v93, i8 %v98, i8 %v103, i8 %v108, i8 %v113, i8 %v118, i8 %v123, i8 %v128, i8 %v133, i8 %v138) #0
  br label %bb25
bb25:
  %v140 = extractvalue { [8 x i8], [8 x i8] } %v139, 0
  store [8 x i8] %v140, ptr %v24, align 1
  %v141 = extractvalue { [8 x i8], [8 x i8] } %v139, 1
  store [8 x i8] %v141, ptr %v25, align 1
  %v142 = add i64 %v46, 16
  %v143 = zext i32 %v18 to i64
  %v144 = mul i64 %v36, %v143
  %v145 = mul i64 %v144, 256
  %v146 = zext i32 %v41 to i64
  %v147 = mul i64 %v146, 256
  %v148 = add i64 %v145, %v147
  br label %bb26
bb26:
  %v149 = phi float [ 0.0, %bb25 ], [ %v172, %bb32 ]
  %v150 = phi i64 [ 0, %bb25 ], [ %v173, %bb32 ]
  %v151 = icmp ult i64 %v150, 8
  %v152 = xor i1 %v151, 1
  br i1 %v152, label %bb33, label %bb27
bb27:
  br label %bb28
bb28:
  %v153 = phi float [ 0.0, %bb27 ], [ %v165, %bb30 ]
  %v154 = phi i64 [ 0, %bb27 ], [ %v166, %bb30 ]
  %v155 = icmp ult i64 %v154, 32
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb31, label %bb29
bb29:
  %v157 = mul i64 %v150, 32
  %v158 = add i64 %v148, %v157
  %v159 = add i64 %v158, %v154
  %v160 = extractvalue { ptr, i64 } %v16, 1
  %v161 = icmp ult i64 %v159, %v160
  br i1 %v161, label %bb30, label %bb78
bb30:
  %v162 = extractvalue { ptr, i64 } %v16, 0
  %v163 = getelementptr inbounds float, ptr %v162, i64 %v159
  %v164 = load float, ptr %v163, align 4
  %v165 = fadd contract float %v153, %v164
  %v166 = add i64 %v154, 1
  br label %bb28
bb31:
  %v167 = icmp ult i64 %v150, 8
  br i1 %v167, label %bb32, label %bb79
bb32:
  %v168 = getelementptr inbounds [8 x i8], ptr %v25, i32 0, i64 %v150
  %v169 = load i8, ptr %v168, align 1
  %v170 = uitofp i8 %v169 to float
  %v171 = fmul contract float %v170, %v153
  %v172 = fadd contract float %v149, %v171
  %v173 = add i64 %v150, 1
  br label %bb26
bb33:
  %v174 = fmul contract float %v78, %v149
  %v175 = fsub contract float %v40, %v174
  br label %bb34
bb34:
  %v176 = phi float [ %v175, %bb33 ], [ %v244, %bb47 ]
  %v177 = phi i64 [ 0, %bb33 ], [ %v218, %bb47 ]
  %v178 = phi i64 [ 0, %bb33 ], [ %v245, %bb47 ]
  %v179 = phi i64 [ 0, %bb33 ], [ %v246, %bb47 ]
  %v180 = icmp ult i64 %v179, 4
  %v181 = xor i1 %v180, 1
  br i1 %v181, label %bb48, label %bb35
bb35:
  %v182 = mul i64 %v179, 32
  %v183 = add i64 %v142, %v182
  %v184 = icmp ult i64 %v177, 8
  br i1 %v184, label %bb36, label %bb80
bb36:
  %v185 = getelementptr inbounds [8 x i8], ptr %v24, i32 0, i64 %v177
  %v186 = load i8, ptr %v185, align 1
  %v187 = uitofp i8 %v186 to float
  %v188 = add i64 %v177, 1
  br label %bb37
bb37:
  %v189 = phi float [ 0.0, %bb36 ], [ %v208, %bb40 ]
  %v190 = phi i64 [ 0, %bb36 ], [ %v209, %bb40 ]
  %v191 = icmp ult i64 %v190, 32
  %v192 = xor i1 %v191, 1
  br i1 %v192, label %bb41, label %bb38
bb38:
  %v193 = add i64 %v183, %v190
  %v194 = icmp ult i64 %v193, %v47
  br i1 %v194, label %bb39, label %bb81
bb39:
  %v195 = extractvalue { ptr, i64 } %v15, 0
  %v196 = getelementptr inbounds i8, ptr %v195, i64 %v193
  %v197 = load i8, ptr %v196, align 1
  %v198 = and i8 %v197, 15
  %v199 = uitofp i8 %v198 to float
  %v200 = add i64 %v148, %v178
  %v201 = add i64 %v200, %v190
  %v202 = extractvalue { ptr, i64 } %v16, 1
  %v203 = icmp ult i64 %v201, %v202
  br i1 %v203, label %bb40, label %bb82
bb40:
  %v204 = extractvalue { ptr, i64 } %v16, 0
  %v205 = getelementptr inbounds float, ptr %v204, i64 %v201
  %v206 = load float, ptr %v205, align 4
  %v207 = fmul contract float %v199, %v206
  %v208 = fadd contract float %v189, %v207
  %v209 = add i64 %v190, 1
  br label %bb37
bb41:
  %v210 = fmul contract float %v77, %v187
  %v211 = fmul contract float %v210, %v189
  %v212 = fadd contract float %v176, %v211
  %v213 = add i64 %v178, 32
  %v214 = icmp ult i64 %v188, 8
  br i1 %v214, label %bb42, label %bb83
bb42:
  %v215 = getelementptr inbounds [8 x i8], ptr %v24, i32 0, i64 %v188
  %v216 = load i8, ptr %v215, align 1
  %v217 = uitofp i8 %v216 to float
  %v218 = add i64 %v188, 1
  br label %bb43
bb43:
  %v219 = phi float [ 0.0, %bb42 ], [ %v240, %bb46 ]
  %v220 = phi i64 [ 0, %bb42 ], [ %v241, %bb46 ]
  %v221 = icmp ult i64 %v220, 32
  %v222 = xor i1 %v221, 1
  br i1 %v222, label %bb47, label %bb44
bb44:
  %v223 = add i64 %v183, %v220
  %v224 = icmp ult i64 %v223, %v47
  br i1 %v224, label %bb45, label %bb84
bb45:
  %v225 = extractvalue { ptr, i64 } %v15, 0
  %v226 = getelementptr inbounds i8, ptr %v225, i64 %v223
  %v227 = load i8, ptr %v226, align 1
  %v228 = trunc i32 4 to i8
  %v229 = and i8 %v228, 7
  %v230 = lshr i8 %v227, %v229
  %v231 = uitofp i8 %v230 to float
  %v232 = add i64 %v148, %v213
  %v233 = add i64 %v232, %v220
  %v234 = extractvalue { ptr, i64 } %v16, 1
  %v235 = icmp ult i64 %v233, %v234
  br i1 %v235, label %bb46, label %bb85
bb46:
  %v236 = extractvalue { ptr, i64 } %v16, 0
  %v237 = getelementptr inbounds float, ptr %v236, i64 %v233
  %v238 = load float, ptr %v237, align 4
  %v239 = fmul contract float %v231, %v238
  %v240 = fadd contract float %v219, %v239
  %v241 = add i64 %v220, 1
  br label %bb43
bb47:
  %v242 = fmul contract float %v77, %v217
  %v243 = fmul contract float %v242, %v219
  %v244 = fadd contract float %v212, %v243
  %v245 = add i64 %v213, 32
  %v246 = add i64 %v179, 1
  br label %bb34
bb48:
  %v247 = add i32 %v41, 1
  br label %bb5
bb49:
  %v248 = icmp eq i64 %v27, 18446744073709551615
  br i1 %v248, label %bb57, label %bb54
bb50:
  %v249 = extractvalue { ptr } %v261, 0
  store float %v40, ptr %v249, align 4
  br label %bb52
bb51:
  br label %bb52
bb52:
  br label %bb53
bb53:
  ret void
bb54:
  %v250 = extractvalue { ptr, i64 } %v20, 1
  %v251 = icmp ult i64 %v27, %v250
  %v252 = xor i1 %v251, 1
  br i1 %v252, label %bb56, label %bb55
bb55:
  %v253 = extractvalue { ptr, i64 } %v20, 0
  %v254 = getelementptr inbounds float, ptr %v253, i64 %v27
  %v255 = insertvalue { ptr } undef, ptr %v254, 0
  %v256 = extractvalue { ptr } %v255, 0
  br label %bb58
bb56:
  br label %bb57
bb57:
  %v257 = inttoptr i64 0 to ptr
  %v258 = insertvalue { ptr } undef, ptr %v257, 0
  %v259 = extractvalue { ptr } %v258, 0
  br label %bb58
bb58:
  %v260 = phi ptr [ %v256, %bb55 ], [ %v259, %bb57 ]
  %v261 = insertvalue { ptr } undef, ptr %v260, 0
  %v262 = extractvalue { ptr } %v261, 0
  %v263 = ptrtoint ptr %v262 to i64
  %v264 = sub i64 %v263, 0
  %v265 = icmp ule i64 %v264, 0
  %v266 = add i64 %v264, 0
  %v267 = select i1 %v265, i64 %v266, i64 1
  %v268 = icmp eq i64 %v267, 1
  br i1 %v268, label %bb50, label %bb59
bb59:
  %v269 = icmp eq i64 %v267, 0
  br i1 %v269, label %bb51, label %bb60
bb60:
  unreachable
bb61:
  call void @llvm.trap() #0
  unreachable
bb62:
  call void @llvm.trap() #0
  unreachable
bb63:
  call void @llvm.trap() #0
  unreachable
bb64:
  call void @llvm.trap() #0
  unreachable
bb65:
  call void @llvm.trap() #0
  unreachable
bb66:
  call void @llvm.trap() #0
  unreachable
bb67:
  call void @llvm.trap() #0
  unreachable
bb68:
  call void @llvm.trap() #0
  unreachable
bb69:
  call void @llvm.trap() #0
  unreachable
bb70:
  call void @llvm.trap() #0
  unreachable
bb71:
  call void @llvm.trap() #0
  unreachable
bb72:
  call void @llvm.trap() #0
  unreachable
bb73:
  call void @llvm.trap() #0
  unreachable
bb74:
  call void @llvm.trap() #0
  unreachable
bb75:
  call void @llvm.trap() #0
  unreachable
bb76:
  call void @llvm.trap() #0
  unreachable
bb77:
  call void @llvm.trap() #0
  unreachable
bb78:
  call void @llvm.trap() #0
  unreachable
bb79:
  call void @llvm.trap() #0
  unreachable
bb80:
  call void @llvm.trap() #0
  unreachable
bb81:
  call void @llvm.trap() #0
  unreachable
bb82:
  call void @llvm.trap() #0
  unreachable
bb83:
  call void @llvm.trap() #0
  unreachable
bb84:
  call void @llvm.trap() #0
  unreachable
bb85:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_q8_0_project_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr %v12, i64 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v13, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi { ptr, i64 } [ %v23, %entry ]
  %v33 = alloca [2 x i8], align 1
  %v34 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v35 = zext i32 %v34 to i64
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v37 = zext i32 %v36 to i64
  %v38 = zext i32 %v28 to i64
  %v39 = zext i32 %v29 to i64
  %v40 = mul i64 %v38, %v39
  %v41 = zext i32 %v30 to i64
  %v42 = mul i64 %v40, %v41
  %v43 = icmp uge i64 %v37, %v42
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb4, label %bb3
bb3:
  br label %bb33
bb4:
  %v45 = icmp eq i64 %v41, 0
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb5, label %bb34
bb5:
  %v47 = urem i64 %v37, %v41
  %v48 = udiv i64 %v37, %v41
  %v49 = extractvalue { ptr, i64 } %v27, 1
  %v50 = icmp ult i64 %v48, %v49
  br i1 %v50, label %bb6, label %bb35
bb6:
  %v51 = extractvalue { ptr, i64 } %v27, 0
  %v52 = getelementptr inbounds i32, ptr %v51, i64 %v48
  %v53 = load i32, ptr %v52, align 4
  %v54 = zext i32 %v53 to i64
  %v55 = extractvalue { ptr, i64 } %v26, 1
  %v56 = icmp ult i64 %v54, %v55
  br i1 %v56, label %bb7, label %bb36
bb7:
  %v57 = extractvalue { ptr, i64 } %v26, 0
  %v58 = getelementptr inbounds i32, ptr %v57, i64 %v54
  %v59 = load i32, ptr %v58, align 4
  %v60 = zext i32 %v59 to i64
  %v61 = zext i32 %v31 to i64
  %v62 = udiv i64 %v61, 32
  %v63 = mul i64 %v62, 34
  br label %bb8
bb8:
  %v64 = phi float [ 0.0, %bb7 ], [ %v92, %bb17 ]
  %v65 = phi i64 [ %v35, %bb7 ], [ %v114, %bb17 ]
  %v66 = icmp ult i64 %v65, %v62
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb18, label %bb9
bb9:
  %v68 = mul i64 %v60, %v41
  %v69 = add i64 %v68, %v47
  %v70 = mul i64 %v69, %v63
  %v71 = mul i64 %v65, 34
  %v72 = add i64 %v70, %v71
  %v73 = extractvalue { ptr, i64 } %v24, 1
  %v74 = icmp ult i64 %v72, %v73
  br i1 %v74, label %bb10, label %bb37
bb10:
  %v75 = extractvalue { ptr, i64 } %v24, 0
  %v76 = getelementptr inbounds i8, ptr %v75, i64 %v72
  %v77 = load i8, ptr %v76, align 1
  %v78 = add i64 %v72, 1
  %v79 = icmp ult i64 %v78, %v73
  br i1 %v79, label %bb11, label %bb38
bb11:
  %v80 = extractvalue { ptr, i64 } %v24, 0
  %v81 = getelementptr inbounds i8, ptr %v80, i64 %v78
  %v82 = load i8, ptr %v81, align 1
  %v83 = getelementptr inbounds [2 x i8], ptr %v33, i32 0, i64 0
  store i8 %v77, ptr %v83, align 1
  %v84 = getelementptr inbounds [2 x i8], ptr %v33, i32 0, i64 1
  store i8 %v82, ptr %v84, align 1
  %v85 = load [2 x i8], ptr %v33, align 1
  %v86 = alloca [2 x i8], align 2
  store [2 x i8] %v85, ptr %v86, align 2
  %v87 = load i16, ptr %v86, align 2
  %v88 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v87) #0
  br label %bb12
bb12:
  %v89 = mul i64 %v54, %v62
  %v90 = add i64 %v89, %v65
  %v91 = mul i64 %v90, 32
  br label %bb13
bb13:
  %v92 = phi float [ %v64, %bb12 ], [ %v112, %bb16 ]
  %v93 = phi i64 [ 0, %bb12 ], [ %v113, %bb16 ]
  %v94 = icmp ult i64 %v93, 32
  %v95 = xor i1 %v94, 1
  br i1 %v95, label %bb17, label %bb14
bb14:
  %v96 = add i64 %v72, 2
  %v97 = add i64 %v96, %v93
  %v98 = icmp ult i64 %v97, %v73
  br i1 %v98, label %bb15, label %bb39
bb15:
  %v99 = extractvalue { ptr, i64 } %v24, 0
  %v100 = getelementptr inbounds i8, ptr %v99, i64 %v97
  %v101 = load i8, ptr %v100, align 1
  %v102 = bitcast i8 %v101 to i8
  %v103 = sitofp i8 %v102 to float
  %v104 = fmul contract float %v88, %v103
  %v105 = add i64 %v91, %v93
  %v106 = extractvalue { ptr, i64 } %v25, 1
  %v107 = icmp ult i64 %v105, %v106
  br i1 %v107, label %bb16, label %bb40
bb16:
  %v108 = extractvalue { ptr, i64 } %v25, 0
  %v109 = getelementptr inbounds float, ptr %v108, i64 %v105
  %v110 = load float, ptr %v109, align 4
  %v111 = fmul contract float %v104, %v110
  %v112 = fadd contract float %v92, %v111
  %v113 = add i64 %v93, 1
  br label %bb13
bb17:
  %v114 = add i64 %v65, 32
  br label %bb8
bb18:
  %v115 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_7, i64 %v35
  br label %bb19
bb19:
  store float %v64, ptr addrspace(3) %v115, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb20
bb20:
  br label %bb21
bb21:
  %v117 = phi i64 [ 16, %bb20 ], [ %v130, %bb28 ]
  %v118 = icmp ugt i64 %v117, 0
  %v119 = xor i1 %v118, 1
  br i1 %v119, label %bb29, label %bb22
bb22:
  %v120 = icmp ult i64 %v35, %v117
  %v121 = xor i1 %v120, 1
  br i1 %v121, label %bb26, label %bb23
bb23:
  %v122 = bitcast ptr addrspace(3) @__shared_mem_7 to ptr addrspace(3)
  %v123 = add i64 %v35, %v117
  %v124 = getelementptr inbounds float, ptr addrspace(3) %v122, i64 %v123
  br label %bb24
bb24:
  %v125 = load float, ptr addrspace(3) %v124, align 4
  %v126 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_7, i64 %v35
  br label %bb25
bb25:
  %v127 = load float, ptr addrspace(3) %v126, align 4
  %v128 = fadd contract float %v127, %v125
  store float %v128, ptr addrspace(3) %v126, align 4
  br label %bb27
bb26:
  br label %bb27
bb27:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb28
bb28:
  %v130 = udiv i64 %v117, 2
  br label %bb21
bb29:
  %v131 = icmp eq i64 %v35, 0
  br i1 %v131, label %bb30, label %bb32
bb30:
  %v132 = bitcast ptr addrspace(3) @__shared_mem_7 to ptr addrspace(3)
  %v133 = getelementptr inbounds float, ptr addrspace(3) %v132, i64 0
  br label %bb31
bb31:
  %v134 = load float, ptr addrspace(3) %v133, align 4
  %v135 = mul i64 %v54, %v41
  %v136 = add i64 %v135, %v47
  %v137 = extractvalue { ptr, i64 } %v32, 0
  %v138 = getelementptr inbounds float, ptr %v137, i64 %v136
  store float %v134, ptr %v138, align 4
  br label %bb32
bb32:
  br label %bb33
bb33:
  ret void
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @silu_gate(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = alloca {  }, align 1
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v16) #0
  br label %bb1
bb1:
  %v18 = icmp eq i64 %v17, 18446744073709551615
  br i1 %v18, label %bb10, label %bb7
bb2:
  %v19 = extractvalue { ptr } %v43, 0
  %v20 = extractvalue { ptr, i64 } %v12, 1
  %v21 = icmp ult i64 %v17, %v20
  br i1 %v21, label %bb3, label %bb15
bb3:
  %v22 = extractvalue { ptr, i64 } %v12, 0
  %v23 = getelementptr inbounds float, ptr %v22, i64 %v17
  %v24 = load float, ptr %v23, align 4
  %v25 = extractvalue { ptr, i64 } %v13, 1
  %v26 = icmp ult i64 %v17, %v25
  br i1 %v26, label %bb4, label %bb16
bb4:
  %v27 = extractvalue { ptr, i64 } %v13, 0
  %v28 = getelementptr inbounds float, ptr %v27, i64 %v17
  %v29 = load float, ptr %v28, align 4
  %v30 = fneg float %v24
  %v31 = call float @__nv_expf(float %v30) #0
  br label %bb13
bb5:
  br label %bb6
bb6:
  ret void
bb7:
  %v32 = extractvalue { ptr, i64 } %v14, 1
  %v33 = icmp ult i64 %v17, %v32
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb9, label %bb8
bb8:
  %v35 = extractvalue { ptr, i64 } %v14, 0
  %v36 = getelementptr inbounds float, ptr %v35, i64 %v17
  %v37 = insertvalue { ptr } undef, ptr %v36, 0
  %v38 = extractvalue { ptr } %v37, 0
  br label %bb11
bb9:
  br label %bb10
bb10:
  %v39 = inttoptr i64 0 to ptr
  %v40 = insertvalue { ptr } undef, ptr %v39, 0
  %v41 = extractvalue { ptr } %v40, 0
  br label %bb11
bb11:
  %v42 = phi ptr [ %v38, %bb8 ], [ %v41, %bb10 ]
  %v43 = insertvalue { ptr } undef, ptr %v42, 0
  %v44 = extractvalue { ptr } %v43, 0
  %v45 = ptrtoint ptr %v44 to i64
  %v46 = sub i64 %v45, 0
  %v47 = icmp ule i64 %v46, 0
  %v48 = add i64 %v46, 0
  %v49 = select i1 %v47, i64 %v48, i64 1
  %v50 = icmp eq i64 %v49, 1
  br i1 %v50, label %bb2, label %bb12
bb12:
  %v51 = icmp eq i64 %v49, 0
  br i1 %v51, label %bb5, label %bb14
bb13:
  %v52 = fadd contract float 1.0, %v31
  %v53 = fdiv contract float %v24, %v52
  %v54 = fmul contract float %v53, %v29
  store float %v54, ptr %v19, align 4
  br label %bb6
bb14:
  unreachable
bb15:
  call void @llvm.trap() #0
  unreachable
bb16:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_q8_0_row(ptr %v0, i64 %v1, i32 %v2, i32 %v3, i32 %v4, ptr %v5, i64 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v5, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v6, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi i32 [ %v2, %entry ]
  %v13 = phi i32 [ %v3, %entry ]
  %v14 = phi i32 [ %v4, %entry ]
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = alloca {  }, align 1
  %v17 = alloca [2 x i8], align 1
  %v18 = bitcast ptr %v16 to ptr
  %v19 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v18) #0
  br label %bb1
bb1:
  %v20 = zext i32 %v13 to i64
  %v21 = icmp uge i64 %v19, %v20
  %v22 = xor i1 %v21, 1
  br i1 %v22, label %bb3, label %bb2
bb2:
  br label %bb11
bb3:
  %v23 = zext i32 %v14 to i64
  %v24 = mul i64 %v23, 34
  %v25 = udiv i64 %v19, 32
  %v26 = urem i64 %v19, 32
  %v27 = zext i32 %v12 to i64
  %v28 = mul i64 %v27, %v24
  %v29 = mul i64 %v25, 34
  %v30 = add i64 %v28, %v29
  %v31 = extractvalue { ptr, i64 } %v11, 1
  %v32 = icmp ult i64 %v30, %v31
  br i1 %v32, label %bb4, label %bb19
bb4:
  %v33 = extractvalue { ptr, i64 } %v11, 0
  %v34 = getelementptr inbounds i8, ptr %v33, i64 %v30
  %v35 = load i8, ptr %v34, align 1
  %v36 = add i64 %v30, 1
  %v37 = icmp ult i64 %v36, %v31
  br i1 %v37, label %bb5, label %bb20
bb5:
  %v38 = extractvalue { ptr, i64 } %v11, 0
  %v39 = getelementptr inbounds i8, ptr %v38, i64 %v36
  %v40 = load i8, ptr %v39, align 1
  %v41 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 0
  store i8 %v35, ptr %v41, align 1
  %v42 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 1
  store i8 %v40, ptr %v42, align 1
  %v43 = load [2 x i8], ptr %v17, align 1
  %v44 = alloca [2 x i8], align 2
  store [2 x i8] %v43, ptr %v44, align 2
  %v45 = load i16, ptr %v44, align 2
  %v46 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v45) #0
  br label %bb6
bb6:
  %v47 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v47, label %bb15, label %bb12
bb7:
  %v48 = extractvalue { ptr } %v69, 0
  %v49 = add i64 %v30, 2
  %v50 = add i64 %v49, %v26
  %v51 = icmp ult i64 %v50, %v31
  br i1 %v51, label %bb8, label %bb21
bb8:
  %v52 = extractvalue { ptr, i64 } %v11, 0
  %v53 = getelementptr inbounds i8, ptr %v52, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v55 = bitcast i8 %v54 to i8
  %v56 = sitofp i8 %v55 to float
  %v57 = fmul contract float %v46, %v56
  store float %v57, ptr %v48, align 4
  br label %bb10
bb9:
  br label %bb10
bb10:
  br label %bb11
bb11:
  ret void
bb12:
  %v58 = extractvalue { ptr, i64 } %v15, 1
  %v59 = icmp ult i64 %v19, %v58
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb14, label %bb13
bb13:
  %v61 = extractvalue { ptr, i64 } %v15, 0
  %v62 = getelementptr inbounds float, ptr %v61, i64 %v19
  %v63 = insertvalue { ptr } undef, ptr %v62, 0
  %v64 = extractvalue { ptr } %v63, 0
  br label %bb16
bb14:
  br label %bb15
bb15:
  %v65 = inttoptr i64 0 to ptr
  %v66 = insertvalue { ptr } undef, ptr %v65, 0
  %v67 = extractvalue { ptr } %v66, 0
  br label %bb16
bb16:
  %v68 = phi ptr [ %v64, %bb13 ], [ %v67, %bb15 ]
  %v69 = insertvalue { ptr } undef, ptr %v68, 0
  %v70 = extractvalue { ptr } %v69, 0
  %v71 = ptrtoint ptr %v70 to i64
  %v72 = sub i64 %v71, 0
  %v73 = icmp ule i64 %v72, 0
  %v74 = add i64 %v72, 0
  %v75 = select i1 %v73, i64 %v74, i64 1
  %v76 = icmp eq i64 %v75, 1
  br i1 %v76, label %bb7, label %bb17
bb17:
  %v77 = icmp eq i64 %v75, 0
  br i1 %v77, label %bb9, label %bb18
bb18:
  unreachable
bb19:
  call void @llvm.trap() #0
  unreachable
bb20:
  call void @llvm.trap() #0
  unreachable
bb21:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_q4k_project(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, ptr %v13, i64 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v14, 1
  br label %bb0
bb0:
  %v23 = phi { ptr, i64 } [ %v16, %entry ]
  %v24 = phi { ptr, i64 } [ %v18, %entry ]
  %v25 = phi { ptr, i64 } [ %v20, %entry ]
  %v26 = phi i32 [ %v6, %entry ]
  %v27 = phi i32 [ %v7, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi { ptr, i64 } [ %v22, %entry ]
  %v34 = alloca {  }, align 1
  %v35 = bitcast ptr %v34 to ptr
  %v36 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v35) #0
  br label %bb1
bb1:
  %v37 = zext i32 %v26 to i64
  %v38 = zext i32 %v27 to i64
  %v39 = mul i64 %v37, %v38
  %v40 = zext i32 %v28 to i64
  %v41 = mul i64 %v39, %v40
  %v42 = icmp uge i64 %v36, %v41
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb3, label %bb2
bb2:
  br label %bb15
bb3:
  %v44 = icmp eq i64 %v40, 0
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb4, label %bb23
bb4:
  %v46 = urem i64 %v36, %v40
  %v47 = udiv i64 %v36, %v40
  %v48 = icmp eq i64 %v38, 0
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb5, label %bb24
bb5:
  %v50 = udiv i64 %v47, %v38
  %v51 = extractvalue { ptr, i64 } %v25, 1
  %v52 = icmp ult i64 %v47, %v51
  br i1 %v52, label %bb6, label %bb25
bb6:
  %v53 = extractvalue { ptr, i64 } %v25, 0
  %v54 = getelementptr inbounds i32, ptr %v53, i64 %v47
  %v55 = load i32, ptr %v54, align 4
  %v56 = zext i32 %v55 to i64
  %v57 = udiv i32 %v29, 256
  %v58 = zext i32 %v57 to i64
  %v59 = mul i64 %v58, 144
  %v60 = zext i32 %v30 to i64
  %v61 = mul i64 %v60, %v59
  %v62 = icmp eq i32 %v32, 0
  br i1 %v62, label %bb8, label %bb7
bb7:
  %v63 = zext i32 %v29 to i64
  %v64 = mul i64 %v47, %v63
  br label %bb9
bb8:
  %v65 = zext i32 %v29 to i64
  %v66 = mul i64 %v50, %v65
  br label %bb9
bb9:
  %v67 = phi i64 [ %v64, %bb7 ], [ %v66, %bb8 ]
  %v68 = mul i64 %v56, %v61
  %v69 = zext i32 %v31 to i64
  %v70 = add i64 %v69, %v46
  %v71 = mul i64 %v70, %v59
  %v72 = add i64 %v68, %v71
  %v73 = extractvalue { ptr, i64 } %v23, 0
  %v74 = extractvalue { ptr, i64 } %v23, 1
  %v75 = extractvalue { ptr, i64 } %v24, 0
  %v76 = extractvalue { ptr, i64 } %v24, 1
  %v77 = call float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v73, i64 %v74, i64 %v72, ptr %v75, i64 %v76, i64 %v67, i32 %v57) #0
  br label %bb10
bb10:
  %v78 = bitcast ptr %v34 to ptr
  %v79 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v78) #0
  br label %bb11
bb11:
  %v80 = icmp eq i64 %v79, 18446744073709551615
  br i1 %v80, label %bb19, label %bb16
bb12:
  %v81 = extractvalue { ptr } %v93, 0
  store float %v77, ptr %v81, align 4
  br label %bb14
bb13:
  br label %bb14
bb14:
  br label %bb15
bb15:
  ret void
bb16:
  %v82 = extractvalue { ptr, i64 } %v33, 1
  %v83 = icmp ult i64 %v79, %v82
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb18, label %bb17
bb17:
  %v85 = extractvalue { ptr, i64 } %v33, 0
  %v86 = getelementptr inbounds float, ptr %v85, i64 %v79
  %v87 = insertvalue { ptr } undef, ptr %v86, 0
  %v88 = extractvalue { ptr } %v87, 0
  br label %bb20
bb18:
  br label %bb19
bb19:
  %v89 = inttoptr i64 0 to ptr
  %v90 = insertvalue { ptr } undef, ptr %v89, 0
  %v91 = extractvalue { ptr } %v90, 0
  br label %bb20
bb20:
  %v92 = phi ptr [ %v88, %bb17 ], [ %v91, %bb19 ]
  %v93 = insertvalue { ptr } undef, ptr %v92, 0
  %v94 = extractvalue { ptr } %v93, 0
  %v95 = ptrtoint ptr %v94 to i64
  %v96 = sub i64 %v95, 0
  %v97 = icmp ule i64 %v96, 0
  %v98 = add i64 %v96, 0
  %v99 = select i1 %v97, i64 %v98, i64 1
  %v100 = icmp eq i64 %v99, 1
  br i1 %v100, label %bb12, label %bb21
bb21:
  %v101 = icmp eq i64 %v99, 0
  br i1 %v101, label %bb13, label %bb22
bb22:
  unreachable
bb23:
  call void @llvm.trap() #0
  unreachable
bb24:
  call void @llvm.trap() #0
  unreachable
bb25:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_q6k_rows(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = bitcast ptr %v21 to ptr
  %v24 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v23) #0
  br label %bb1
bb1:
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v18 to i64
  %v27 = mul i64 %v25, %v26
  %v28 = icmp uge i64 %v24, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb3, label %bb2
bb2:
  br label %bb23
bb3:
  %v30 = icmp eq i64 %v26, 0
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb4, label %bb31
bb4:
  %v32 = udiv i64 %v24, %v26
  %v33 = urem i64 %v24, %v26
  %v34 = zext i32 %v19 to i64
  %v35 = mul i64 %v34, 210
  %v36 = udiv i64 %v33, 256
  %v37 = urem i64 %v33, 256
  %v38 = extractvalue { ptr, i64 } %v16, 1
  %v39 = icmp ult i64 %v32, %v38
  br i1 %v39, label %bb5, label %bb32
bb5:
  %v40 = extractvalue { ptr, i64 } %v16, 0
  %v41 = getelementptr inbounds i32, ptr %v40, i64 %v32
  %v42 = load i32, ptr %v41, align 4
  %v43 = zext i32 %v42 to i64
  %v44 = mul i64 %v43, %v35
  %v45 = mul i64 %v36, 210
  %v46 = add i64 %v44, %v45
  %v47 = add i64 %v46, 208
  %v48 = extractvalue { ptr, i64 } %v15, 1
  %v49 = icmp ult i64 %v47, %v48
  br i1 %v49, label %bb6, label %bb33
bb6:
  %v50 = extractvalue { ptr, i64 } %v15, 0
  %v51 = getelementptr inbounds i8, ptr %v50, i64 %v47
  %v52 = load i8, ptr %v51, align 1
  %v53 = add i64 %v46, 209
  %v54 = icmp ult i64 %v53, %v48
  br i1 %v54, label %bb7, label %bb34
bb7:
  %v55 = extractvalue { ptr, i64 } %v15, 0
  %v56 = getelementptr inbounds i8, ptr %v55, i64 %v53
  %v57 = load i8, ptr %v56, align 1
  %v58 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v52, ptr %v58, align 1
  %v59 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v57, ptr %v59, align 1
  %v60 = load [2 x i8], ptr %v22, align 1
  %v61 = alloca [2 x i8], align 2
  store [2 x i8] %v60, ptr %v61, align 2
  %v62 = load i16, ptr %v61, align 2
  %v63 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v62) #0
  br label %bb8
bb8:
  %v64 = udiv i64 %v37, 128
  %v65 = urem i64 %v37, 128
  %v66 = mul i64 %v64, 64
  %v67 = add i64 %v46, %v66
  %v68 = add i64 %v46, 128
  %v69 = mul i64 %v64, 32
  %v70 = add i64 %v68, %v69
  %v71 = add i64 %v46, 192
  %v72 = mul i64 %v64, 8
  %v73 = add i64 %v71, %v72
  %v74 = urem i64 %v65, 32
  %v75 = udiv i64 %v65, 32
  %v76 = trunc i64 %v75 to i32
  %v77 = mul i32 %v76, 2
  %v78 = icmp eq i64 %v75, 0
  br i1 %v78, label %bb10, label %bb9
bb9:
  %v79 = icmp eq i64 %v75, 2
  br i1 %v79, label %bb10, label %bb11
bb10:
  br label %bb12
bb11:
  %v80 = add i64 %v74, 32
  br label %bb12
bb12:
  %v81 = phi i64 [ %v74, %bb10 ], [ %v80, %bb11 ]
  %v82 = icmp ult i64 %v75, 2
  %v83 = xor i1 %v82, 1
  br i1 %v83, label %bb15, label %bb13
bb13:
  %v84 = add i64 %v67, %v81
  %v85 = icmp ult i64 %v84, %v48
  br i1 %v85, label %bb14, label %bb35
bb14:
  %v86 = extractvalue { ptr, i64 } %v15, 0
  %v87 = getelementptr inbounds i8, ptr %v86, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = and i8 %v88, 15
  %v90 = zext i8 %v89 to i32
  br label %bb17
bb15:
  %v91 = add i64 %v67, %v81
  %v92 = icmp ult i64 %v91, %v48
  br i1 %v92, label %bb16, label %bb36
bb16:
  %v93 = extractvalue { ptr, i64 } %v15, 0
  %v94 = getelementptr inbounds i8, ptr %v93, i64 %v91
  %v95 = load i8, ptr %v94, align 1
  %v96 = trunc i32 4 to i8
  %v97 = and i8 %v96, 7
  %v98 = lshr i8 %v95, %v97
  %v99 = zext i8 %v98 to i32
  br label %bb17
bb17:
  %v100 = phi i32 [ %v90, %bb14 ], [ %v99, %bb16 ]
  %v101 = add i64 %v70, %v74
  %v102 = icmp ult i64 %v101, %v48
  br i1 %v102, label %bb18, label %bb37
bb18:
  %v103 = extractvalue { ptr, i64 } %v15, 0
  %v104 = getelementptr inbounds i8, ptr %v103, i64 %v101
  %v105 = load i8, ptr %v104, align 1
  %v106 = trunc i32 %v77 to i8
  %v107 = and i8 %v106, 7
  %v108 = lshr i8 %v105, %v107
  %v109 = and i8 %v108, 3
  %v110 = zext i8 %v109 to i32
  %v111 = and i32 4, 31
  %v112 = shl i32 %v110, %v111
  %v113 = or i32 %v100, %v112
  %v114 = sub i32 %v113, 32
  %v115 = icmp eq i64 %v24, 18446744073709551615
  br i1 %v115, label %bb27, label %bb24
bb19:
  %v116 = extractvalue { ptr } %v141, 0
  %v117 = udiv i64 %v74, 16
  %v118 = add i64 %v73, %v117
  %v119 = mul i64 %v75, 2
  %v120 = add i64 %v118, %v119
  %v121 = icmp ult i64 %v120, %v48
  br i1 %v121, label %bb20, label %bb38
bb20:
  %v122 = extractvalue { ptr, i64 } %v15, 0
  %v123 = getelementptr inbounds i8, ptr %v122, i64 %v120
  %v124 = load i8, ptr %v123, align 1
  %v125 = bitcast i8 %v124 to i8
  %v126 = sitofp i8 %v125 to float
  %v127 = fmul contract float %v63, %v126
  %v128 = sitofp i32 %v114 to float
  %v129 = fmul contract float %v127, %v128
  store float %v129, ptr %v116, align 4
  br label %bb22
bb21:
  br label %bb22
bb22:
  br label %bb23
bb23:
  ret void
bb24:
  %v130 = extractvalue { ptr, i64 } %v20, 1
  %v131 = icmp ult i64 %v24, %v130
  %v132 = xor i1 %v131, 1
  br i1 %v132, label %bb26, label %bb25
bb25:
  %v133 = extractvalue { ptr, i64 } %v20, 0
  %v134 = getelementptr inbounds float, ptr %v133, i64 %v24
  %v135 = insertvalue { ptr } undef, ptr %v134, 0
  %v136 = extractvalue { ptr } %v135, 0
  br label %bb28
bb26:
  br label %bb27
bb27:
  %v137 = inttoptr i64 0 to ptr
  %v138 = insertvalue { ptr } undef, ptr %v137, 0
  %v139 = extractvalue { ptr } %v138, 0
  br label %bb28
bb28:
  %v140 = phi ptr [ %v136, %bb25 ], [ %v139, %bb27 ]
  %v141 = insertvalue { ptr } undef, ptr %v140, 0
  %v142 = extractvalue { ptr } %v141, 0
  %v143 = ptrtoint ptr %v142 to i64
  %v144 = sub i64 %v143, 0
  %v145 = icmp ule i64 %v144, 0
  %v146 = add i64 %v144, 0
  %v147 = select i1 %v145, i64 %v146, i64 1
  %v148 = icmp eq i64 %v147, 1
  br i1 %v148, label %bb19, label %bb29
bb29:
  %v149 = icmp eq i64 %v147, 0
  br i1 %v149, label %bb21, label %bb30
bb30:
  unreachable
bb31:
  call void @llvm.trap() #0
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q6k_gemv_row(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = bitcast ptr %v21 to ptr
  %v24 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v23) #0
  br label %bb1
bb1:
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v19 to i64
  %v27 = mul i64 %v25, %v26
  %v28 = icmp uge i64 %v24, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb3, label %bb2
bb2:
  br label %bb36
bb3:
  %v30 = icmp eq i64 %v25, 0
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb4, label %bb44
bb4:
  %v32 = urem i64 %v24, %v25
  %v33 = udiv i64 %v24, %v25
  %v34 = mul i32 %v18, 210
  %v35 = zext i32 %v34 to i64
  %v36 = mul i64 %v32, %v35
  br label %bb5
bb5:
  %v37 = phi float [ 0.0, %bb4 ], [ %v67, %bb31 ]
  %v38 = phi i32 [ 0, %bb4 ], [ %v247, %bb31 ]
  %v39 = icmp ult i32 %v38, %v18
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb32, label %bb6
bb6:
  %v41 = zext i32 %v38 to i64
  %v42 = mul i64 %v41, 210
  %v43 = add i64 %v36, %v42
  %v44 = add i64 %v43, 208
  %v45 = extractvalue { ptr, i64 } %v15, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb7, label %bb45
bb7:
  %v47 = extractvalue { ptr, i64 } %v15, 0
  %v48 = getelementptr inbounds i8, ptr %v47, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v50 = add i64 %v43, 209
  %v51 = icmp ult i64 %v50, %v45
  br i1 %v51, label %bb8, label %bb46
bb8:
  %v52 = extractvalue { ptr, i64 } %v15, 0
  %v53 = getelementptr inbounds i8, ptr %v52, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v55 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v49, ptr %v55, align 1
  %v56 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v54, ptr %v56, align 1
  %v57 = load [2 x i8], ptr %v22, align 1
  %v58 = alloca [2 x i8], align 2
  store [2 x i8] %v57, ptr %v58, align 2
  %v59 = load i16, ptr %v58, align 2
  %v60 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v59) #0
  br label %bb9
bb9:
  %v61 = zext i32 %v18 to i64
  %v62 = mul i64 %v33, %v61
  %v63 = mul i64 %v62, 256
  %v64 = zext i32 %v38 to i64
  %v65 = mul i64 %v64, 256
  %v66 = add i64 %v63, %v65
  br label %bb10
bb10:
  %v67 = phi float [ %v37, %bb9 ], [ %v81, %bb30 ]
  %v68 = phi i64 [ 0, %bb9 ], [ %v246, %bb30 ]
  %v69 = icmp ult i64 %v68, 2
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb31, label %bb11
bb11:
  %v71 = mul i64 %v68, 64
  %v72 = add i64 %v43, %v71
  %v73 = add i64 %v43, 128
  %v74 = mul i64 %v68, 32
  %v75 = add i64 %v73, %v74
  %v76 = add i64 %v43, 192
  %v77 = mul i64 %v68, 8
  %v78 = add i64 %v76, %v77
  %v79 = mul i64 %v68, 128
  %v80 = add i64 %v66, %v79
  br label %bb12
bb12:
  %v81 = phi float [ %v67, %bb11 ], [ %v244, %bb29 ]
  %v82 = phi i64 [ 0, %bb11 ], [ %v245, %bb29 ]
  %v83 = icmp ult i64 %v82, 32
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb30, label %bb13
bb13:
  %v85 = udiv i64 %v82, 16
  %v86 = add i64 %v72, %v82
  %v87 = icmp ult i64 %v86, %v45
  br i1 %v87, label %bb14, label %bb47
bb14:
  %v88 = extractvalue { ptr, i64 } %v15, 0
  %v89 = getelementptr inbounds i8, ptr %v88, i64 %v86
  %v90 = load i8, ptr %v89, align 1
  %v91 = and i8 %v90, 15
  %v92 = zext i8 %v91 to i32
  %v93 = add i64 %v75, %v82
  %v94 = icmp ult i64 %v93, %v45
  br i1 %v94, label %bb15, label %bb48
bb15:
  %v95 = extractvalue { ptr, i64 } %v15, 0
  %v96 = getelementptr inbounds i8, ptr %v95, i64 %v93
  %v97 = load i8, ptr %v96, align 1
  %v98 = and i8 %v97, 3
  %v99 = zext i8 %v98 to i32
  %v100 = and i32 4, 31
  %v101 = shl i32 %v99, %v100
  %v102 = or i32 %v92, %v101
  %v103 = sub i32 %v102, 32
  %v104 = add i64 %v72, %v82
  %v105 = add i64 %v104, 32
  %v106 = icmp ult i64 %v105, %v45
  br i1 %v106, label %bb16, label %bb49
bb16:
  %v107 = extractvalue { ptr, i64 } %v15, 0
  %v108 = getelementptr inbounds i8, ptr %v107, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v110 = and i8 %v109, 15
  %v111 = zext i8 %v110 to i32
  %v112 = add i64 %v75, %v82
  %v113 = icmp ult i64 %v112, %v45
  br i1 %v113, label %bb17, label %bb50
bb17:
  %v114 = extractvalue { ptr, i64 } %v15, 0
  %v115 = getelementptr inbounds i8, ptr %v114, i64 %v112
  %v116 = load i8, ptr %v115, align 1
  %v117 = trunc i32 2 to i8
  %v118 = and i8 %v117, 7
  %v119 = lshr i8 %v116, %v118
  %v120 = and i8 %v119, 3
  %v121 = zext i8 %v120 to i32
  %v122 = and i32 4, 31
  %v123 = shl i32 %v121, %v122
  %v124 = or i32 %v111, %v123
  %v125 = sub i32 %v124, 32
  %v126 = add i64 %v72, %v82
  %v127 = icmp ult i64 %v126, %v45
  br i1 %v127, label %bb18, label %bb51
bb18:
  %v128 = extractvalue { ptr, i64 } %v15, 0
  %v129 = getelementptr inbounds i8, ptr %v128, i64 %v126
  %v130 = load i8, ptr %v129, align 1
  %v131 = trunc i32 4 to i8
  %v132 = and i8 %v131, 7
  %v133 = lshr i8 %v130, %v132
  %v134 = zext i8 %v133 to i32
  %v135 = add i64 %v75, %v82
  %v136 = icmp ult i64 %v135, %v45
  br i1 %v136, label %bb19, label %bb52
bb19:
  %v137 = extractvalue { ptr, i64 } %v15, 0
  %v138 = getelementptr inbounds i8, ptr %v137, i64 %v135
  %v139 = load i8, ptr %v138, align 1
  %v140 = trunc i32 4 to i8
  %v141 = and i8 %v140, 7
  %v142 = lshr i8 %v139, %v141
  %v143 = and i8 %v142, 3
  %v144 = zext i8 %v143 to i32
  %v145 = and i32 4, 31
  %v146 = shl i32 %v144, %v145
  %v147 = or i32 %v134, %v146
  %v148 = sub i32 %v147, 32
  %v149 = add i64 %v72, %v82
  %v150 = add i64 %v149, 32
  %v151 = icmp ult i64 %v150, %v45
  br i1 %v151, label %bb20, label %bb53
bb20:
  %v152 = extractvalue { ptr, i64 } %v15, 0
  %v153 = getelementptr inbounds i8, ptr %v152, i64 %v150
  %v154 = load i8, ptr %v153, align 1
  %v155 = trunc i32 4 to i8
  %v156 = and i8 %v155, 7
  %v157 = lshr i8 %v154, %v156
  %v158 = zext i8 %v157 to i32
  %v159 = add i64 %v75, %v82
  %v160 = icmp ult i64 %v159, %v45
  br i1 %v160, label %bb21, label %bb54
bb21:
  %v161 = extractvalue { ptr, i64 } %v15, 0
  %v162 = getelementptr inbounds i8, ptr %v161, i64 %v159
  %v163 = load i8, ptr %v162, align 1
  %v164 = trunc i32 6 to i8
  %v165 = and i8 %v164, 7
  %v166 = lshr i8 %v163, %v165
  %v167 = and i8 %v166, 3
  %v168 = zext i8 %v167 to i32
  %v169 = and i32 4, 31
  %v170 = shl i32 %v168, %v169
  %v171 = or i32 %v158, %v170
  %v172 = sub i32 %v171, 32
  %v173 = add i64 %v78, %v85
  %v174 = icmp ult i64 %v173, %v45
  br i1 %v174, label %bb22, label %bb55
bb22:
  %v175 = extractvalue { ptr, i64 } %v15, 0
  %v176 = getelementptr inbounds i8, ptr %v175, i64 %v173
  %v177 = load i8, ptr %v176, align 1
  %v178 = bitcast i8 %v177 to i8
  %v179 = sitofp i8 %v178 to float
  %v180 = add i64 %v173, 2
  %v181 = icmp ult i64 %v180, %v45
  br i1 %v181, label %bb23, label %bb56
bb23:
  %v182 = extractvalue { ptr, i64 } %v15, 0
  %v183 = getelementptr inbounds i8, ptr %v182, i64 %v180
  %v184 = load i8, ptr %v183, align 1
  %v185 = bitcast i8 %v184 to i8
  %v186 = sitofp i8 %v185 to float
  %v187 = add i64 %v173, 4
  %v188 = icmp ult i64 %v187, %v45
  br i1 %v188, label %bb24, label %bb57
bb24:
  %v189 = extractvalue { ptr, i64 } %v15, 0
  %v190 = getelementptr inbounds i8, ptr %v189, i64 %v187
  %v191 = load i8, ptr %v190, align 1
  %v192 = bitcast i8 %v191 to i8
  %v193 = sitofp i8 %v192 to float
  %v194 = add i64 %v173, 6
  %v195 = icmp ult i64 %v194, %v45
  br i1 %v195, label %bb25, label %bb58
bb25:
  %v196 = extractvalue { ptr, i64 } %v15, 0
  %v197 = getelementptr inbounds i8, ptr %v196, i64 %v194
  %v198 = load i8, ptr %v197, align 1
  %v199 = bitcast i8 %v198 to i8
  %v200 = sitofp i8 %v199 to float
  %v201 = fmul contract float %v60, %v179
  %v202 = sitofp i32 %v103 to float
  %v203 = fmul contract float %v201, %v202
  %v204 = add i64 %v80, %v82
  %v205 = extractvalue { ptr, i64 } %v16, 1
  %v206 = icmp ult i64 %v204, %v205
  br i1 %v206, label %bb26, label %bb59
bb26:
  %v207 = extractvalue { ptr, i64 } %v16, 0
  %v208 = getelementptr inbounds float, ptr %v207, i64 %v204
  %v209 = load float, ptr %v208, align 4
  %v210 = fmul contract float %v203, %v209
  %v211 = fadd contract float %v81, %v210
  %v212 = fmul contract float %v60, %v186
  %v213 = sitofp i32 %v125 to float
  %v214 = fmul contract float %v212, %v213
  %v215 = add i64 %v80, %v82
  %v216 = add i64 %v215, 32
  %v217 = icmp ult i64 %v216, %v205
  br i1 %v217, label %bb27, label %bb60
bb27:
  %v218 = extractvalue { ptr, i64 } %v16, 0
  %v219 = getelementptr inbounds float, ptr %v218, i64 %v216
  %v220 = load float, ptr %v219, align 4
  %v221 = fmul contract float %v214, %v220
  %v222 = fadd contract float %v211, %v221
  %v223 = fmul contract float %v60, %v193
  %v224 = sitofp i32 %v148 to float
  %v225 = fmul contract float %v223, %v224
  %v226 = add i64 %v80, %v82
  %v227 = add i64 %v226, 64
  %v228 = icmp ult i64 %v227, %v205
  br i1 %v228, label %bb28, label %bb61
bb28:
  %v229 = extractvalue { ptr, i64 } %v16, 0
  %v230 = getelementptr inbounds float, ptr %v229, i64 %v227
  %v231 = load float, ptr %v230, align 4
  %v232 = fmul contract float %v225, %v231
  %v233 = fadd contract float %v222, %v232
  %v234 = fmul contract float %v60, %v200
  %v235 = sitofp i32 %v172 to float
  %v236 = fmul contract float %v234, %v235
  %v237 = add i64 %v80, %v82
  %v238 = add i64 %v237, 96
  %v239 = icmp ult i64 %v238, %v205
  br i1 %v239, label %bb29, label %bb62
bb29:
  %v240 = extractvalue { ptr, i64 } %v16, 0
  %v241 = getelementptr inbounds float, ptr %v240, i64 %v238
  %v242 = load float, ptr %v241, align 4
  %v243 = fmul contract float %v236, %v242
  %v244 = fadd contract float %v233, %v243
  %v245 = add i64 %v82, 1
  br label %bb12
bb30:
  %v246 = add i64 %v68, 1
  br label %bb10
bb31:
  %v247 = add i32 %v38, 1
  br label %bb5
bb32:
  %v248 = icmp eq i64 %v24, 18446744073709551615
  br i1 %v248, label %bb40, label %bb37
bb33:
  %v249 = extractvalue { ptr } %v261, 0
  store float %v37, ptr %v249, align 4
  br label %bb35
bb34:
  br label %bb35
bb35:
  br label %bb36
bb36:
  ret void
bb37:
  %v250 = extractvalue { ptr, i64 } %v20, 1
  %v251 = icmp ult i64 %v24, %v250
  %v252 = xor i1 %v251, 1
  br i1 %v252, label %bb39, label %bb38
bb38:
  %v253 = extractvalue { ptr, i64 } %v20, 0
  %v254 = getelementptr inbounds float, ptr %v253, i64 %v24
  %v255 = insertvalue { ptr } undef, ptr %v254, 0
  %v256 = extractvalue { ptr } %v255, 0
  br label %bb41
bb39:
  br label %bb40
bb40:
  %v257 = inttoptr i64 0 to ptr
  %v258 = insertvalue { ptr } undef, ptr %v257, 0
  %v259 = extractvalue { ptr } %v258, 0
  br label %bb41
bb41:
  %v260 = phi ptr [ %v256, %bb38 ], [ %v259, %bb40 ]
  %v261 = insertvalue { ptr } undef, ptr %v260, 0
  %v262 = extractvalue { ptr } %v261, 0
  %v263 = ptrtoint ptr %v262 to i64
  %v264 = sub i64 %v263, 0
  %v265 = icmp ule i64 %v264, 0
  %v266 = add i64 %v264, 0
  %v267 = select i1 %v265, i64 %v266, i64 1
  %v268 = icmp eq i64 %v267, 1
  br i1 %v268, label %bb33, label %bb42
bb42:
  %v269 = icmp eq i64 %v267, 0
  br i1 %v269, label %bb34, label %bb43
bb43:
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
bb57:
  call void @llvm.trap() #0
  unreachable
bb58:
  call void @llvm.trap() #0
  unreachable
bb59:
  call void @llvm.trap() #0
  unreachable
bb60:
  call void @llvm.trap() #0
  unreachable
bb61:
  call void @llvm.trap() #0
  unreachable
bb62:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @scale_f32(float %v0, ptr %v1, i64 %v2, ptr %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v1, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi float [ %v0, %entry ]
  %v10 = phi { ptr, i64 } [ %v6, %entry ]
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = alloca {  }, align 1
  %v13 = bitcast ptr %v12 to ptr
  %v14 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v13) #0
  br label %bb1
bb1:
  %v15 = icmp eq i64 %v14, 18446744073709551615
  br i1 %v15, label %bb9, label %bb6
bb2:
  %v16 = extractvalue { ptr } %v34, 0
  %v17 = extractvalue { ptr, i64 } %v10, 1
  %v18 = icmp ult i64 %v14, %v17
  br i1 %v18, label %bb3, label %bb13
bb3:
  %v19 = extractvalue { ptr, i64 } %v10, 0
  %v20 = getelementptr inbounds float, ptr %v19, i64 %v14
  %v21 = load float, ptr %v20, align 4
  %v22 = fmul contract float %v21, %v9
  store float %v22, ptr %v16, align 4
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v23 = extractvalue { ptr, i64 } %v11, 1
  %v24 = icmp ult i64 %v14, %v23
  %v25 = xor i1 %v24, 1
  br i1 %v25, label %bb8, label %bb7
bb7:
  %v26 = extractvalue { ptr, i64 } %v11, 0
  %v27 = getelementptr inbounds float, ptr %v26, i64 %v14
  %v28 = insertvalue { ptr } undef, ptr %v27, 0
  %v29 = extractvalue { ptr } %v28, 0
  br label %bb10
bb8:
  br label %bb9
bb9:
  %v30 = inttoptr i64 0 to ptr
  %v31 = insertvalue { ptr } undef, ptr %v30, 0
  %v32 = extractvalue { ptr } %v31, 0
  br label %bb10
bb10:
  %v33 = phi ptr [ %v29, %bb7 ], [ %v32, %bb9 ]
  %v34 = insertvalue { ptr } undef, ptr %v33, 0
  %v35 = extractvalue { ptr } %v34, 0
  %v36 = ptrtoint ptr %v35 to i64
  %v37 = sub i64 %v36, 0
  %v38 = icmp ule i64 %v37, 0
  %v39 = add i64 %v37, 0
  %v40 = select i1 %v38, i64 %v39, i64 1
  %v41 = icmp eq i64 %v40, 1
  br i1 %v41, label %bb2, label %bb11
bb11:
  %v42 = icmp eq i64 %v40, 0
  br i1 %v42, label %bb4, label %bb12
bb12:
  unreachable
bb13:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q8_0_gemm_element(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = bitcast ptr %v21 to ptr
  %v24 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v23) #0
  br label %bb1
bb1:
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v19 to i64
  %v27 = mul i64 %v25, %v26
  %v28 = icmp uge i64 %v24, %v27
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb3, label %bb2
bb2:
  br label %bb19
bb3:
  %v30 = icmp eq i64 %v25, 0
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb4, label %bb27
bb4:
  %v32 = urem i64 %v24, %v25
  %v33 = udiv i64 %v24, %v25
  %v34 = zext i32 %v18 to i64
  %v35 = mul i64 %v34, 34
  br label %bb5
bb5:
  %v36 = phi float [ 0.0, %bb4 ], [ %v62, %bb14 ]
  %v37 = phi i64 [ 0, %bb4 ], [ %v84, %bb14 ]
  %v38 = icmp ult i64 %v37, %v34
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb15, label %bb6
bb6:
  %v40 = mul i64 %v32, %v35
  %v41 = mul i64 %v37, 34
  %v42 = add i64 %v40, %v41
  %v43 = extractvalue { ptr, i64 } %v15, 1
  %v44 = icmp ult i64 %v42, %v43
  br i1 %v44, label %bb7, label %bb28
bb7:
  %v45 = extractvalue { ptr, i64 } %v15, 0
  %v46 = getelementptr inbounds i8, ptr %v45, i64 %v42
  %v47 = load i8, ptr %v46, align 1
  %v48 = add i64 %v42, 1
  %v49 = icmp ult i64 %v48, %v43
  br i1 %v49, label %bb8, label %bb29
bb8:
  %v50 = extractvalue { ptr, i64 } %v15, 0
  %v51 = getelementptr inbounds i8, ptr %v50, i64 %v48
  %v52 = load i8, ptr %v51, align 1
  %v53 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v47, ptr %v53, align 1
  %v54 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v52, ptr %v54, align 1
  %v55 = load [2 x i8], ptr %v22, align 1
  %v56 = alloca [2 x i8], align 2
  store [2 x i8] %v55, ptr %v56, align 2
  %v57 = load i16, ptr %v56, align 2
  %v58 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v57) #0
  br label %bb9
bb9:
  %v59 = mul i64 %v33, %v34
  %v60 = add i64 %v59, %v37
  %v61 = mul i64 %v60, 32
  br label %bb10
bb10:
  %v62 = phi float [ %v36, %bb9 ], [ %v82, %bb13 ]
  %v63 = phi i64 [ 0, %bb9 ], [ %v83, %bb13 ]
  %v64 = icmp ult i64 %v63, 32
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb14, label %bb11
bb11:
  %v66 = add i64 %v42, 2
  %v67 = add i64 %v66, %v63
  %v68 = icmp ult i64 %v67, %v43
  br i1 %v68, label %bb12, label %bb30
bb12:
  %v69 = extractvalue { ptr, i64 } %v15, 0
  %v70 = getelementptr inbounds i8, ptr %v69, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v72 = bitcast i8 %v71 to i8
  %v73 = sitofp i8 %v72 to float
  %v74 = fmul contract float %v58, %v73
  %v75 = add i64 %v61, %v63
  %v76 = extractvalue { ptr, i64 } %v16, 1
  %v77 = icmp ult i64 %v75, %v76
  br i1 %v77, label %bb13, label %bb31
bb13:
  %v78 = extractvalue { ptr, i64 } %v16, 0
  %v79 = getelementptr inbounds float, ptr %v78, i64 %v75
  %v80 = load float, ptr %v79, align 4
  %v81 = fmul contract float %v74, %v80
  %v82 = fadd contract float %v62, %v81
  %v83 = add i64 %v63, 1
  br label %bb10
bb14:
  %v84 = add i64 %v37, 1
  br label %bb5
bb15:
  %v85 = icmp eq i64 %v24, 18446744073709551615
  br i1 %v85, label %bb23, label %bb20
bb16:
  %v86 = extractvalue { ptr } %v98, 0
  store float %v36, ptr %v86, align 4
  br label %bb18
bb17:
  br label %bb18
bb18:
  br label %bb19
bb19:
  ret void
bb20:
  %v87 = extractvalue { ptr, i64 } %v20, 1
  %v88 = icmp ult i64 %v24, %v87
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb22, label %bb21
bb21:
  %v90 = extractvalue { ptr, i64 } %v20, 0
  %v91 = getelementptr inbounds float, ptr %v90, i64 %v24
  %v92 = insertvalue { ptr } undef, ptr %v91, 0
  %v93 = extractvalue { ptr } %v92, 0
  br label %bb24
bb22:
  br label %bb23
bb23:
  %v94 = inttoptr i64 0 to ptr
  %v95 = insertvalue { ptr } undef, ptr %v94, 0
  %v96 = extractvalue { ptr } %v95, 0
  br label %bb24
bb24:
  %v97 = phi ptr [ %v93, %bb21 ], [ %v96, %bb23 ]
  %v98 = insertvalue { ptr } undef, ptr %v97, 0
  %v99 = extractvalue { ptr } %v98, 0
  %v100 = ptrtoint ptr %v99 to i64
  %v101 = sub i64 %v100, 0
  %v102 = icmp ule i64 %v101, 0
  %v103 = add i64 %v101, 0
  %v104 = select i1 %v102, i64 %v103, i64 1
  %v105 = icmp eq i64 %v104, 1
  br i1 %v105, label %bb16, label %bb25
bb25:
  %v106 = icmp eq i64 %v104, 0
  br i1 %v106, label %bb17, label %bb26
bb26:
  unreachable
bb27:
  call void @llvm.trap() #0
  unreachable
bb28:
  call void @llvm.trap() #0
  unreachable
bb29:
  call void @llvm.trap() #0
  unreachable
bb30:
  call void @llvm.trap() #0
  unreachable
bb31:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @argmax_token(ptr %v0, i64 %v1, i32 %v2, ptr %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi i32 [ %v2, %entry ]
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v13 = zext i32 %v12 to i64
  br label %bb2
bb2:
  %v14 = phi float [ 0xFFF0000000000000, %bb1 ], [ %v33, %bb11 ]
  %v15 = phi i32 [ 0, %bb1 ], [ %v34, %bb11 ]
  %v16 = phi i64 [ %v13, %bb1 ], [ %v35, %bb11 ]
  %v17 = zext i32 %v10 to i64
  %v18 = icmp ult i64 %v16, %v17
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb12, label %bb3
bb3:
  %v20 = extractvalue { ptr, i64 } %v9, 1
  %v21 = icmp ult i64 %v16, %v20
  %v22 = extractvalue { ptr, i64 } %v9, 0
  %v23 = getelementptr inbounds float, ptr %v22, i64 %v16
  %v24 = load float, ptr %v23, align 4
  %v25 = fcmp ogt float %v24, %v14
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb5, label %bb4
bb4:
  br label %bb8
bb5:
  %v27 = fcmp oeq float %v24, %v14
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb10, label %bb6
bb6:
  %v29 = trunc i64 %v16 to i32
  %v30 = icmp ult i32 %v29, %v15
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb9, label %bb7
bb7:
  br label %bb8
bb8:
  %v32 = trunc i64 %v16 to i32
  br label %bb11
bb9:
  br label %bb11
bb10:
  br label %bb11
bb11:
  %v33 = phi float [ %v24, %bb8 ], [ %v14, %bb9 ], [ %v14, %bb10 ]
  %v34 = phi i32 [ %v32, %bb8 ], [ %v15, %bb9 ], [ %v15, %bb10 ]
  %v35 = add i64 %v16, 256
  br label %bb2
bb12:
  %v36 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_8, i64 %v13
  br label %bb13
bb13:
  store float %v14, ptr addrspace(3) %v36, align 4
  %v37 = getelementptr inbounds i32, ptr addrspace(3) @__shared_mem_9, i64 %v13
  br label %bb14
bb14:
  store i32 %v15, ptr addrspace(3) %v37, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v39 = phi i64 [ 128, %bb15 ], [ %v66, %bb31 ]
  %v40 = icmp eq i64 %v39, 0
  br i1 %v40, label %bb32, label %bb17
bb17:
  %v41 = icmp ult i64 %v13, %v39
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb29, label %bb18
bb18:
  %v43 = bitcast ptr addrspace(3) @__shared_mem_8 to ptr addrspace(3)
  %v44 = add i64 %v13, %v39
  %v45 = getelementptr inbounds float, ptr addrspace(3) %v43, i64 %v44
  br label %bb19
bb19:
  %v46 = load float, ptr addrspace(3) %v45, align 4
  %v47 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v48 = add i64 %v13, %v39
  %v49 = getelementptr inbounds i32, ptr addrspace(3) %v47, i64 %v48
  br label %bb20
bb20:
  %v50 = load i32, ptr addrspace(3) %v49, align 4
  %v51 = bitcast ptr addrspace(3) @__shared_mem_8 to ptr addrspace(3)
  %v52 = getelementptr inbounds float, ptr addrspace(3) %v51, i64 %v13
  br label %bb21
bb21:
  %v53 = load float, ptr addrspace(3) %v52, align 4
  %v54 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v55 = getelementptr inbounds i32, ptr addrspace(3) %v54, i64 %v13
  br label %bb22
bb22:
  %v56 = load i32, ptr addrspace(3) %v55, align 4
  %v57 = fcmp ogt float %v46, %v53
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb23, label %bb25
bb23:
  %v59 = fcmp oeq float %v46, %v53
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb28, label %bb24
bb24:
  %v61 = icmp ult i32 %v50, %v56
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb28, label %bb25
bb25:
  %v63 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_8, i64 %v13
  br label %bb26
bb26:
  store float %v46, ptr addrspace(3) %v63, align 4
  %v64 = getelementptr inbounds i32, ptr addrspace(3) @__shared_mem_9, i64 %v13
  br label %bb27
bb27:
  store i32 %v50, ptr addrspace(3) %v64, align 4
  br label %bb28
bb28:
  br label %bb30
bb29:
  br label %bb30
bb30:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb31
bb31:
  %v66 = udiv i64 %v39, 2
  br label %bb16
bb32:
  %v67 = icmp eq i64 %v13, 0
  br i1 %v67, label %bb33, label %bb35
bb33:
  %v68 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v69 = getelementptr inbounds i32, ptr addrspace(3) %v68, i64 0
  br label %bb34
bb34:
  %v70 = load i32, ptr addrspace(3) %v69, align 4
  %v71 = extractvalue { ptr, i64 } %v11, 0
  %v72 = uitofp i32 %v70 to float
  store float %v72, ptr %v71, align 4
  br label %bb35
bb35:
  ret void
}

define ptx_kernel void @q4k_q8_gemv_warp4(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, ptr %v9, i64 %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v5, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v10, 1
  br label %bb0
bb0:
  %v19 = phi { ptr, i64 } [ %v12, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = phi i32 [ %v6, %entry ]
  %v23 = phi i32 [ %v7, %entry ]
  %v24 = phi i32 [ %v8, %entry ]
  %v25 = phi { ptr, i64 } [ %v18, %entry ]
  %v26 = alloca { [0 x i32], { { i32 } } }, align 4
  %v27 = alloca { [0 x i32], { { i32 } } }, align 4
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v29 = zext i32 %v28 to i64
  %v30 = zext i32 %v22 to i64
  %v31 = add i64 %v30, 3
  %v32 = udiv i64 %v31, 4
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v34 = zext i32 %v33 to i64
  %v35 = zext i32 %v24 to i64
  %v36 = mul i64 %v32, %v35
  %v37 = icmp uge i64 %v34, %v36
  %v38 = xor i1 %v37, 1
  br i1 %v38, label %bb4, label %bb3
bb3:
  br label %bb40
bb4:
  %v39 = icmp eq i64 %v32, 0
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb5, label %bb45
bb5:
  %v41 = udiv i64 %v34, %v32
  %v42 = urem i64 %v34, %v32
  %v43 = mul i64 %v42, 4
  %v44 = zext i32 %v23 to i64
  %v45 = mul i64 %v44, 144
  %v46 = mul i64 %v41, %v44
  %v47 = mul i64 %v46, 256
  br label %bb6
bb6:
  %v48 = phi float [ 0.0, %bb5 ], [ %v55, %bb22 ]
  %v49 = phi float [ 0.0, %bb5 ], [ %v56, %bb22 ]
  %v50 = phi float [ 0.0, %bb5 ], [ %v57, %bb22 ]
  %v51 = phi float [ 0.0, %bb5 ], [ %v58, %bb22 ]
  %v52 = phi i64 [ 0, %bb5 ], [ %v148, %bb22 ]
  %v53 = icmp ult i64 %v52, %v44
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb23, label %bb7
bb7:
  br label %bb8
bb8:
  %v55 = phi float [ %v48, %bb7 ], [ %v113, %bb21 ]
  %v56 = phi float [ %v49, %bb7 ], [ %v124, %bb21 ]
  %v57 = phi float [ %v50, %bb7 ], [ %v135, %bb21 ]
  %v58 = phi float [ %v51, %bb7 ], [ %v146, %bb21 ]
  %v59 = phi i64 [ 0, %bb7 ], [ %v147, %bb21 ]
  %v60 = icmp ult i64 %v59, 2
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb22, label %bb9
bb9:
  %v62 = mul i64 %v59, 128
  %v63 = mul i64 %v29, 4
  %v64 = add i64 %v62, %v63
  %v65 = mul i64 %v52, 256
  %v66 = add i64 %v47, %v65
  %v67 = add i64 %v66, %v64
  %v68 = extractvalue { ptr, i64 } %v20, 0
  %v69 = getelementptr inbounds i8, ptr %v68, i64 %v67
  store { [0 x i32], { { i32 } } } undef, ptr %v26, align 4
  %v70 = bitcast ptr %v69 to ptr
  %v71 = bitcast ptr %v26 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %v71, ptr %v70, i64 4, i1 0) #0
  %v73 = load { [0 x i32], { { i32 } } }, ptr %v26, align 4
  store { [0 x i32], { { i32 } } } %v73, ptr %v27, align 4
  %v74 = getelementptr inbounds i8, ptr %v27, i32 0
  %v75 = bitcast ptr %v74 to ptr
  %v76 = load i32, ptr %v75, align 4
  %v77 = trunc i32 %v76 to i8
  %v78 = bitcast i8 %v77 to i8
  %v79 = and i32 8, 31
  %v80 = lshr i32 %v76, %v79
  %v81 = trunc i32 %v80 to i8
  %v82 = bitcast i8 %v81 to i8
  %v83 = and i32 16, 31
  %v84 = lshr i32 %v76, %v83
  %v85 = trunc i32 %v84 to i8
  %v86 = bitcast i8 %v85 to i8
  %v87 = and i32 24, 31
  %v88 = lshr i32 %v76, %v87
  %v89 = trunc i32 %v88 to i8
  %v90 = bitcast i8 %v89 to i8
  %v91 = sext i8 %v78 to i32
  %v92 = sext i8 %v82 to i32
  %v93 = add i32 %v91, %v92
  %v94 = sext i8 %v86 to i32
  %v95 = add i32 %v93, %v94
  %v96 = sext i8 %v90 to i32
  %v97 = add i32 %v95, %v96
  %v98 = udiv i64 %v67, 32
  %v99 = extractvalue { ptr, i64 } %v21, 1
  %v100 = icmp ult i64 %v98, %v99
  %v101 = extractvalue { ptr, i64 } %v21, 0
  %v102 = getelementptr inbounds float, ptr %v101, i64 %v98
  %v103 = load float, ptr %v102, align 4
  %v104 = icmp ult i64 %v43, %v30
  %v105 = xor i1 %v104, 1
  br i1 %v105, label %bb12, label %bb10
bb10:
  %v106 = mul i64 %v43, %v45
  %v107 = mul i64 %v52, 144
  %v108 = add i64 %v106, %v107
  %v109 = extractvalue { ptr, i64 } %v19, 0
  %v110 = extractvalue { ptr, i64 } %v19, 1
  %v111 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v109, i64 %v110, i64 %v108, i64 %v64, i32 %v76, i32 %v97, float %v103) #0
  br label %bb11
bb11:
  %v112 = fadd contract float %v55, %v111
  br label %bb12
bb12:
  %v113 = phi float [ %v55, %bb9 ], [ %v112, %bb11 ]
  %v114 = add i64 %v43, 1
  %v115 = icmp ult i64 %v114, %v30
  %v116 = xor i1 %v115, 1
  br i1 %v116, label %bb15, label %bb13
bb13:
  %v117 = mul i64 %v114, %v45
  %v118 = mul i64 %v52, 144
  %v119 = add i64 %v117, %v118
  %v120 = extractvalue { ptr, i64 } %v19, 0
  %v121 = extractvalue { ptr, i64 } %v19, 1
  %v122 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v120, i64 %v121, i64 %v119, i64 %v64, i32 %v76, i32 %v97, float %v103) #0
  br label %bb14
bb14:
  %v123 = fadd contract float %v56, %v122
  br label %bb15
bb15:
  %v124 = phi float [ %v56, %bb12 ], [ %v123, %bb14 ]
  %v125 = add i64 %v43, 2
  %v126 = icmp ult i64 %v125, %v30
  %v127 = xor i1 %v126, 1
  br i1 %v127, label %bb18, label %bb16
bb16:
  %v128 = mul i64 %v125, %v45
  %v129 = mul i64 %v52, 144
  %v130 = add i64 %v128, %v129
  %v131 = extractvalue { ptr, i64 } %v19, 0
  %v132 = extractvalue { ptr, i64 } %v19, 1
  %v133 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v131, i64 %v132, i64 %v130, i64 %v64, i32 %v76, i32 %v97, float %v103) #0
  br label %bb17
bb17:
  %v134 = fadd contract float %v57, %v133
  br label %bb18
bb18:
  %v135 = phi float [ %v57, %bb15 ], [ %v134, %bb17 ]
  %v136 = add i64 %v43, 3
  %v137 = icmp ult i64 %v136, %v30
  %v138 = xor i1 %v137, 1
  br i1 %v138, label %bb21, label %bb19
bb19:
  %v139 = mul i64 %v136, %v45
  %v140 = mul i64 %v52, 144
  %v141 = add i64 %v139, %v140
  %v142 = extractvalue { ptr, i64 } %v19, 0
  %v143 = extractvalue { ptr, i64 } %v19, 1
  %v144 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v142, i64 %v143, i64 %v141, i64 %v64, i32 %v76, i32 %v97, float %v103) #0
  br label %bb20
bb20:
  %v145 = fadd contract float %v58, %v144
  br label %bb21
bb21:
  %v146 = phi float [ %v58, %bb18 ], [ %v145, %bb20 ]
  %v147 = add i64 %v59, 1
  br label %bb8
bb22:
  %v148 = add i64 %v52, 1
  br label %bb6
bb23:
  br label %bb24
bb24:
  %v149 = phi float [ %v48, %bb23 ], [ %v182, %bb44 ]
  %v150 = phi float [ %v49, %bb23 ], [ %v184, %bb44 ]
  %v151 = phi float [ %v50, %bb23 ], [ %v186, %bb44 ]
  %v152 = phi float [ %v51, %bb23 ], [ %v188, %bb44 ]
  %v153 = phi i32 [ 16, %bb23 ], [ %v189, %bb44 ]
  %v154 = icmp ugt i32 %v153, 0
  %v155 = xor i1 %v154, 1
  br i1 %v155, label %bb26, label %bb25
bb25:
  %v156 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v149, i32 %v153, i32 31) #0
  br label %bb41
bb26:
  %v157 = icmp eq i64 %v29, 0
  br i1 %v157, label %bb27, label %bb39
bb27:
  %v158 = mul i64 %v41, %v30
  %v159 = add i64 %v158, %v43
  %v160 = icmp ult i64 %v43, %v30
  %v161 = xor i1 %v160, 1
  br i1 %v161, label %bb29, label %bb28
bb28:
  %v162 = extractvalue { ptr, i64 } %v25, 0
  %v163 = getelementptr inbounds float, ptr %v162, i64 %v159
  store float %v149, ptr %v163, align 4
  br label %bb29
bb29:
  %v164 = add i64 %v43, 1
  %v165 = icmp ult i64 %v164, %v30
  %v166 = xor i1 %v165, 1
  br i1 %v166, label %bb31, label %bb30
bb30:
  %v167 = add i64 %v159, 1
  %v168 = extractvalue { ptr, i64 } %v25, 0
  %v169 = getelementptr inbounds float, ptr %v168, i64 %v167
  store float %v150, ptr %v169, align 4
  br label %bb32
bb31:
  br label %bb32
bb32:
  %v170 = add i64 %v43, 2
  %v171 = icmp ult i64 %v170, %v30
  %v172 = xor i1 %v171, 1
  br i1 %v172, label %bb34, label %bb33
bb33:
  %v173 = add i64 %v159, 2
  %v174 = extractvalue { ptr, i64 } %v25, 0
  %v175 = getelementptr inbounds float, ptr %v174, i64 %v173
  store float %v151, ptr %v175, align 4
  br label %bb35
bb34:
  br label %bb35
bb35:
  %v176 = add i64 %v43, 3
  %v177 = icmp ult i64 %v176, %v30
  %v178 = xor i1 %v177, 1
  br i1 %v178, label %bb37, label %bb36
bb36:
  %v179 = add i64 %v159, 3
  %v180 = extractvalue { ptr, i64 } %v25, 0
  %v181 = getelementptr inbounds float, ptr %v180, i64 %v179
  store float %v152, ptr %v181, align 4
  br label %bb38
bb37:
  br label %bb38
bb38:
  br label %bb39
bb39:
  br label %bb40
bb40:
  ret void
bb41:
  %v182 = fadd contract float %v149, %v156
  %v183 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v150, i32 %v153, i32 31) #0
  br label %bb42
bb42:
  %v184 = fadd contract float %v150, %v183
  %v185 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v151, i32 %v153, i32 31) #0
  br label %bb43
bb43:
  %v186 = fadd contract float %v151, %v185
  %v187 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v152, i32 %v153, i32 31) #0
  br label %bb44
bb44:
  %v188 = fadd contract float %v152, %v187
  %v189 = udiv i32 %v153, 2
  br label %bb24
bb45:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @attention_canvas_paged_heads(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, float %v15, i32 %v16, i32 %v17, i32 %v18, i32 %v19, i32 %v20, ptr %v21, i64 %v22) #0 {
entry:
  %v23 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v1, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v3, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v5, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v7, 1
  %v31 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v32 = insertvalue { ptr, i64 } %v31, i64 %v9, 1
  %v33 = insertvalue { ptr, i64 } undef, ptr %v21, 0
  %v34 = insertvalue { ptr, i64 } %v33, i64 %v22, 1
  br label %bb0
bb0:
  %v35 = phi { ptr, i64 } [ %v24, %entry ]
  %v36 = phi { ptr, i64 } [ %v26, %entry ]
  %v37 = phi { ptr, i64 } [ %v28, %entry ]
  %v38 = phi { ptr, i64 } [ %v30, %entry ]
  %v39 = phi { ptr, i64 } [ %v32, %entry ]
  %v40 = phi i32 [ %v10, %entry ]
  %v41 = phi i32 [ %v11, %entry ]
  %v42 = phi i32 [ %v12, %entry ]
  %v43 = phi i32 [ %v13, %entry ]
  %v44 = phi i32 [ %v14, %entry ]
  %v45 = phi float [ %v15, %entry ]
  %v46 = phi i32 [ %v16, %entry ]
  %v47 = phi i32 [ %v17, %entry ]
  %v48 = phi i32 [ %v18, %entry ]
  %v49 = phi i32 [ %v19, %entry ]
  %v50 = phi i32 [ %v20, %entry ]
  %v51 = phi { ptr, i64 } [ %v34, %entry ]
  %v52 = alloca {  }, align 1
  %v53 = bitcast ptr %v52 to ptr
  %v54 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v53) #0
  br label %bb1
bb1:
  %v55 = zext i32 %v40 to i64
  %v56 = zext i32 %v42 to i64
  %v57 = mul i64 %v55, %v56
  %v58 = icmp uge i64 %v54, %v57
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb3, label %bb2
bb2:
  br label %bb60
bb3:
  %v60 = icmp eq i64 %v56, 0
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb4, label %bb65
bb4:
  %v62 = udiv i64 %v54, %v56
  %v63 = urem i64 %v54, %v56
  %v64 = zext i32 %v44 to i64
  %v65 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v43, i32 1) #0
  br label %bb5
bb5:
  %v66 = zext i32 %v65 to i64
  %v67 = icmp eq i64 %v66, 0
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb6, label %bb66
bb6:
  %v69 = udiv i64 %v56, %v66
  %v70 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i64 %v69, i64 1) #0
  br label %bb7
bb7:
  %v71 = icmp eq i64 %v70, 0
  %v72 = xor i1 %v71, 1
  br i1 %v72, label %bb8, label %bb67
bb8:
  %v73 = udiv i64 %v63, %v70
  %v74 = mul i64 %v62, %v56
  %v75 = mul i64 %v74, %v64
  %v76 = mul i64 %v63, %v64
  %v77 = add i64 %v75, %v76
  br label %bb9
bb9:
  %v78 = phi i64 [ 0, %bb8 ], [ %v84, %bb10 ]
  %v79 = icmp ult i64 %v78, %v64
  %v80 = xor i1 %v79, 1
  br i1 %v80, label %bb11, label %bb10
bb10:
  %v81 = add i64 %v77, %v78
  %v82 = extractvalue { ptr, i64 } %v51, 0
  %v83 = getelementptr inbounds float, ptr %v82, i64 %v81
  store float 0.0, ptr %v83, align 4
  %v84 = add i64 %v78, 1
  br label %bb9
bb11:
  br label %bb12
bb12:
  %v85 = phi float [ 0.0, %bb11 ], [ %v250, %bb52 ]
  %v86 = phi float [ 0.0, %bb11 ], [ %v251, %bb52 ]
  %v87 = phi i1 [ 0, %bb11 ], [ %v252, %bb52 ]
  %v88 = phi i64 [ 0, %bb11 ], [ %v253, %bb52 ]
  %v89 = zext i32 %v41 to i64
  %v90 = add i64 %v89, %v55
  %v91 = icmp ult i64 %v88, %v90
  %v92 = xor i1 %v91, 1
  br i1 %v92, label %bb53, label %bb13
bb13:
  %v93 = icmp ult i64 %v88, %v89
  %v94 = xor i1 %v93, 1
  br i1 %v94, label %bb36, label %bb14
bb14:
  %v95 = zext i32 %v48 to i64
  %v96 = icmp eq i64 %v95, 0
  %v97 = xor i1 %v96, 1
  br i1 %v97, label %bb15, label %bb68
bb15:
  %v98 = udiv i64 %v88, %v95
  %v99 = urem i64 %v88, %v95
  %v100 = zext i32 %v46 to i64
  %v101 = zext i32 %v47 to i64
  %v102 = mul i64 %v100, %v101
  %v103 = add i64 %v102, %v98
  %v104 = extractvalue { ptr, i64 } %v37, 1
  %v105 = icmp ult i64 %v103, %v104
  br i1 %v105, label %bb16, label %bb69
bb16:
  %v106 = extractvalue { ptr, i64 } %v37, 0
  %v107 = getelementptr inbounds i32, ptr %v106, i64 %v103
  %v108 = load i32, ptr %v107, align 4
  %v109 = zext i32 %v108 to i64
  %v110 = zext i32 %v50 to i64
  %v111 = mul i64 %v110, 2
  %v112 = zext i32 %v49 to i64
  %v113 = mul i64 %v109, %v112
  %v114 = mul i64 %v99, %v111
  %v115 = add i64 %v113, %v114
  %v116 = mul i64 %v73, %v64
  %v117 = mul i64 %v116, 2
  %v118 = add i64 %v115, %v117
  br label %bb17
bb17:
  %v119 = phi i64 [ 0, %bb16 ], [ %v152, %bb22 ]
  %v120 = phi float [ 0.0, %bb16 ], [ %v151, %bb22 ]
  %v121 = icmp ult i64 %v119, %v64
  %v122 = xor i1 %v121, 1
  br i1 %v122, label %bb23, label %bb18
bb18:
  %v123 = mul i64 %v119, 2
  %v124 = add i64 %v118, %v123
  %v125 = extractvalue { ptr, i64 } %v36, 1
  %v126 = icmp ult i64 %v124, %v125
  br i1 %v126, label %bb19, label %bb70
bb19:
  %v127 = extractvalue { ptr, i64 } %v36, 0
  %v128 = getelementptr inbounds i8, ptr %v127, i64 %v124
  %v129 = load i8, ptr %v128, align 1
  %v130 = zext i8 %v129 to i16
  %v131 = mul i64 %v119, 2
  %v132 = add i64 %v118, %v131
  %v133 = add i64 %v132, 1
  %v134 = icmp ult i64 %v133, %v125
  br i1 %v134, label %bb20, label %bb71
bb20:
  %v135 = extractvalue { ptr, i64 } %v36, 0
  %v136 = getelementptr inbounds i8, ptr %v135, i64 %v133
  %v137 = load i8, ptr %v136, align 1
  %v138 = zext i8 %v137 to i16
  %v139 = trunc i32 8 to i16
  %v140 = and i16 %v139, 15
  %v141 = shl i16 %v138, %v140
  %v142 = or i16 %v130, %v141
  %v143 = add i64 %v77, %v119
  %v144 = extractvalue { ptr, i64 } %v35, 1
  %v145 = icmp ult i64 %v143, %v144
  br i1 %v145, label %bb21, label %bb72
bb21:
  %v146 = extractvalue { ptr, i64 } %v35, 0
  %v147 = getelementptr inbounds float, ptr %v146, i64 %v143
  %v148 = load float, ptr %v147, align 4
  %v149 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v142) #0
  br label %bb22
bb22:
  %v150 = fmul contract float %v148, %v149
  %v151 = fadd contract float %v120, %v150
  %v152 = add i64 %v119, 1
  br label %bb17
bb23:
  %v153 = fmul contract float %v120, %v45
  %v154 = xor i1 %v87, 1
  br i1 %v154, label %bb25, label %bb24
bb24:
  %v155 = fcmp ogt float %v153, %v85
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb27, label %bb26
bb25:
  br label %bb29
bb26:
  %v157 = fsub contract float %v85, %v153
  %v158 = call float @__nv_expf(float %v157) #0
  br label %bb61
bb27:
  br label %bb28
bb28:
  %v159 = phi float [ %v85, %bb27 ], [ %v153, %bb61 ]
  %v160 = phi float [ 1.0, %bb27 ], [ %v158, %bb61 ]
  br label %bb29
bb29:
  %v161 = phi float [ %v153, %bb25 ], [ %v159, %bb28 ]
  %v162 = phi float [ 0.0, %bb25 ], [ %v160, %bb28 ]
  %v163 = fsub contract float %v153, %v161
  %v164 = call float @__nv_expf(float %v163) #0
  br label %bb62
bb30:
  %v165 = phi i64 [ %v196, %bb34 ], [ 0, %bb62 ]
  %v166 = icmp ult i64 %v165, %v64
  %v167 = xor i1 %v166, 1
  br i1 %v167, label %bb35, label %bb31
bb31:
  %v168 = mul i64 %v165, 2
  %v169 = add i64 %v270, %v168
  %v170 = extractvalue { ptr, i64 } %v36, 1
  %v171 = icmp ult i64 %v169, %v170
  br i1 %v171, label %bb32, label %bb73
bb32:
  %v172 = extractvalue { ptr, i64 } %v36, 0
  %v173 = getelementptr inbounds i8, ptr %v172, i64 %v169
  %v174 = load i8, ptr %v173, align 1
  %v175 = zext i8 %v174 to i16
  %v176 = mul i64 %v165, 2
  %v177 = add i64 %v270, %v176
  %v178 = add i64 %v177, 1
  %v179 = icmp ult i64 %v178, %v170
  br i1 %v179, label %bb33, label %bb74
bb33:
  %v180 = extractvalue { ptr, i64 } %v36, 0
  %v181 = getelementptr inbounds i8, ptr %v180, i64 %v178
  %v182 = load i8, ptr %v181, align 1
  %v183 = zext i8 %v182 to i16
  %v184 = trunc i32 8 to i16
  %v185 = and i16 %v184, 15
  %v186 = shl i16 %v183, %v185
  %v187 = or i16 %v175, %v186
  %v188 = add i64 %v77, %v165
  %v189 = extractvalue { ptr, i64 } %v51, 0
  %v190 = getelementptr inbounds float, ptr %v189, i64 %v188
  %v191 = load float, ptr %v190, align 4
  %v192 = fmul contract float %v191, %v162
  %v193 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v187) #0
  br label %bb34
bb34:
  %v194 = fmul contract float %v164, %v193
  %v195 = fadd contract float %v192, %v194
  store float %v195, ptr %v190, align 4
  %v196 = add i64 %v165, 1
  br label %bb30
bb35:
  br label %bb52
bb36:
  %v197 = sub i64 %v88, %v89
  %v198 = mul i64 %v197, %v66
  %v199 = mul i64 %v198, %v64
  %v200 = mul i64 %v73, %v64
  %v201 = add i64 %v199, %v200
  br label %bb37
bb37:
  %v202 = phi i64 [ 0, %bb36 ], [ %v220, %bb40 ]
  %v203 = phi float [ 0.0, %bb36 ], [ %v219, %bb40 ]
  %v204 = icmp ult i64 %v202, %v64
  %v205 = xor i1 %v204, 1
  br i1 %v205, label %bb41, label %bb38
bb38:
  %v206 = add i64 %v77, %v202
  %v207 = extractvalue { ptr, i64 } %v35, 1
  %v208 = icmp ult i64 %v206, %v207
  br i1 %v208, label %bb39, label %bb75
bb39:
  %v209 = extractvalue { ptr, i64 } %v35, 0
  %v210 = getelementptr inbounds float, ptr %v209, i64 %v206
  %v211 = load float, ptr %v210, align 4
  %v212 = add i64 %v201, %v202
  %v213 = extractvalue { ptr, i64 } %v38, 1
  %v214 = icmp ult i64 %v212, %v213
  br i1 %v214, label %bb40, label %bb76
bb40:
  %v215 = extractvalue { ptr, i64 } %v38, 0
  %v216 = getelementptr inbounds float, ptr %v215, i64 %v212
  %v217 = load float, ptr %v216, align 4
  %v218 = fmul contract float %v211, %v217
  %v219 = fadd contract float %v203, %v218
  %v220 = add i64 %v202, 1
  br label %bb37
bb41:
  %v221 = fmul contract float %v203, %v45
  %v222 = xor i1 %v87, 1
  br i1 %v222, label %bb43, label %bb42
bb42:
  %v223 = fcmp ogt float %v221, %v85
  %v224 = xor i1 %v223, 1
  br i1 %v224, label %bb45, label %bb44
bb43:
  br label %bb47
bb44:
  %v225 = fsub contract float %v85, %v221
  %v226 = call float @__nv_expf(float %v225) #0
  br label %bb63
bb45:
  br label %bb46
bb46:
  %v227 = phi float [ %v85, %bb45 ], [ %v221, %bb63 ]
  %v228 = phi float [ 1.0, %bb45 ], [ %v226, %bb63 ]
  br label %bb47
bb47:
  %v229 = phi float [ %v221, %bb43 ], [ %v227, %bb46 ]
  %v230 = phi float [ 0.0, %bb43 ], [ %v228, %bb46 ]
  %v231 = fsub contract float %v221, %v229
  %v232 = call float @__nv_expf(float %v231) #0
  br label %bb64
bb48:
  %v233 = phi i64 [ %v249, %bb50 ], [ 0, %bb64 ]
  %v234 = icmp ult i64 %v233, %v64
  %v235 = xor i1 %v234, 1
  br i1 %v235, label %bb51, label %bb49
bb49:
  %v236 = add i64 %v77, %v233
  %v237 = extractvalue { ptr, i64 } %v51, 0
  %v238 = getelementptr inbounds float, ptr %v237, i64 %v236
  %v239 = load float, ptr %v238, align 4
  %v240 = fmul contract float %v239, %v230
  %v241 = add i64 %v201, %v233
  %v242 = extractvalue { ptr, i64 } %v39, 1
  %v243 = icmp ult i64 %v241, %v242
  br i1 %v243, label %bb50, label %bb77
bb50:
  %v244 = extractvalue { ptr, i64 } %v39, 0
  %v245 = getelementptr inbounds float, ptr %v244, i64 %v241
  %v246 = load float, ptr %v245, align 4
  %v247 = fmul contract float %v232, %v246
  %v248 = fadd contract float %v240, %v247
  store float %v248, ptr %v238, align 4
  %v249 = add i64 %v233, 1
  br label %bb48
bb51:
  br label %bb52
bb52:
  %v250 = phi float [ %v161, %bb35 ], [ %v229, %bb51 ]
  %v251 = phi float [ %v266, %bb35 ], [ %v272, %bb51 ]
  %v252 = phi i1 [ 1, %bb35 ], [ 1, %bb51 ]
  %v253 = add i64 %v88, 1
  br label %bb12
bb53:
  %v254 = fcmp ogt float %v86, 0.0
  %v255 = xor i1 %v254, 1
  br i1 %v255, label %bb55, label %bb54
bb54:
  br label %bb56
bb55:
  br label %bb59
bb56:
  %v256 = phi i64 [ 0, %bb54 ], [ %v264, %bb57 ]
  %v257 = icmp ult i64 %v256, %v64
  %v258 = xor i1 %v257, 1
  br i1 %v258, label %bb58, label %bb57
bb57:
  %v259 = add i64 %v77, %v256
  %v260 = extractvalue { ptr, i64 } %v51, 0
  %v261 = getelementptr inbounds float, ptr %v260, i64 %v259
  %v262 = load float, ptr %v261, align 4
  %v263 = fdiv contract float %v262, %v86
  store float %v263, ptr %v261, align 4
  %v264 = add i64 %v256, 1
  br label %bb56
bb58:
  br label %bb59
bb59:
  br label %bb60
bb60:
  ret void
bb61:
  br label %bb28
bb62:
  %v265 = fmul contract float %v86, %v162
  %v266 = fadd contract float %v265, %v164
  %v267 = mul i64 %v95, %v111
  %v268 = add i64 %v113, %v267
  %v269 = add i64 %v268, %v114
  %v270 = add i64 %v269, %v117
  br label %bb30
bb63:
  br label %bb46
bb64:
  %v271 = fmul contract float %v86, %v230
  %v272 = fadd contract float %v271, %v232
  br label %bb48
bb65:
  call void @llvm.trap() #0
  unreachable
bb66:
  call void @llvm.trap() #0
  unreachable
bb67:
  call void @llvm.trap() #0
  unreachable
bb68:
  call void @llvm.trap() #0
  unreachable
bb69:
  call void @llvm.trap() #0
  unreachable
bb70:
  call void @llvm.trap() #0
  unreachable
bb71:
  call void @llvm.trap() #0
  unreachable
bb72:
  call void @llvm.trap() #0
  unreachable
bb73:
  call void @llvm.trap() #0
  unreachable
bb74:
  call void @llvm.trap() #0
  unreachable
bb75:
  call void @llvm.trap() #0
  unreachable
bb76:
  call void @llvm.trap() #0
  unreachable
bb77:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q6k_q8_gemv_multiwarp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr %v12, i64 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v13, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi { ptr, i64 } [ %v23, %entry ]
  %v33 = alloca { [0 x i32], { { i32 } } }, align 4
  %v34 = alloca { [0 x i32], { { i32 } } }, align 4
  %v35 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v36 = zext i32 %v35 to i64
  %v37 = and i64 %v36, 31
  %v38 = zext i32 5 to i64
  %v39 = and i64 %v38, 63
  %v40 = lshr i64 %v36, %v39
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v42 = zext i32 %v41 to i64
  %v43 = zext i32 5 to i64
  %v44 = and i64 %v43, 63
  %v45 = lshr i64 %v42, %v44
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb3
bb3:
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v29 to i64
  %v49 = zext i32 %v31 to i64
  %v50 = mul i64 %v48, %v49
  %v51 = icmp uge i64 %v47, %v50
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb5, label %bb4
bb4:
  br label %bb6
bb5:
  %v53 = icmp uge i64 %v40, %v45
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb7, label %bb6
bb6:
  br label %bb37
bb7:
  %v55 = icmp eq i64 %v48, 0
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb8, label %bb40
bb8:
  %v57 = urem i64 %v47, %v48
  %v58 = udiv i64 %v47, %v48
  %v59 = zext i32 %v30 to i64
  %v60 = mul i64 %v59, 210
  %v61 = mul i64 %v58, %v59
  %v62 = mul i64 %v61, 256
  br label %bb9
bb9:
  %v63 = phi float [ 0.0, %bb8 ], [ %v68, %bb14 ]
  %v64 = phi i64 [ %v40, %bb8 ], [ %v100, %bb14 ]
  %v65 = icmp ult i64 %v64, %v59
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb15, label %bb10
bb10:
  %v67 = mul i64 %v37, 8
  br label %bb11
bb11:
  %v68 = phi float [ %v63, %bb10 ], [ %v98, %bb13 ]
  %v69 = phi i64 [ 0, %bb10 ], [ %v99, %bb13 ]
  %v70 = icmp ult i64 %v69, 2
  %v71 = xor i1 %v70, 1
  br i1 %v71, label %bb14, label %bb12
bb12:
  %v72 = mul i64 %v69, 4
  %v73 = add i64 %v67, %v72
  %v74 = mul i64 %v64, 256
  %v75 = add i64 %v62, %v74
  %v76 = add i64 %v75, %v73
  %v77 = extractvalue { ptr, i64 } %v25, 0
  %v78 = getelementptr inbounds i8, ptr %v77, i64 %v76
  store { [0 x i32], { { i32 } } } undef, ptr %v33, align 4
  %v79 = bitcast ptr %v78 to ptr
  %v80 = bitcast ptr %v33 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %v80, ptr %v79, i64 4, i1 0) #0
  %v82 = load { [0 x i32], { { i32 } } }, ptr %v33, align 4
  store { [0 x i32], { { i32 } } } %v82, ptr %v34, align 4
  %v83 = getelementptr inbounds i8, ptr %v34, i32 0
  %v84 = bitcast ptr %v83 to ptr
  %v85 = load i32, ptr %v84, align 4
  %v86 = udiv i64 %v76, 32
  %v87 = extractvalue { ptr, i64 } %v26, 1
  %v88 = icmp ult i64 %v86, %v87
  %v89 = extractvalue { ptr, i64 } %v26, 0
  %v90 = getelementptr inbounds float, ptr %v89, i64 %v86
  %v91 = load float, ptr %v90, align 4
  %v92 = mul i64 %v57, %v60
  %v93 = mul i64 %v64, 210
  %v94 = add i64 %v92, %v93
  %v95 = extractvalue { ptr, i64 } %v24, 0
  %v96 = extractvalue { ptr, i64 } %v24, 1
  %v97 = call float @cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk(ptr %v95, i64 %v96, i64 %v94, i64 %v73, i32 %v85, float %v91) #0
  br label %bb13
bb13:
  %v98 = fadd contract float %v68, %v97
  %v99 = add i64 %v69, 1
  br label %bb11
bb14:
  %v100 = add i64 %v64, %v45
  br label %bb9
bb15:
  br label %bb16
bb16:
  %v101 = phi float [ %v63, %bb15 ], [ %v133, %bb38 ]
  %v102 = phi i32 [ 16, %bb15 ], [ %v134, %bb38 ]
  %v103 = icmp ugt i32 %v102, 0
  %v104 = xor i1 %v103, 1
  br i1 %v104, label %bb18, label %bb17
bb17:
  %v105 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v101, i32 %v102, i32 31) #0
  br label %bb38
bb18:
  %v106 = icmp eq i64 %v37, 0
  %v107 = icmp eq i64 %v37, 0
  br i1 %v107, label %bb19, label %bb21
bb19:
  %v108 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_10, i64 %v40
  br label %bb20
bb20:
  store float %v101, ptr addrspace(3) %v108, align 4
  br label %bb21
bb21:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb22
bb22:
  %v110 = icmp eq i64 %v40, 0
  br i1 %v110, label %bb23, label %bb36
bb23:
  %v111 = icmp ult i64 %v37, %v45
  %v112 = xor i1 %v111, 1
  br i1 %v112, label %bb26, label %bb24
bb24:
  %v113 = bitcast ptr addrspace(3) @__shared_mem_10 to ptr addrspace(3)
  %v114 = getelementptr inbounds float, ptr addrspace(3) %v113, i64 %v37
  br label %bb25
bb25:
  %v115 = load float, ptr addrspace(3) %v114, align 4
  br label %bb27
bb26:
  br label %bb27
bb27:
  %v116 = phi float [ %v115, %bb25 ], [ 0.0, %bb26 ]
  br label %bb28
bb28:
  %v117 = phi float [ %v116, %bb27 ], [ %v135, %bb39 ]
  %v118 = phi i32 [ 16, %bb27 ], [ %v136, %bb39 ]
  %v119 = icmp ugt i32 %v118, 0
  %v120 = xor i1 %v119, 1
  br i1 %v120, label %bb30, label %bb29
bb29:
  %v121 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v117, i32 %v118, i32 31) #0
  br label %bb39
bb30:
  %v122 = xor i1 %v106, 1
  br i1 %v122, label %bb35, label %bb31
bb31:
  %v123 = icmp eq i32 %v28, 0
  br i1 %v123, label %bb33, label %bb32
bb32:
  %v124 = extractvalue { ptr, i64 } %v27, 1
  %v125 = icmp ult i64 %v47, %v124
  %v126 = extractvalue { ptr, i64 } %v27, 0
  %v127 = getelementptr inbounds float, ptr %v126, i64 %v47
  %v128 = load float, ptr %v127, align 4
  br label %bb34
bb33:
  br label %bb34
bb34:
  %v129 = phi float [ %v128, %bb32 ], [ 0.0, %bb33 ]
  %v130 = extractvalue { ptr, i64 } %v32, 0
  %v131 = getelementptr inbounds float, ptr %v130, i64 %v47
  %v132 = fadd contract float %v117, %v129
  store float %v132, ptr %v131, align 4
  br label %bb35
bb35:
  br label %bb36
bb36:
  br label %bb37
bb37:
  ret void
bb38:
  %v133 = fadd contract float %v101, %v105
  %v134 = udiv i32 %v102, 2
  br label %bb16
bb39:
  %v135 = fadd contract float %v117, %v121
  %v136 = udiv i32 %v118, 2
  br label %bb28
bb40:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q5_0_gemm_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca [2 x i8], align 1
  %v22 = alloca [4 x i8], align 1
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v24 = zext i32 %v23 to i64
  %v25 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v26 = zext i32 %v25 to i64
  %v27 = zext i32 %v19 to i64
  %v28 = zext i32 %v17 to i64
  %v29 = mul i64 %v27, %v28
  %v30 = icmp uge i64 %v26, %v29
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb4, label %bb3
bb3:
  br label %bb36
bb4:
  %v32 = icmp eq i64 %v28, 0
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb5, label %bb37
bb5:
  %v34 = urem i64 %v26, %v28
  %v35 = udiv i64 %v26, %v28
  %v36 = zext i32 %v18 to i64
  %v37 = mul i64 %v36, 22
  br label %bb6
bb6:
  %v38 = phi float [ 0.0, %bb5 ], [ %v91, %bb20 ]
  %v39 = phi i64 [ %v24, %bb5 ], [ %v147, %bb20 ]
  %v40 = icmp ult i64 %v39, %v36
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb21, label %bb7
bb7:
  %v42 = mul i64 %v34, %v37
  %v43 = mul i64 %v39, 22
  %v44 = add i64 %v42, %v43
  %v45 = extractvalue { ptr, i64 } %v15, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb8, label %bb38
bb8:
  %v47 = extractvalue { ptr, i64 } %v15, 0
  %v48 = getelementptr inbounds i8, ptr %v47, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v50 = add i64 %v44, 1
  %v51 = icmp ult i64 %v50, %v45
  br i1 %v51, label %bb9, label %bb39
bb9:
  %v52 = extractvalue { ptr, i64 } %v15, 0
  %v53 = getelementptr inbounds i8, ptr %v52, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v55 = getelementptr inbounds [2 x i8], ptr %v21, i32 0, i64 0
  store i8 %v49, ptr %v55, align 1
  %v56 = getelementptr inbounds [2 x i8], ptr %v21, i32 0, i64 1
  store i8 %v54, ptr %v56, align 1
  %v57 = load [2 x i8], ptr %v21, align 1
  %v58 = alloca [2 x i8], align 2
  store [2 x i8] %v57, ptr %v58, align 2
  %v59 = load i16, ptr %v58, align 2
  %v60 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v59) #0
  br label %bb10
bb10:
  %v61 = add i64 %v44, 2
  %v62 = icmp ult i64 %v61, %v45
  br i1 %v62, label %bb11, label %bb40
bb11:
  %v63 = extractvalue { ptr, i64 } %v15, 0
  %v64 = getelementptr inbounds i8, ptr %v63, i64 %v61
  %v65 = load i8, ptr %v64, align 1
  %v66 = add i64 %v44, 3
  %v67 = icmp ult i64 %v66, %v45
  br i1 %v67, label %bb12, label %bb41
bb12:
  %v68 = extractvalue { ptr, i64 } %v15, 0
  %v69 = getelementptr inbounds i8, ptr %v68, i64 %v66
  %v70 = load i8, ptr %v69, align 1
  %v71 = add i64 %v44, 4
  %v72 = icmp ult i64 %v71, %v45
  br i1 %v72, label %bb13, label %bb42
bb13:
  %v73 = extractvalue { ptr, i64 } %v15, 0
  %v74 = getelementptr inbounds i8, ptr %v73, i64 %v71
  %v75 = load i8, ptr %v74, align 1
  %v76 = add i64 %v44, 5
  %v77 = icmp ult i64 %v76, %v45
  br i1 %v77, label %bb14, label %bb43
bb14:
  %v78 = extractvalue { ptr, i64 } %v15, 0
  %v79 = getelementptr inbounds i8, ptr %v78, i64 %v76
  %v80 = load i8, ptr %v79, align 1
  %v81 = getelementptr inbounds [4 x i8], ptr %v22, i32 0, i64 0
  store i8 %v65, ptr %v81, align 1
  %v82 = getelementptr inbounds [4 x i8], ptr %v22, i32 0, i64 1
  store i8 %v70, ptr %v82, align 1
  %v83 = getelementptr inbounds [4 x i8], ptr %v22, i32 0, i64 2
  store i8 %v75, ptr %v83, align 1
  %v84 = getelementptr inbounds [4 x i8], ptr %v22, i32 0, i64 3
  store i8 %v80, ptr %v84, align 1
  %v85 = load [4 x i8], ptr %v22, align 1
  %v86 = alloca [4 x i8], align 4
  store [4 x i8] %v85, ptr %v86, align 4
  %v87 = load i32, ptr %v86, align 4
  %v88 = mul i64 %v35, %v36
  %v89 = add i64 %v88, %v39
  %v90 = mul i64 %v89, 32
  br label %bb15
bb15:
  %v91 = phi float [ %v38, %bb14 ], [ %v145, %bb19 ]
  %v92 = phi i64 [ 0, %bb14 ], [ %v146, %bb19 ]
  %v93 = icmp ult i64 %v92, 16
  %v94 = xor i1 %v93, 1
  br i1 %v94, label %bb20, label %bb16
bb16:
  %v95 = add i64 %v44, 6
  %v96 = add i64 %v95, %v92
  %v97 = icmp ult i64 %v96, %v45
  br i1 %v97, label %bb17, label %bb44
bb17:
  %v98 = extractvalue { ptr, i64 } %v15, 0
  %v99 = getelementptr inbounds i8, ptr %v98, i64 %v96
  %v100 = load i8, ptr %v99, align 1
  %v101 = trunc i64 %v92 to i32
  %v102 = and i32 %v101, 31
  %v103 = lshr i32 %v87, %v102
  %v104 = and i32 %v103, 1
  %v105 = bitcast i32 %v104 to i32
  %v106 = and i32 4, 31
  %v107 = shl i32 %v105, %v106
  %v108 = and i8 %v100, 15
  %v109 = zext i8 %v108 to i32
  %v110 = or i32 %v107, %v109
  %v111 = sub i32 %v110, 16
  %v112 = add i64 %v92, 16
  %v113 = trunc i64 %v112 to i32
  %v114 = and i32 %v113, 31
  %v115 = lshr i32 %v87, %v114
  %v116 = and i32 %v115, 1
  %v117 = bitcast i32 %v116 to i32
  %v118 = and i32 4, 31
  %v119 = shl i32 %v117, %v118
  %v120 = trunc i32 4 to i8
  %v121 = and i8 %v120, 7
  %v122 = lshr i8 %v100, %v121
  %v123 = zext i8 %v122 to i32
  %v124 = or i32 %v119, %v123
  %v125 = sub i32 %v124, 16
  %v126 = sitofp i32 %v111 to float
  %v127 = fmul contract float %v60, %v126
  %v128 = add i64 %v90, %v92
  %v129 = extractvalue { ptr, i64 } %v16, 1
  %v130 = icmp ult i64 %v128, %v129
  br i1 %v130, label %bb18, label %bb45
bb18:
  %v131 = extractvalue { ptr, i64 } %v16, 0
  %v132 = getelementptr inbounds float, ptr %v131, i64 %v128
  %v133 = load float, ptr %v132, align 4
  %v134 = fmul contract float %v127, %v133
  %v135 = sitofp i32 %v125 to float
  %v136 = fmul contract float %v60, %v135
  %v137 = add i64 %v90, %v92
  %v138 = add i64 %v137, 16
  %v139 = icmp ult i64 %v138, %v129
  br i1 %v139, label %bb19, label %bb46
bb19:
  %v140 = extractvalue { ptr, i64 } %v16, 0
  %v141 = getelementptr inbounds float, ptr %v140, i64 %v138
  %v142 = load float, ptr %v141, align 4
  %v143 = fmul contract float %v136, %v142
  %v144 = fadd contract float %v134, %v143
  %v145 = fadd contract float %v91, %v144
  %v146 = add i64 %v92, 1
  br label %bb15
bb20:
  %v147 = add i64 %v39, 32
  br label %bb6
bb21:
  %v148 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_11, i64 %v24
  br label %bb22
bb22:
  store float %v38, ptr addrspace(3) %v148, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb23
bb23:
  br label %bb24
bb24:
  %v150 = phi i64 [ 16, %bb23 ], [ %v163, %bb31 ]
  %v151 = icmp ugt i64 %v150, 0
  %v152 = xor i1 %v151, 1
  br i1 %v152, label %bb32, label %bb25
bb25:
  %v153 = icmp ult i64 %v24, %v150
  %v154 = xor i1 %v153, 1
  br i1 %v154, label %bb29, label %bb26
bb26:
  %v155 = bitcast ptr addrspace(3) @__shared_mem_11 to ptr addrspace(3)
  %v156 = add i64 %v24, %v150
  %v157 = getelementptr inbounds float, ptr addrspace(3) %v155, i64 %v156
  br label %bb27
bb27:
  %v158 = load float, ptr addrspace(3) %v157, align 4
  %v159 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_11, i64 %v24
  br label %bb28
bb28:
  %v160 = load float, ptr addrspace(3) %v159, align 4
  %v161 = fadd contract float %v160, %v158
  store float %v161, ptr addrspace(3) %v159, align 4
  br label %bb30
bb29:
  br label %bb30
bb30:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb31
bb31:
  %v163 = udiv i64 %v150, 2
  br label %bb24
bb32:
  %v164 = icmp eq i64 %v24, 0
  br i1 %v164, label %bb33, label %bb35
bb33:
  %v165 = bitcast ptr addrspace(3) @__shared_mem_11 to ptr addrspace(3)
  %v166 = getelementptr inbounds float, ptr addrspace(3) %v165, i64 0
  br label %bb34
bb34:
  %v167 = load float, ptr addrspace(3) %v166, align 4
  %v168 = extractvalue { ptr, i64 } %v20, 0
  %v169 = getelementptr inbounds float, ptr %v168, i64 %v26
  store float %v167, ptr %v169, align 4
  br label %bb35
bb35:
  br label %bb36
bb36:
  ret void
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_q4k_project_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, ptr %v15, i64 %v16) #0 {
entry:
  %v17 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v1, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v3, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v5, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v7, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v15, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v16, 1
  br label %bb0
bb0:
  %v27 = phi { ptr, i64 } [ %v18, %entry ]
  %v28 = phi { ptr, i64 } [ %v20, %entry ]
  %v29 = phi { ptr, i64 } [ %v22, %entry ]
  %v30 = phi { ptr, i64 } [ %v24, %entry ]
  %v31 = phi i32 [ %v8, %entry ]
  %v32 = phi i32 [ %v9, %entry ]
  %v33 = phi i32 [ %v10, %entry ]
  %v34 = phi i32 [ %v11, %entry ]
  %v35 = phi i32 [ %v12, %entry ]
  %v36 = phi i32 [ %v13, %entry ]
  %v37 = phi i32 [ %v14, %entry ]
  %v38 = phi { ptr, i64 } [ %v26, %entry ]
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v40 = zext i32 %v39 to i64
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v42 = zext i32 %v41 to i64
  %v43 = zext i32 %v31 to i64
  %v44 = zext i32 %v32 to i64
  %v45 = mul i64 %v43, %v44
  %v46 = zext i32 %v33 to i64
  %v47 = mul i64 %v45, %v46
  %v48 = icmp uge i64 %v42, %v47
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb4, label %bb3
bb3:
  br label %bb30
bb4:
  %v50 = icmp eq i64 %v46, 0
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb5, label %bb31
bb5:
  %v52 = urem i64 %v42, %v46
  %v53 = udiv i64 %v42, %v46
  %v54 = extractvalue { ptr, i64 } %v30, 1
  %v55 = icmp ult i64 %v53, %v54
  br i1 %v55, label %bb6, label %bb32
bb6:
  %v56 = extractvalue { ptr, i64 } %v30, 0
  %v57 = getelementptr inbounds i32, ptr %v56, i64 %v53
  %v58 = load i32, ptr %v57, align 4
  %v59 = zext i32 %v58 to i64
  %v60 = icmp eq i64 %v44, 0
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb7, label %bb33
bb7:
  %v62 = udiv i64 %v59, %v44
  %v63 = extractvalue { ptr, i64 } %v29, 1
  %v64 = icmp ult i64 %v59, %v63
  br i1 %v64, label %bb8, label %bb34
bb8:
  %v65 = extractvalue { ptr, i64 } %v29, 0
  %v66 = getelementptr inbounds i32, ptr %v65, i64 %v59
  %v67 = load i32, ptr %v66, align 4
  %v68 = zext i32 %v67 to i64
  %v69 = zext i32 %v34 to i64
  %v70 = udiv i64 %v69, 256
  %v71 = mul i64 %v70, 144
  %v72 = zext i32 %v35 to i64
  %v73 = mul i64 %v72, %v71
  %v74 = icmp eq i32 %v37, 0
  br i1 %v74, label %bb10, label %bb9
bb9:
  br label %bb11
bb10:
  br label %bb11
bb11:
  %v75 = phi i64 [ %v59, %bb9 ], [ %v62, %bb10 ]
  br label %bb12
bb12:
  %v76 = phi float [ 0.0, %bb11 ], [ %v95, %bb14 ]
  %v77 = phi i64 [ %v40, %bb11 ], [ %v96, %bb14 ]
  %v78 = icmp ult i64 %v77, %v70
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb15, label %bb13
bb13:
  %v80 = mul i64 %v68, %v73
  %v81 = zext i32 %v36 to i64
  %v82 = add i64 %v81, %v52
  %v83 = mul i64 %v82, %v71
  %v84 = add i64 %v80, %v83
  %v85 = mul i64 %v77, 144
  %v86 = add i64 %v84, %v85
  %v87 = mul i64 %v75, %v70
  %v88 = add i64 %v87, %v77
  %v89 = mul i64 %v88, 256
  %v90 = extractvalue { ptr, i64 } %v27, 0
  %v91 = extractvalue { ptr, i64 } %v27, 1
  %v92 = extractvalue { ptr, i64 } %v28, 0
  %v93 = extractvalue { ptr, i64 } %v28, 1
  %v94 = call float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v90, i64 %v91, i64 %v86, ptr %v92, i64 %v93, i64 %v89, i32 1) #0
  br label %bb14
bb14:
  %v95 = fadd contract float %v76, %v94
  %v96 = add i64 %v77, 32
  br label %bb12
bb15:
  %v97 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_12, i64 %v40
  br label %bb16
bb16:
  store float %v76, ptr addrspace(3) %v97, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb17
bb17:
  br label %bb18
bb18:
  %v99 = phi i64 [ 16, %bb17 ], [ %v112, %bb25 ]
  %v100 = icmp ugt i64 %v99, 0
  %v101 = xor i1 %v100, 1
  br i1 %v101, label %bb26, label %bb19
bb19:
  %v102 = icmp ult i64 %v40, %v99
  %v103 = xor i1 %v102, 1
  br i1 %v103, label %bb23, label %bb20
bb20:
  %v104 = bitcast ptr addrspace(3) @__shared_mem_12 to ptr addrspace(3)
  %v105 = add i64 %v40, %v99
  %v106 = getelementptr inbounds float, ptr addrspace(3) %v104, i64 %v105
  br label %bb21
bb21:
  %v107 = load float, ptr addrspace(3) %v106, align 4
  %v108 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_12, i64 %v40
  br label %bb22
bb22:
  %v109 = load float, ptr addrspace(3) %v108, align 4
  %v110 = fadd contract float %v109, %v107
  store float %v110, ptr addrspace(3) %v108, align 4
  br label %bb24
bb23:
  br label %bb24
bb24:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb25
bb25:
  %v112 = udiv i64 %v99, 2
  br label %bb18
bb26:
  %v113 = icmp eq i64 %v40, 0
  br i1 %v113, label %bb27, label %bb29
bb27:
  %v114 = bitcast ptr addrspace(3) @__shared_mem_12 to ptr addrspace(3)
  %v115 = getelementptr inbounds float, ptr addrspace(3) %v114, i64 0
  br label %bb28
bb28:
  %v116 = load float, ptr addrspace(3) %v115, align 4
  %v117 = mul i64 %v59, %v46
  %v118 = add i64 %v117, %v52
  %v119 = extractvalue { ptr, i64 } %v38, 0
  %v120 = getelementptr inbounds float, ptr %v119, i64 %v118
  store float %v116, ptr %v120, align 4
  br label %bb29
bb29:
  br label %bb30
bb30:
  ret void
bb31:
  call void @llvm.trap() #0
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @add_inplace_f32(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = alloca {  }, align 1
  %v11 = bitcast ptr %v10 to ptr
  %v12 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v11) #0
  br label %bb1
bb1:
  %v13 = icmp eq i64 %v12, 18446744073709551615
  br i1 %v13, label %bb9, label %bb6
bb2:
  %v14 = extractvalue { ptr } %v33, 0
  %v15 = extractvalue { ptr, i64 } %v8, 1
  %v16 = icmp ult i64 %v12, %v15
  br i1 %v16, label %bb3, label %bb13
bb3:
  %v17 = extractvalue { ptr, i64 } %v8, 0
  %v18 = getelementptr inbounds float, ptr %v17, i64 %v12
  %v19 = load float, ptr %v18, align 4
  %v20 = load float, ptr %v14, align 4
  %v21 = fadd contract float %v20, %v19
  store float %v21, ptr %v14, align 4
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v22 = extractvalue { ptr, i64 } %v9, 1
  %v23 = icmp ult i64 %v12, %v22
  %v24 = xor i1 %v23, 1
  br i1 %v24, label %bb8, label %bb7
bb7:
  %v25 = extractvalue { ptr, i64 } %v9, 0
  %v26 = getelementptr inbounds float, ptr %v25, i64 %v12
  %v27 = insertvalue { ptr } undef, ptr %v26, 0
  %v28 = extractvalue { ptr } %v27, 0
  br label %bb10
bb8:
  br label %bb9
bb9:
  %v29 = inttoptr i64 0 to ptr
  %v30 = insertvalue { ptr } undef, ptr %v29, 0
  %v31 = extractvalue { ptr } %v30, 0
  br label %bb10
bb10:
  %v32 = phi ptr [ %v28, %bb7 ], [ %v31, %bb9 ]
  %v33 = insertvalue { ptr } undef, ptr %v32, 0
  %v34 = extractvalue { ptr } %v33, 0
  %v35 = ptrtoint ptr %v34 to i64
  %v36 = sub i64 %v35, 0
  %v37 = icmp ule i64 %v36, 0
  %v38 = add i64 %v36, 0
  %v39 = select i1 %v37, i64 %v38, i64 1
  %v40 = icmp eq i64 %v39, 1
  br i1 %v40, label %bb2, label %bb11
bb11:
  %v41 = icmp eq i64 %v39, 0
  br i1 %v41, label %bb4, label %bb12
bb12:
  unreachable
bb13:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q5_0_gemm_element(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = alloca [4 x i8], align 1
  %v24 = bitcast ptr %v21 to ptr
  %v25 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v24) #0
  br label %bb1
bb1:
  %v26 = zext i32 %v17 to i64
  %v27 = zext i32 %v19 to i64
  %v28 = mul i64 %v26, %v27
  %v29 = icmp uge i64 %v25, %v28
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb3, label %bb2
bb2:
  br label %bb24
bb3:
  %v31 = icmp eq i64 %v26, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb4, label %bb32
bb4:
  %v33 = urem i64 %v25, %v26
  %v34 = udiv i64 %v25, %v26
  %v35 = zext i32 %v18 to i64
  %v36 = mul i64 %v35, 22
  br label %bb5
bb5:
  %v37 = phi float [ 0.0, %bb4 ], [ %v90, %bb19 ]
  %v38 = phi i64 [ 0, %bb4 ], [ %v146, %bb19 ]
  %v39 = icmp ult i64 %v38, %v35
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb20, label %bb6
bb6:
  %v41 = mul i64 %v33, %v36
  %v42 = mul i64 %v38, 22
  %v43 = add i64 %v41, %v42
  %v44 = extractvalue { ptr, i64 } %v15, 1
  %v45 = icmp ult i64 %v43, %v44
  br i1 %v45, label %bb7, label %bb33
bb7:
  %v46 = extractvalue { ptr, i64 } %v15, 0
  %v47 = getelementptr inbounds i8, ptr %v46, i64 %v43
  %v48 = load i8, ptr %v47, align 1
  %v49 = add i64 %v43, 1
  %v50 = icmp ult i64 %v49, %v44
  br i1 %v50, label %bb8, label %bb34
bb8:
  %v51 = extractvalue { ptr, i64 } %v15, 0
  %v52 = getelementptr inbounds i8, ptr %v51, i64 %v49
  %v53 = load i8, ptr %v52, align 1
  %v54 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v48, ptr %v54, align 1
  %v55 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v53, ptr %v55, align 1
  %v56 = load [2 x i8], ptr %v22, align 1
  %v57 = alloca [2 x i8], align 2
  store [2 x i8] %v56, ptr %v57, align 2
  %v58 = load i16, ptr %v57, align 2
  %v59 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v58) #0
  br label %bb9
bb9:
  %v60 = add i64 %v43, 2
  %v61 = icmp ult i64 %v60, %v44
  br i1 %v61, label %bb10, label %bb35
bb10:
  %v62 = extractvalue { ptr, i64 } %v15, 0
  %v63 = getelementptr inbounds i8, ptr %v62, i64 %v60
  %v64 = load i8, ptr %v63, align 1
  %v65 = add i64 %v43, 3
  %v66 = icmp ult i64 %v65, %v44
  br i1 %v66, label %bb11, label %bb36
bb11:
  %v67 = extractvalue { ptr, i64 } %v15, 0
  %v68 = getelementptr inbounds i8, ptr %v67, i64 %v65
  %v69 = load i8, ptr %v68, align 1
  %v70 = add i64 %v43, 4
  %v71 = icmp ult i64 %v70, %v44
  br i1 %v71, label %bb12, label %bb37
bb12:
  %v72 = extractvalue { ptr, i64 } %v15, 0
  %v73 = getelementptr inbounds i8, ptr %v72, i64 %v70
  %v74 = load i8, ptr %v73, align 1
  %v75 = add i64 %v43, 5
  %v76 = icmp ult i64 %v75, %v44
  br i1 %v76, label %bb13, label %bb38
bb13:
  %v77 = extractvalue { ptr, i64 } %v15, 0
  %v78 = getelementptr inbounds i8, ptr %v77, i64 %v75
  %v79 = load i8, ptr %v78, align 1
  %v80 = getelementptr inbounds [4 x i8], ptr %v23, i32 0, i64 0
  store i8 %v64, ptr %v80, align 1
  %v81 = getelementptr inbounds [4 x i8], ptr %v23, i32 0, i64 1
  store i8 %v69, ptr %v81, align 1
  %v82 = getelementptr inbounds [4 x i8], ptr %v23, i32 0, i64 2
  store i8 %v74, ptr %v82, align 1
  %v83 = getelementptr inbounds [4 x i8], ptr %v23, i32 0, i64 3
  store i8 %v79, ptr %v83, align 1
  %v84 = load [4 x i8], ptr %v23, align 1
  %v85 = alloca [4 x i8], align 4
  store [4 x i8] %v84, ptr %v85, align 4
  %v86 = load i32, ptr %v85, align 4
  %v87 = mul i64 %v34, %v35
  %v88 = add i64 %v87, %v38
  %v89 = mul i64 %v88, 32
  br label %bb14
bb14:
  %v90 = phi float [ %v37, %bb13 ], [ %v144, %bb18 ]
  %v91 = phi i64 [ 0, %bb13 ], [ %v145, %bb18 ]
  %v92 = icmp ult i64 %v91, 16
  %v93 = xor i1 %v92, 1
  br i1 %v93, label %bb19, label %bb15
bb15:
  %v94 = add i64 %v43, 6
  %v95 = add i64 %v94, %v91
  %v96 = icmp ult i64 %v95, %v44
  br i1 %v96, label %bb16, label %bb39
bb16:
  %v97 = extractvalue { ptr, i64 } %v15, 0
  %v98 = getelementptr inbounds i8, ptr %v97, i64 %v95
  %v99 = load i8, ptr %v98, align 1
  %v100 = trunc i64 %v91 to i32
  %v101 = and i32 %v100, 31
  %v102 = lshr i32 %v86, %v101
  %v103 = and i32 %v102, 1
  %v104 = bitcast i32 %v103 to i32
  %v105 = and i32 4, 31
  %v106 = shl i32 %v104, %v105
  %v107 = and i8 %v99, 15
  %v108 = zext i8 %v107 to i32
  %v109 = or i32 %v106, %v108
  %v110 = sub i32 %v109, 16
  %v111 = add i64 %v91, 16
  %v112 = trunc i64 %v111 to i32
  %v113 = and i32 %v112, 31
  %v114 = lshr i32 %v86, %v113
  %v115 = and i32 %v114, 1
  %v116 = bitcast i32 %v115 to i32
  %v117 = and i32 4, 31
  %v118 = shl i32 %v116, %v117
  %v119 = trunc i32 4 to i8
  %v120 = and i8 %v119, 7
  %v121 = lshr i8 %v99, %v120
  %v122 = zext i8 %v121 to i32
  %v123 = or i32 %v118, %v122
  %v124 = sub i32 %v123, 16
  %v125 = sitofp i32 %v110 to float
  %v126 = fmul contract float %v59, %v125
  %v127 = add i64 %v89, %v91
  %v128 = extractvalue { ptr, i64 } %v16, 1
  %v129 = icmp ult i64 %v127, %v128
  br i1 %v129, label %bb17, label %bb40
bb17:
  %v130 = extractvalue { ptr, i64 } %v16, 0
  %v131 = getelementptr inbounds float, ptr %v130, i64 %v127
  %v132 = load float, ptr %v131, align 4
  %v133 = fmul contract float %v126, %v132
  %v134 = fadd contract float %v90, %v133
  %v135 = sitofp i32 %v124 to float
  %v136 = fmul contract float %v59, %v135
  %v137 = add i64 %v89, %v91
  %v138 = add i64 %v137, 16
  %v139 = icmp ult i64 %v138, %v128
  br i1 %v139, label %bb18, label %bb41
bb18:
  %v140 = extractvalue { ptr, i64 } %v16, 0
  %v141 = getelementptr inbounds float, ptr %v140, i64 %v138
  %v142 = load float, ptr %v141, align 4
  %v143 = fmul contract float %v136, %v142
  %v144 = fadd contract float %v134, %v143
  %v145 = add i64 %v91, 1
  br label %bb14
bb19:
  %v146 = add i64 %v38, 1
  br label %bb5
bb20:
  %v147 = icmp eq i64 %v25, 18446744073709551615
  br i1 %v147, label %bb28, label %bb25
bb21:
  %v148 = extractvalue { ptr } %v160, 0
  store float %v37, ptr %v148, align 4
  br label %bb23
bb22:
  br label %bb23
bb23:
  br label %bb24
bb24:
  ret void
bb25:
  %v149 = extractvalue { ptr, i64 } %v20, 1
  %v150 = icmp ult i64 %v25, %v149
  %v151 = xor i1 %v150, 1
  br i1 %v151, label %bb27, label %bb26
bb26:
  %v152 = extractvalue { ptr, i64 } %v20, 0
  %v153 = getelementptr inbounds float, ptr %v152, i64 %v25
  %v154 = insertvalue { ptr } undef, ptr %v153, 0
  %v155 = extractvalue { ptr } %v154, 0
  br label %bb29
bb27:
  br label %bb28
bb28:
  %v156 = inttoptr i64 0 to ptr
  %v157 = insertvalue { ptr } undef, ptr %v156, 0
  %v158 = extractvalue { ptr } %v157, 0
  br label %bb29
bb29:
  %v159 = phi ptr [ %v155, %bb26 ], [ %v158, %bb28 ]
  %v160 = insertvalue { ptr } undef, ptr %v159, 0
  %v161 = extractvalue { ptr } %v160, 0
  %v162 = ptrtoint ptr %v161 to i64
  %v163 = sub i64 %v162, 0
  %v164 = icmp ule i64 %v163, 0
  %v165 = add i64 %v163, 0
  %v166 = select i1 %v164, i64 %v165, i64 1
  %v167 = icmp eq i64 %v166, 1
  br i1 %v167, label %bb21, label %bb30
bb30:
  %v168 = icmp eq i64 %v166, 0
  br i1 %v168, label %bb22, label %bb31
bb31:
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_prefix_offsets(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = alloca {  }, align 1
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v16) #0
  br label %bb1
bb1:
  %v18 = icmp eq i64 %v17, 0
  br i1 %v18, label %bb3, label %bb2
bb2:
  br label %bb8
bb3:
  br label %bb4
bb4:
  %v19 = phi i32 [ 0, %bb3 ], [ %v32, %bb6 ]
  %v20 = phi i64 [ 0, %bb3 ], [ %v33, %bb6 ]
  %v21 = extractvalue { ptr, i64 } %v12, 1
  %v22 = icmp ult i64 %v20, %v21
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb7, label %bb5
bb5:
  %v24 = extractvalue { ptr, i64 } %v13, 0
  %v25 = getelementptr inbounds i32, ptr %v24, i64 %v20
  store i32 %v19, ptr %v25, align 4
  %v26 = extractvalue { ptr, i64 } %v14, 0
  %v27 = getelementptr inbounds i32, ptr %v26, i64 %v20
  store i32 0, ptr %v27, align 4
  %v28 = icmp ult i64 %v20, %v21
  br i1 %v28, label %bb6, label %bb9
bb6:
  %v29 = extractvalue { ptr, i64 } %v12, 0
  %v30 = getelementptr inbounds i32, ptr %v29, i64 %v20
  %v31 = load i32, ptr %v30, align 4
  %v32 = add i32 %v19, %v31
  %v33 = add i64 %v20, 1
  br label %bb4
bb7:
  br label %bb8
bb8:
  ret void
bb9:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @kv_write_batch(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15) #0 {
entry:
  %v16 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v1, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v3, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v5, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v7, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v17, %entry ]
  %v25 = phi { ptr, i64 } [ %v19, %entry ]
  %v26 = phi { ptr, i64 } [ %v21, %entry ]
  %v27 = phi { ptr, i64 } [ %v23, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = phi i32 [ %v15, %entry ]
  %v36 = alloca {  }, align 1
  %v37 = bitcast ptr %v36 to ptr
  %v38 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v37) #0
  br label %bb1
bb1:
  %v39 = trunc i64 %v38 to i32
  %v40 = mul i32 %v28, %v33
  %v41 = icmp uge i32 %v39, %v40
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb3, label %bb2
bb2:
  br label %bb13
bb3:
  %v43 = icmp eq i32 %v33, 0
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb4, label %bb14
bb4:
  %v45 = udiv i32 %v39, %v33
  %v46 = urem i32 %v39, %v33
  %v47 = zext i32 %v45 to i64
  %v48 = extractvalue { ptr, i64 } %v27, 1
  %v49 = icmp ult i64 %v47, %v48
  br i1 %v49, label %bb5, label %bb15
bb5:
  %v50 = extractvalue { ptr, i64 } %v27, 0
  %v51 = getelementptr inbounds i32, ptr %v50, i64 %v47
  %v52 = load i32, ptr %v51, align 4
  %v53 = icmp eq i32 %v34, 0
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb6, label %bb16
bb6:
  %v55 = udiv i32 %v52, %v34
  %v56 = urem i32 %v52, %v34
  %v57 = mul i32 %v45, %v30
  %v58 = add i32 %v57, %v29
  %v59 = mul i32 %v58, %v32
  %v60 = add i32 %v59, %v55
  %v61 = zext i32 %v60 to i64
  %v62 = extractvalue { ptr, i64 } %v26, 1
  %v63 = icmp ult i64 %v61, %v62
  br i1 %v63, label %bb7, label %bb17
bb7:
  %v64 = extractvalue { ptr, i64 } %v26, 0
  %v65 = getelementptr inbounds i32, ptr %v64, i64 %v61
  %v66 = load i32, ptr %v65, align 4
  %v67 = zext i32 %v66 to i64
  %v68 = zext i32 %v33 to i64
  %v69 = mul i64 %v68, 2
  %v70 = icmp eq i32 %v31, 0
  br i1 %v70, label %bb9, label %bb8
bb8:
  %v71 = zext i32 %v34 to i64
  %v72 = mul i64 %v71, %v69
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v73 = phi i64 [ %v72, %bb8 ], [ 0, %bb9 ]
  %v74 = zext i32 %v35 to i64
  %v75 = mul i64 %v67, %v74
  %v76 = add i64 %v75, %v73
  %v77 = zext i32 %v56 to i64
  %v78 = mul i64 %v77, %v69
  %v79 = add i64 %v76, %v78
  %v80 = zext i32 %v39 to i64
  %v81 = extractvalue { ptr, i64 } %v24, 1
  %v82 = icmp ult i64 %v80, %v81
  br i1 %v82, label %bb11, label %bb18
bb11:
  %v83 = extractvalue { ptr, i64 } %v24, 0
  %v84 = getelementptr inbounds float, ptr %v83, i64 %v80
  %v85 = load float, ptr %v84, align 4
  %v86 = call i16 @cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits(float %v85) #0
  br label %bb12
bb12:
  %v87 = zext i32 %v46 to i64
  %v88 = mul i64 %v87, 2
  %v89 = add i64 %v79, %v88
  %v90 = and i16 %v86, 255
  %v91 = extractvalue { ptr, i64 } %v25, 0
  %v92 = getelementptr inbounds i8, ptr %v91, i64 %v89
  %v93 = trunc i16 %v90 to i8
  store i8 %v93, ptr %v92, align 1
  %v94 = trunc i32 8 to i16
  %v95 = and i16 %v94, 15
  %v96 = lshr i16 %v86, %v95
  %v97 = add i64 %v89, 1
  %v98 = getelementptr inbounds i8, ptr %v91, i64 %v97
  %v99 = trunc i16 %v96 to i8
  store i8 %v99, ptr %v98, align 1
  br label %bb13
bb13:
  ret void
bb14:
  call void @llvm.trap() #0
  unreachable
bb15:
  call void @llvm.trap() #0
  unreachable
bb16:
  call void @llvm.trap() #0
  unreachable
bb17:
  call void @llvm.trap() #0
  unreachable
bb18:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @attention_paged_batch_heads(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, float %v12, i32 %v13, i32 %v14, i32 %v15, i32 %v16, i32 %v17, i32 %v18, ptr %v19, i64 %v20) #0 {
entry:
  %v21 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v1, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v3, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v5, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v7, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v19, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v20, 1
  br label %bb0
bb0:
  %v31 = phi { ptr, i64 } [ %v22, %entry ]
  %v32 = phi { ptr, i64 } [ %v24, %entry ]
  %v33 = phi { ptr, i64 } [ %v26, %entry ]
  %v34 = phi { ptr, i64 } [ %v28, %entry ]
  %v35 = phi i32 [ %v8, %entry ]
  %v36 = phi i32 [ %v9, %entry ]
  %v37 = phi i32 [ %v10, %entry ]
  %v38 = phi i32 [ %v11, %entry ]
  %v39 = phi float [ %v12, %entry ]
  %v40 = phi i32 [ %v13, %entry ]
  %v41 = phi i32 [ %v14, %entry ]
  %v42 = phi i32 [ %v15, %entry ]
  %v43 = phi i32 [ %v16, %entry ]
  %v44 = phi i32 [ %v17, %entry ]
  %v45 = phi i32 [ %v18, %entry ]
  %v46 = phi { ptr, i64 } [ %v30, %entry ]
  %v47 = alloca {  }, align 1
  %v48 = bitcast ptr %v47 to ptr
  %v49 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v48) #0
  br label %bb1
bb1:
  %v50 = trunc i64 %v49 to i32
  %v51 = mul i32 %v35, %v36
  %v52 = icmp uge i32 %v50, %v51
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb3, label %bb2
bb2:
  br label %bb43
bb3:
  %v54 = icmp eq i32 %v36, 0
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb4, label %bb46
bb4:
  %v56 = udiv i32 %v50, %v36
  %v57 = urem i32 %v50, %v36
  %v58 = zext i32 %v38 to i64
  %v59 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v37, i32 1) #0
  br label %bb5
bb5:
  %v60 = icmp eq i32 %v59, 0
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb6, label %bb47
bb6:
  %v62 = udiv i32 %v36, %v59
  %v63 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v62, i32 1) #0
  br label %bb7
bb7:
  %v64 = icmp eq i32 %v63, 0
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb8, label %bb48
bb8:
  %v66 = udiv i32 %v57, %v63
  %v67 = zext i32 %v66 to i64
  %v68 = zext i32 %v45 to i64
  %v69 = zext i32 %v43 to i64
  %v70 = mul i64 %v68, 2
  %v71 = mul i64 %v69, %v70
  %v72 = zext i32 %v56 to i64
  %v73 = zext i32 %v36 to i64
  %v74 = mul i64 %v72, %v73
  %v75 = zext i32 %v57 to i64
  %v76 = add i64 %v74, %v75
  %v77 = mul i64 %v76, %v58
  br label %bb9
bb9:
  %v78 = phi i64 [ 0, %bb8 ], [ %v84, %bb10 ]
  %v79 = icmp ult i64 %v78, %v58
  %v80 = xor i1 %v79, 1
  br i1 %v80, label %bb11, label %bb10
bb10:
  %v81 = add i64 %v77, %v78
  %v82 = extractvalue { ptr, i64 } %v46, 0
  %v83 = getelementptr inbounds float, ptr %v82, i64 %v81
  store float 0.0, ptr %v83, align 4
  %v84 = add i64 %v78, 1
  br label %bb9
bb11:
  br label %bb12
bb12:
  %v85 = phi float [ 0.0, %bb11 ], [ %v161, %bb35 ]
  %v86 = phi float [ 0.0, %bb11 ], [ %v215, %bb35 ]
  %v87 = phi i1 [ 0, %bb11 ], [ 1, %bb35 ]
  %v88 = phi i64 [ 0, %bb11 ], [ %v202, %bb35 ]
  %v89 = extractvalue { ptr, i64 } %v34, 1
  %v90 = icmp ult i64 %v72, %v89
  br i1 %v90, label %bb13, label %bb49
bb13:
  %v91 = extractvalue { ptr, i64 } %v34, 0
  %v92 = getelementptr inbounds i32, ptr %v91, i64 %v72
  %v93 = load i32, ptr %v92, align 4
  %v94 = zext i32 %v93 to i64
  %v95 = icmp ult i64 %v88, %v94
  %v96 = xor i1 %v95, 1
  br i1 %v96, label %bb36, label %bb14
bb14:
  %v97 = icmp eq i64 %v69, 0
  %v98 = xor i1 %v97, 1
  br i1 %v98, label %bb15, label %bb50
bb15:
  %v99 = udiv i64 %v88, %v69
  %v100 = urem i64 %v88, %v69
  %v101 = mul i32 %v56, %v41
  %v102 = add i32 %v101, %v40
  %v103 = mul i32 %v102, %v42
  %v104 = zext i32 %v103 to i64
  %v105 = add i64 %v104, %v99
  %v106 = extractvalue { ptr, i64 } %v33, 1
  %v107 = icmp ult i64 %v105, %v106
  br i1 %v107, label %bb16, label %bb51
bb16:
  %v108 = extractvalue { ptr, i64 } %v33, 0
  %v109 = getelementptr inbounds i32, ptr %v108, i64 %v105
  %v110 = load i32, ptr %v109, align 4
  %v111 = zext i32 %v110 to i64
  %v112 = zext i32 %v44 to i64
  %v113 = mul i64 %v111, %v112
  %v114 = mul i64 %v100, %v70
  %v115 = add i64 %v113, %v114
  %v116 = mul i64 %v67, %v58
  %v117 = mul i64 %v116, 2
  %v118 = add i64 %v115, %v117
  br label %bb17
bb17:
  %v119 = phi i64 [ 0, %bb16 ], [ %v152, %bb22 ]
  %v120 = phi float [ 0.0, %bb16 ], [ %v151, %bb22 ]
  %v121 = icmp ult i64 %v119, %v58
  %v122 = xor i1 %v121, 1
  br i1 %v122, label %bb23, label %bb18
bb18:
  %v123 = mul i64 %v119, 2
  %v124 = add i64 %v118, %v123
  %v125 = extractvalue { ptr, i64 } %v32, 1
  %v126 = icmp ult i64 %v124, %v125
  br i1 %v126, label %bb19, label %bb52
bb19:
  %v127 = extractvalue { ptr, i64 } %v32, 0
  %v128 = getelementptr inbounds i8, ptr %v127, i64 %v124
  %v129 = load i8, ptr %v128, align 1
  %v130 = zext i8 %v129 to i16
  %v131 = mul i64 %v119, 2
  %v132 = add i64 %v118, %v131
  %v133 = add i64 %v132, 1
  %v134 = icmp ult i64 %v133, %v125
  br i1 %v134, label %bb20, label %bb53
bb20:
  %v135 = extractvalue { ptr, i64 } %v32, 0
  %v136 = getelementptr inbounds i8, ptr %v135, i64 %v133
  %v137 = load i8, ptr %v136, align 1
  %v138 = zext i8 %v137 to i16
  %v139 = trunc i32 8 to i16
  %v140 = and i16 %v139, 15
  %v141 = shl i16 %v138, %v140
  %v142 = or i16 %v130, %v141
  %v143 = add i64 %v77, %v119
  %v144 = extractvalue { ptr, i64 } %v31, 1
  %v145 = icmp ult i64 %v143, %v144
  br i1 %v145, label %bb21, label %bb54
bb21:
  %v146 = extractvalue { ptr, i64 } %v31, 0
  %v147 = getelementptr inbounds float, ptr %v146, i64 %v143
  %v148 = load float, ptr %v147, align 4
  %v149 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v142) #0
  br label %bb22
bb22:
  %v150 = fmul contract float %v148, %v149
  %v151 = fadd contract float %v120, %v150
  %v152 = add i64 %v119, 1
  br label %bb17
bb23:
  %v153 = fmul contract float %v120, %v39
  %v154 = xor i1 %v87, 1
  br i1 %v154, label %bb25, label %bb24
bb24:
  %v155 = fcmp ogt float %v153, %v85
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb27, label %bb26
bb25:
  br label %bb29
bb26:
  %v157 = fsub contract float %v85, %v153
  %v158 = call float @__nv_expf(float %v157) #0
  br label %bb44
bb27:
  br label %bb28
bb28:
  %v159 = phi float [ %v85, %bb27 ], [ %v153, %bb44 ]
  %v160 = phi float [ 1.0, %bb27 ], [ %v158, %bb44 ]
  br label %bb29
bb29:
  %v161 = phi float [ %v153, %bb25 ], [ %v159, %bb28 ]
  %v162 = phi float [ 0.0, %bb25 ], [ %v160, %bb28 ]
  %v163 = fsub contract float %v153, %v161
  %v164 = call float @__nv_expf(float %v163) #0
  br label %bb45
bb30:
  %v165 = phi i64 [ %v201, %bb34 ], [ 0, %bb45 ]
  %v166 = icmp ult i64 %v165, %v58
  %v167 = xor i1 %v166, 1
  br i1 %v167, label %bb35, label %bb31
bb31:
  %v168 = add i64 %v113, %v71
  %v169 = add i64 %v168, %v114
  %v170 = add i64 %v169, %v117
  %v171 = mul i64 %v165, 2
  %v172 = add i64 %v170, %v171
  %v173 = extractvalue { ptr, i64 } %v32, 1
  %v174 = icmp ult i64 %v172, %v173
  br i1 %v174, label %bb32, label %bb55
bb32:
  %v175 = extractvalue { ptr, i64 } %v32, 0
  %v176 = getelementptr inbounds i8, ptr %v175, i64 %v172
  %v177 = load i8, ptr %v176, align 1
  %v178 = zext i8 %v177 to i16
  %v179 = mul i64 %v165, 2
  %v180 = add i64 %v170, %v179
  %v181 = add i64 %v180, 1
  %v182 = icmp ult i64 %v181, %v173
  br i1 %v182, label %bb33, label %bb56
bb33:
  %v183 = extractvalue { ptr, i64 } %v32, 0
  %v184 = getelementptr inbounds i8, ptr %v183, i64 %v181
  %v185 = load i8, ptr %v184, align 1
  %v186 = zext i8 %v185 to i16
  %v187 = trunc i32 8 to i16
  %v188 = and i16 %v187, 15
  %v189 = shl i16 %v186, %v188
  %v190 = or i16 %v178, %v189
  %v191 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v190) #0
  br label %bb34
bb34:
  %v192 = add i64 %v77, %v165
  %v193 = extractvalue { ptr, i64 } %v46, 0
  %v194 = getelementptr inbounds float, ptr %v193, i64 %v192
  %v195 = load float, ptr %v194, align 4
  %v196 = fmul contract float %v195, %v162
  %v197 = fmul contract float %v164, %v191
  %v198 = add i64 %v77, %v165
  %v199 = getelementptr inbounds float, ptr %v193, i64 %v198
  %v200 = fadd contract float %v196, %v197
  store float %v200, ptr %v199, align 4
  %v201 = add i64 %v165, 1
  br label %bb30
bb35:
  %v202 = add i64 %v88, 1
  br label %bb12
bb36:
  %v203 = fcmp ogt float %v86, 0.0
  %v204 = xor i1 %v203, 1
  br i1 %v204, label %bb38, label %bb37
bb37:
  br label %bb39
bb38:
  br label %bb42
bb39:
  %v205 = phi i64 [ 0, %bb37 ], [ %v213, %bb40 ]
  %v206 = icmp ult i64 %v205, %v58
  %v207 = xor i1 %v206, 1
  br i1 %v207, label %bb41, label %bb40
bb40:
  %v208 = add i64 %v77, %v205
  %v209 = extractvalue { ptr, i64 } %v46, 0
  %v210 = getelementptr inbounds float, ptr %v209, i64 %v208
  %v211 = load float, ptr %v210, align 4
  %v212 = fdiv contract float %v211, %v86
  store float %v212, ptr %v210, align 4
  %v213 = add i64 %v205, 1
  br label %bb39
bb41:
  br label %bb42
bb42:
  br label %bb43
bb43:
  ret void
bb44:
  br label %bb28
bb45:
  %v214 = fmul contract float %v86, %v162
  %v215 = fadd contract float %v214, %v164
  br label %bb30
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
}

declare float @__nv_sinf(float)
declare float @__nv_cosf(float)

define ptx_kernel void @rope_batch(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr %v10, i64 %v11) #0 {
entry:
  %v12 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v1, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v3, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v5, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v11, 1
  br label %bb0
bb0:
  %v20 = phi { ptr, i64 } [ %v13, %entry ]
  %v21 = phi { ptr, i64 } [ %v15, %entry ]
  %v22 = phi { ptr, i64 } [ %v17, %entry ]
  %v23 = phi i32 [ %v6, %entry ]
  %v24 = phi i32 [ %v7, %entry ]
  %v25 = phi i32 [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi { ptr, i64 } [ %v19, %entry ]
  %v28 = alloca {  }, align 1
  %v29 = bitcast ptr %v28 to ptr
  %v30 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v29) #0
  br label %bb1
bb1:
  %v31 = zext i32 %v26 to i64
  %v32 = icmp uge i64 %v30, %v31
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb3, label %bb2
bb2:
  br label %bb15
bb3:
  %v34 = zext i32 %v24 to i64
  %v35 = icmp eq i64 %v34, 0
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb4, label %bb18
bb4:
  %v37 = urem i64 %v30, %v34
  %v38 = sub i64 %v30, %v37
  %v39 = zext i32 %v23 to i64
  %v40 = mul i64 %v39, %v34
  %v41 = icmp eq i64 %v40, 0
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb5, label %bb19
bb5:
  %v43 = udiv i64 %v30, %v40
  %v44 = zext i32 %v25 to i64
  %v45 = icmp uge i64 %v37, %v44
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb8, label %bb6
bb6:
  %v47 = extractvalue { ptr, i64 } %v20, 1
  %v48 = icmp ult i64 %v30, %v47
  br i1 %v48, label %bb7, label %bb20
bb7:
  %v49 = extractvalue { ptr, i64 } %v20, 0
  %v50 = getelementptr inbounds float, ptr %v49, i64 %v30
  %v51 = load float, ptr %v50, align 4
  %v52 = extractvalue { ptr, i64 } %v27, 0
  %v53 = getelementptr inbounds float, ptr %v52, i64 %v30
  store float %v51, ptr %v53, align 4
  br label %bb15
bb8:
  %v54 = urem i64 %v37, 2
  %v55 = icmp eq i64 %v54, 1
  br i1 %v55, label %bb9, label %bb10
bb9:
  br label %bb15
bb10:
  %v56 = extractvalue { ptr, i64 } %v22, 1
  %v57 = icmp ult i64 %v43, %v56
  br i1 %v57, label %bb11, label %bb21
bb11:
  %v58 = extractvalue { ptr, i64 } %v22, 0
  %v59 = getelementptr inbounds i32, ptr %v58, i64 %v43
  %v60 = load i32, ptr %v59, align 4
  %v61 = uitofp i32 %v60 to float
  %v62 = udiv i64 %v37, 2
  %v63 = extractvalue { ptr, i64 } %v21, 1
  %v64 = icmp ult i64 %v62, %v63
  br i1 %v64, label %bb12, label %bb22
bb12:
  %v65 = extractvalue { ptr, i64 } %v21, 0
  %v66 = getelementptr inbounds float, ptr %v65, i64 %v62
  %v67 = load float, ptr %v66, align 4
  %v68 = fmul contract float %v61, %v67
  %v69 = call float @__nv_sinf(float %v68) #0
  br label %bb16
bb13:
  %v70 = extractvalue { ptr, i64 } %v20, 0
  %v71 = getelementptr inbounds float, ptr %v70, i64 %v88
  %v72 = load float, ptr %v71, align 4
  %v73 = add i64 %v88, 1
  %v74 = icmp ult i64 %v73, %v89
  br i1 %v74, label %bb14, label %bb23
bb14:
  %v75 = extractvalue { ptr, i64 } %v20, 0
  %v76 = getelementptr inbounds float, ptr %v75, i64 %v73
  %v77 = load float, ptr %v76, align 4
  %v78 = fmul contract float %v72, %v87
  %v79 = fmul contract float %v77, %v69
  %v80 = extractvalue { ptr, i64 } %v27, 0
  %v81 = getelementptr inbounds float, ptr %v80, i64 %v88
  %v82 = fsub contract float %v78, %v79
  store float %v82, ptr %v81, align 4
  %v83 = fmul contract float %v72, %v69
  %v84 = fmul contract float %v77, %v87
  %v85 = getelementptr inbounds float, ptr %v80, i64 %v73
  %v86 = fadd contract float %v83, %v84
  store float %v86, ptr %v85, align 4
  br label %bb15
bb15:
  ret void
bb16:
  %v87 = call float @__nv_cosf(float %v68) #0
  br label %bb17
bb17:
  %v88 = add i64 %v38, %v37
  %v89 = extractvalue { ptr, i64 } %v20, 1
  %v90 = icmp ult i64 %v88, %v89
  br i1 %v90, label %bb13, label %bb24
bb18:
  call void @llvm.trap() #0
  unreachable
bb19:
  call void @llvm.trap() #0
  unreachable
bb20:
  call void @llvm.trap() #0
  unreachable
bb21:
  call void @llvm.trap() #0
  unreachable
bb22:
  call void @llvm.trap() #0
  unreachable
bb23:
  call void @llvm.trap() #0
  unreachable
bb24:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q4k_gemv_row_tiled(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, ptr %v6, i64 %v7) #0 {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v7, 1
  br label %bb0
bb0:
  %v14 = phi { ptr, i64 } [ %v9, %entry ]
  %v15 = phi { ptr, i64 } [ %v11, %entry ]
  %v16 = phi i32 [ %v4, %entry ]
  %v17 = phi i32 [ %v5, %entry ]
  %v18 = phi { ptr, i64 } [ %v13, %entry ]
  %v19 = alloca [2 x i8], align 1
  %v20 = alloca [2 x i8], align 1
  %v21 = alloca [8 x i8], align 1
  %v22 = alloca [8 x i8], align 1
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v25 = icmp uge i32 %v24, %v16
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb4, label %bb3
bb3:
  br label %bb64
bb4:
  %v27 = zext i32 %v24 to i64
  %v28 = mul i32 %v17, 144
  %v29 = zext i32 %v28 to i64
  %v30 = mul i64 %v27, %v29
  br label %bb5
bb5:
  %v31 = phi float [ 0.0, %bb4 ], [ %v163, %bb48 ]
  %v32 = phi i32 [ %v23, %bb4 ], [ %v234, %bb48 ]
  %v33 = icmp ult i32 %v32, %v17
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb49, label %bb6
bb6:
  %v35 = zext i32 %v32 to i64
  %v36 = mul i64 %v35, 144
  %v37 = add i64 %v30, %v36
  %v38 = extractvalue { ptr, i64 } %v14, 1
  %v39 = icmp ult i64 %v37, %v38
  br i1 %v39, label %bb7, label %bb65
bb7:
  %v40 = extractvalue { ptr, i64 } %v14, 0
  %v41 = getelementptr inbounds i8, ptr %v40, i64 %v37
  %v42 = load i8, ptr %v41, align 1
  %v43 = add i64 %v37, 1
  %v44 = icmp ult i64 %v43, %v38
  br i1 %v44, label %bb8, label %bb66
bb8:
  %v45 = extractvalue { ptr, i64 } %v14, 0
  %v46 = getelementptr inbounds i8, ptr %v45, i64 %v43
  %v47 = load i8, ptr %v46, align 1
  %v48 = getelementptr inbounds [2 x i8], ptr %v19, i32 0, i64 0
  store i8 %v42, ptr %v48, align 1
  %v49 = getelementptr inbounds [2 x i8], ptr %v19, i32 0, i64 1
  store i8 %v47, ptr %v49, align 1
  %v50 = load [2 x i8], ptr %v19, align 1
  %v51 = alloca [2 x i8], align 2
  store [2 x i8] %v50, ptr %v51, align 2
  %v52 = load i16, ptr %v51, align 2
  %v53 = add i64 %v37, 2
  %v54 = icmp ult i64 %v53, %v38
  br i1 %v54, label %bb9, label %bb67
bb9:
  %v55 = extractvalue { ptr, i64 } %v14, 0
  %v56 = getelementptr inbounds i8, ptr %v55, i64 %v53
  %v57 = load i8, ptr %v56, align 1
  %v58 = add i64 %v37, 3
  %v59 = icmp ult i64 %v58, %v38
  br i1 %v59, label %bb10, label %bb68
bb10:
  %v60 = extractvalue { ptr, i64 } %v14, 0
  %v61 = getelementptr inbounds i8, ptr %v60, i64 %v58
  %v62 = load i8, ptr %v61, align 1
  %v63 = getelementptr inbounds [2 x i8], ptr %v20, i32 0, i64 0
  store i8 %v57, ptr %v63, align 1
  %v64 = getelementptr inbounds [2 x i8], ptr %v20, i32 0, i64 1
  store i8 %v62, ptr %v64, align 1
  %v65 = load [2 x i8], ptr %v20, align 1
  %v66 = alloca [2 x i8], align 2
  store [2 x i8] %v65, ptr %v66, align 2
  %v67 = load i16, ptr %v66, align 2
  %v68 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v52) #0
  br label %bb11
bb11:
  %v69 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v67) #0
  br label %bb12
bb12:
  %v70 = add i64 %v37, 4
  %v71 = icmp ult i64 %v70, %v38
  br i1 %v71, label %bb13, label %bb69
bb13:
  %v72 = extractvalue { ptr, i64 } %v14, 0
  %v73 = getelementptr inbounds i8, ptr %v72, i64 %v70
  %v74 = load i8, ptr %v73, align 1
  %v75 = add i64 %v37, 5
  %v76 = icmp ult i64 %v75, %v38
  br i1 %v76, label %bb14, label %bb70
bb14:
  %v77 = extractvalue { ptr, i64 } %v14, 0
  %v78 = getelementptr inbounds i8, ptr %v77, i64 %v75
  %v79 = load i8, ptr %v78, align 1
  %v80 = add i64 %v37, 6
  %v81 = icmp ult i64 %v80, %v38
  br i1 %v81, label %bb15, label %bb71
bb15:
  %v82 = extractvalue { ptr, i64 } %v14, 0
  %v83 = getelementptr inbounds i8, ptr %v82, i64 %v80
  %v84 = load i8, ptr %v83, align 1
  %v85 = add i64 %v37, 7
  %v86 = icmp ult i64 %v85, %v38
  br i1 %v86, label %bb16, label %bb72
bb16:
  %v87 = extractvalue { ptr, i64 } %v14, 0
  %v88 = getelementptr inbounds i8, ptr %v87, i64 %v85
  %v89 = load i8, ptr %v88, align 1
  %v90 = add i64 %v37, 8
  %v91 = icmp ult i64 %v90, %v38
  br i1 %v91, label %bb17, label %bb73
bb17:
  %v92 = extractvalue { ptr, i64 } %v14, 0
  %v93 = getelementptr inbounds i8, ptr %v92, i64 %v90
  %v94 = load i8, ptr %v93, align 1
  %v95 = add i64 %v37, 9
  %v96 = icmp ult i64 %v95, %v38
  br i1 %v96, label %bb18, label %bb74
bb18:
  %v97 = extractvalue { ptr, i64 } %v14, 0
  %v98 = getelementptr inbounds i8, ptr %v97, i64 %v95
  %v99 = load i8, ptr %v98, align 1
  %v100 = add i64 %v37, 10
  %v101 = icmp ult i64 %v100, %v38
  br i1 %v101, label %bb19, label %bb75
bb19:
  %v102 = extractvalue { ptr, i64 } %v14, 0
  %v103 = getelementptr inbounds i8, ptr %v102, i64 %v100
  %v104 = load i8, ptr %v103, align 1
  %v105 = add i64 %v37, 11
  %v106 = icmp ult i64 %v105, %v38
  br i1 %v106, label %bb20, label %bb76
bb20:
  %v107 = extractvalue { ptr, i64 } %v14, 0
  %v108 = getelementptr inbounds i8, ptr %v107, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v110 = add i64 %v37, 12
  %v111 = icmp ult i64 %v110, %v38
  br i1 %v111, label %bb21, label %bb77
bb21:
  %v112 = extractvalue { ptr, i64 } %v14, 0
  %v113 = getelementptr inbounds i8, ptr %v112, i64 %v110
  %v114 = load i8, ptr %v113, align 1
  %v115 = add i64 %v37, 13
  %v116 = icmp ult i64 %v115, %v38
  br i1 %v116, label %bb22, label %bb78
bb22:
  %v117 = extractvalue { ptr, i64 } %v14, 0
  %v118 = getelementptr inbounds i8, ptr %v117, i64 %v115
  %v119 = load i8, ptr %v118, align 1
  %v120 = add i64 %v37, 14
  %v121 = icmp ult i64 %v120, %v38
  br i1 %v121, label %bb23, label %bb79
bb23:
  %v122 = extractvalue { ptr, i64 } %v14, 0
  %v123 = getelementptr inbounds i8, ptr %v122, i64 %v120
  %v124 = load i8, ptr %v123, align 1
  %v125 = add i64 %v37, 15
  %v126 = icmp ult i64 %v125, %v38
  br i1 %v126, label %bb24, label %bb80
bb24:
  %v127 = extractvalue { ptr, i64 } %v14, 0
  %v128 = getelementptr inbounds i8, ptr %v127, i64 %v125
  %v129 = load i8, ptr %v128, align 1
  %v130 = call { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v74, i8 %v79, i8 %v84, i8 %v89, i8 %v94, i8 %v99, i8 %v104, i8 %v109, i8 %v114, i8 %v119, i8 %v124, i8 %v129) #0
  br label %bb25
bb25:
  %v131 = extractvalue { [8 x i8], [8 x i8] } %v130, 0
  store [8 x i8] %v131, ptr %v21, align 1
  %v132 = extractvalue { [8 x i8], [8 x i8] } %v130, 1
  store [8 x i8] %v132, ptr %v22, align 1
  %v133 = add i64 %v37, 16
  %v134 = zext i32 %v32 to i64
  %v135 = mul i64 %v134, 256
  br label %bb26
bb26:
  %v136 = phi float [ 0.0, %bb25 ], [ %v159, %bb32 ]
  %v137 = phi i64 [ 0, %bb25 ], [ %v160, %bb32 ]
  %v138 = icmp ult i64 %v137, 8
  %v139 = xor i1 %v138, 1
  br i1 %v139, label %bb33, label %bb27
bb27:
  br label %bb28
bb28:
  %v140 = phi float [ 0.0, %bb27 ], [ %v152, %bb30 ]
  %v141 = phi i64 [ 0, %bb27 ], [ %v153, %bb30 ]
  %v142 = icmp ult i64 %v141, 32
  %v143 = xor i1 %v142, 1
  br i1 %v143, label %bb31, label %bb29
bb29:
  %v144 = mul i64 %v137, 32
  %v145 = add i64 %v135, %v144
  %v146 = add i64 %v145, %v141
  %v147 = extractvalue { ptr, i64 } %v15, 1
  %v148 = icmp ult i64 %v146, %v147
  br i1 %v148, label %bb30, label %bb81
bb30:
  %v149 = extractvalue { ptr, i64 } %v15, 0
  %v150 = getelementptr inbounds float, ptr %v149, i64 %v146
  %v151 = load float, ptr %v150, align 4
  %v152 = fadd contract float %v140, %v151
  %v153 = add i64 %v141, 1
  br label %bb28
bb31:
  %v154 = icmp ult i64 %v137, 8
  br i1 %v154, label %bb32, label %bb82
bb32:
  %v155 = getelementptr inbounds [8 x i8], ptr %v22, i32 0, i64 %v137
  %v156 = load i8, ptr %v155, align 1
  %v157 = uitofp i8 %v156 to float
  %v158 = fmul contract float %v157, %v140
  %v159 = fadd contract float %v136, %v158
  %v160 = add i64 %v137, 1
  br label %bb26
bb33:
  %v161 = fmul contract float %v69, %v136
  %v162 = fsub contract float %v31, %v161
  br label %bb34
bb34:
  %v163 = phi float [ %v162, %bb33 ], [ %v231, %bb47 ]
  %v164 = phi i64 [ 0, %bb33 ], [ %v205, %bb47 ]
  %v165 = phi i64 [ 0, %bb33 ], [ %v232, %bb47 ]
  %v166 = phi i64 [ 0, %bb33 ], [ %v233, %bb47 ]
  %v167 = icmp ult i64 %v166, 4
  %v168 = xor i1 %v167, 1
  br i1 %v168, label %bb48, label %bb35
bb35:
  %v169 = mul i64 %v166, 32
  %v170 = add i64 %v133, %v169
  %v171 = icmp ult i64 %v164, 8
  br i1 %v171, label %bb36, label %bb83
bb36:
  %v172 = getelementptr inbounds [8 x i8], ptr %v21, i32 0, i64 %v164
  %v173 = load i8, ptr %v172, align 1
  %v174 = uitofp i8 %v173 to float
  %v175 = add i64 %v164, 1
  br label %bb37
bb37:
  %v176 = phi float [ 0.0, %bb36 ], [ %v195, %bb40 ]
  %v177 = phi i64 [ 0, %bb36 ], [ %v196, %bb40 ]
  %v178 = icmp ult i64 %v177, 32
  %v179 = xor i1 %v178, 1
  br i1 %v179, label %bb41, label %bb38
bb38:
  %v180 = add i64 %v170, %v177
  %v181 = icmp ult i64 %v180, %v38
  br i1 %v181, label %bb39, label %bb84
bb39:
  %v182 = extractvalue { ptr, i64 } %v14, 0
  %v183 = getelementptr inbounds i8, ptr %v182, i64 %v180
  %v184 = load i8, ptr %v183, align 1
  %v185 = and i8 %v184, 15
  %v186 = uitofp i8 %v185 to float
  %v187 = add i64 %v135, %v165
  %v188 = add i64 %v187, %v177
  %v189 = extractvalue { ptr, i64 } %v15, 1
  %v190 = icmp ult i64 %v188, %v189
  br i1 %v190, label %bb40, label %bb85
bb40:
  %v191 = extractvalue { ptr, i64 } %v15, 0
  %v192 = getelementptr inbounds float, ptr %v191, i64 %v188
  %v193 = load float, ptr %v192, align 4
  %v194 = fmul contract float %v186, %v193
  %v195 = fadd contract float %v176, %v194
  %v196 = add i64 %v177, 1
  br label %bb37
bb41:
  %v197 = fmul contract float %v68, %v174
  %v198 = fmul contract float %v197, %v176
  %v199 = fadd contract float %v163, %v198
  %v200 = add i64 %v165, 32
  %v201 = icmp ult i64 %v175, 8
  br i1 %v201, label %bb42, label %bb86
bb42:
  %v202 = getelementptr inbounds [8 x i8], ptr %v21, i32 0, i64 %v175
  %v203 = load i8, ptr %v202, align 1
  %v204 = uitofp i8 %v203 to float
  %v205 = add i64 %v175, 1
  br label %bb43
bb43:
  %v206 = phi float [ 0.0, %bb42 ], [ %v227, %bb46 ]
  %v207 = phi i64 [ 0, %bb42 ], [ %v228, %bb46 ]
  %v208 = icmp ult i64 %v207, 32
  %v209 = xor i1 %v208, 1
  br i1 %v209, label %bb47, label %bb44
bb44:
  %v210 = add i64 %v170, %v207
  %v211 = icmp ult i64 %v210, %v38
  br i1 %v211, label %bb45, label %bb87
bb45:
  %v212 = extractvalue { ptr, i64 } %v14, 0
  %v213 = getelementptr inbounds i8, ptr %v212, i64 %v210
  %v214 = load i8, ptr %v213, align 1
  %v215 = trunc i32 4 to i8
  %v216 = and i8 %v215, 7
  %v217 = lshr i8 %v214, %v216
  %v218 = uitofp i8 %v217 to float
  %v219 = add i64 %v135, %v200
  %v220 = add i64 %v219, %v207
  %v221 = extractvalue { ptr, i64 } %v15, 1
  %v222 = icmp ult i64 %v220, %v221
  br i1 %v222, label %bb46, label %bb88
bb46:
  %v223 = extractvalue { ptr, i64 } %v15, 0
  %v224 = getelementptr inbounds float, ptr %v223, i64 %v220
  %v225 = load float, ptr %v224, align 4
  %v226 = fmul contract float %v218, %v225
  %v227 = fadd contract float %v206, %v226
  %v228 = add i64 %v207, 1
  br label %bb43
bb47:
  %v229 = fmul contract float %v68, %v204
  %v230 = fmul contract float %v229, %v206
  %v231 = fadd contract float %v199, %v230
  %v232 = add i64 %v200, 32
  %v233 = add i64 %v166, 1
  br label %bb34
bb48:
  %v234 = add i32 %v32, 32
  br label %bb5
bb49:
  %v235 = zext i32 %v23 to i64
  %v236 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_13, i64 %v235
  br label %bb50
bb50:
  store float %v31, ptr addrspace(3) %v236, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb51
bb51:
  br label %bb52
bb52:
  %v238 = phi i32 [ 16, %bb51 ], [ %v252, %bb59 ]
  %v239 = icmp ugt i32 %v238, 0
  %v240 = xor i1 %v239, 1
  br i1 %v240, label %bb60, label %bb53
bb53:
  %v241 = icmp ult i32 %v23, %v238
  %v242 = xor i1 %v241, 1
  br i1 %v242, label %bb57, label %bb54
bb54:
  %v243 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v244 = add i32 %v23, %v238
  %v245 = zext i32 %v244 to i64
  %v246 = getelementptr inbounds float, ptr addrspace(3) %v243, i64 %v245
  br label %bb55
bb55:
  %v247 = load float, ptr addrspace(3) %v246, align 4
  %v248 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_13, i64 %v235
  br label %bb56
bb56:
  %v249 = load float, ptr addrspace(3) %v248, align 4
  %v250 = fadd contract float %v249, %v247
  store float %v250, ptr addrspace(3) %v248, align 4
  br label %bb58
bb57:
  br label %bb58
bb58:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb59
bb59:
  %v252 = udiv i32 %v238, 2
  br label %bb52
bb60:
  %v253 = icmp eq i32 %v23, 0
  br i1 %v253, label %bb61, label %bb63
bb61:
  %v254 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v255 = getelementptr inbounds float, ptr addrspace(3) %v254, i64 0
  br label %bb62
bb62:
  %v256 = load float, ptr addrspace(3) %v255, align 4
  %v257 = extractvalue { ptr, i64 } %v18, 0
  %v258 = getelementptr inbounds float, ptr %v257, i64 %v27
  store float %v256, ptr %v258, align 4
  br label %bb63
bb63:
  br label %bb64
bb64:
  ret void
bb65:
  call void @llvm.trap() #0
  unreachable
bb66:
  call void @llvm.trap() #0
  unreachable
bb67:
  call void @llvm.trap() #0
  unreachable
bb68:
  call void @llvm.trap() #0
  unreachable
bb69:
  call void @llvm.trap() #0
  unreachable
bb70:
  call void @llvm.trap() #0
  unreachable
bb71:
  call void @llvm.trap() #0
  unreachable
bb72:
  call void @llvm.trap() #0
  unreachable
bb73:
  call void @llvm.trap() #0
  unreachable
bb74:
  call void @llvm.trap() #0
  unreachable
bb75:
  call void @llvm.trap() #0
  unreachable
bb76:
  call void @llvm.trap() #0
  unreachable
bb77:
  call void @llvm.trap() #0
  unreachable
bb78:
  call void @llvm.trap() #0
  unreachable
bb79:
  call void @llvm.trap() #0
  unreachable
bb80:
  call void @llvm.trap() #0
  unreachable
bb81:
  call void @llvm.trap() #0
  unreachable
bb82:
  call void @llvm.trap() #0
  unreachable
bb83:
  call void @llvm.trap() #0
  unreachable
bb84:
  call void @llvm.trap() #0
  unreachable
bb85:
  call void @llvm.trap() #0
  unreachable
bb86:
  call void @llvm.trap() #0
  unreachable
bb87:
  call void @llvm.trap() #0
  unreachable
bb88:
  call void @llvm.trap() #0
  unreachable
}

declare float @__nv_sqrtf(float)

define ptx_kernel void @rmsnorm_group(ptr %v0, i64 %v1, ptr %v2, i64 %v3, float %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi float [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
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
  %v32 = extractvalue { ptr, i64 } %v15, 1
  %v33 = icmp ult i64 %v31, %v32
  br i1 %v33, label %bb5, label %bb25
bb5:
  %v34 = extractvalue { ptr, i64 } %v15, 0
  %v35 = getelementptr inbounds float, ptr %v34, i64 %v31
  %v36 = load float, ptr %v35, align 4
  %v37 = fmul contract float %v36, %v36
  %v38 = fadd contract float %v27, %v37
  %v39 = add i64 %v28, 256
  br label %bb3
bb6:
  %v40 = zext i32 %v21 to i64
  %v41 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_14, i64 %v40
  br label %bb7
bb7:
  store float %v27, ptr addrspace(3) %v41, align 4
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb8
bb8:
  br label %bb9
bb9:
  %v43 = phi i32 [ 128, %bb8 ], [ %v57, %bb16 ]
  %v44 = icmp ugt i32 %v43, 0
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb17, label %bb10
bb10:
  %v46 = icmp ult i32 %v21, %v43
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb14, label %bb11
bb11:
  %v48 = bitcast ptr addrspace(3) @__shared_mem_14 to ptr addrspace(3)
  %v49 = add i32 %v21, %v43
  %v50 = zext i32 %v49 to i64
  %v51 = getelementptr inbounds float, ptr addrspace(3) %v48, i64 %v50
  br label %bb12
bb12:
  %v52 = load float, ptr addrspace(3) %v51, align 4
  %v53 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_14, i64 %v40
  br label %bb13
bb13:
  %v54 = load float, ptr addrspace(3) %v53, align 4
  %v55 = fadd contract float %v54, %v52
  store float %v55, ptr addrspace(3) %v53, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb16
bb16:
  %v57 = udiv i32 %v43, 2
  br label %bb9
bb17:
  %v58 = bitcast ptr addrspace(3) @__shared_mem_14 to ptr addrspace(3)
  %v59 = getelementptr inbounds float, ptr addrspace(3) %v58, i64 0
  br label %bb18
bb18:
  %v60 = load float, ptr addrspace(3) %v59, align 4
  %v61 = uitofp i32 %v18 to float
  %v62 = fdiv contract float %v60, %v61
  %v63 = fadd contract float %v62, %v17
  %v64 = call float @__nv_sqrtf(float %v63) #0
  br label %bb24
bb19:
  %v65 = phi i64 [ %v84, %bb22 ], [ %v40, %bb24 ]
  %v66 = icmp ult i64 %v65, %v25
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb23, label %bb20
bb20:
  %v68 = add i64 %v24, %v65
  %v69 = extractvalue { ptr, i64 } %v15, 1
  %v70 = icmp ult i64 %v68, %v69
  br i1 %v70, label %bb21, label %bb26
bb21:
  %v71 = extractvalue { ptr, i64 } %v15, 0
  %v72 = getelementptr inbounds float, ptr %v71, i64 %v68
  %v73 = load float, ptr %v72, align 4
  %v74 = fmul contract float %v73, %v85
  %v75 = extractvalue { ptr, i64 } %v16, 1
  %v76 = icmp ult i64 %v65, %v75
  br i1 %v76, label %bb22, label %bb27
bb22:
  %v77 = extractvalue { ptr, i64 } %v16, 0
  %v78 = getelementptr inbounds float, ptr %v77, i64 %v65
  %v79 = load float, ptr %v78, align 4
  %v80 = add i64 %v24, %v65
  %v81 = extractvalue { ptr, i64 } %v20, 0
  %v82 = getelementptr inbounds float, ptr %v81, i64 %v80
  %v83 = fmul contract float %v74, %v79
  store float %v83, ptr %v82, align 4
  %v84 = add i64 %v65, 256
  br label %bb19
bb23:
  ret void
bb24:
  %v85 = fdiv contract float 1.0, %v64
  br label %bb19
bb25:
  call void @llvm.trap() #0
  unreachable
bb26:
  call void @llvm.trap() #0
  unreachable
bb27:
  call void @llvm.trap() #0
  unreachable
}

declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #0

define ptx_kernel void @attention_fa3_decode_paged(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, i32 %v16, ptr %v17, i64 %v18) #0 {
entry:
  %v19 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v1, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v3, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v5, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v17, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v18, 1
  br label %bb0
bb0:
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v6, %entry ]
  %v31 = phi i32 [ %v7, %entry ]
  %v32 = phi i32 [ %v8, %entry ]
  %v33 = phi i32 [ %v9, %entry ]
  %v34 = phi float [ %v10, %entry ]
  %v35 = phi i32 [ %v11, %entry ]
  %v36 = phi i32 [ %v12, %entry ]
  %v37 = phi i32 [ %v13, %entry ]
  %v38 = phi i32 [ %v14, %entry ]
  %v39 = phi i32 [ %v15, %entry ]
  %v40 = phi i32 [ %v16, %entry ]
  %v41 = phi { ptr, i64 } [ %v26, %entry ]
  %v42 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v44 = urem i32 %v43, 32
  %v45 = zext i32 %v44 to i64
  %v46 = udiv i32 %v43, 32
  %v47 = icmp uge i32 %v42, %v30
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb3, label %bb5
bb3:
  %v49 = icmp eq i32 %v32, 0
  br i1 %v49, label %bb5, label %bb4
bb4:
  %v50 = icmp ugt i32 %v32, 128
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb6, label %bb5
bb5:
  br label %bb110
bb6:
  %v52 = zext i32 %v32 to i64
  %v53 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v31, i32 1) #0
  br label %bb7
bb7:
  %v54 = icmp eq i32 %v53, 0
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb8, label %bb115
bb8:
  %v56 = udiv i32 %v30, %v53
  %v57 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v56, i32 1) #0
  br label %bb9
bb9:
  %v58 = icmp eq i32 %v57, 0
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb10, label %bb116
bb10:
  %v60 = udiv i32 %v42, %v57
  %v61 = zext i32 %v60 to i64
  %v62 = zext i32 %v37 to i64
  %v63 = zext i32 %v38 to i64
  %v64 = zext i32 %v39 to i64
  %v65 = mul i64 %v64, 2
  %v66 = mul i64 %v62, %v65
  %v67 = zext i32 %v42 to i64
  %v68 = mul i64 %v67, %v52
  %v69 = icmp ult i32 %v40, 2
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb12, label %bb11
bb11:
  br label %bb13
bb12:
  %v71 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCs5VsnSnoaHeT_12cuda_kernels(i32 %v40, i32 2) #0
  br label %bb13
bb13:
  %v72 = phi i32 [ 2, %bb11 ], [ %v71, %bb12 ]
  %v73 = icmp eq i32 %v46, 0
  %v74 = icmp uge i32 %v46, 1
  %v75 = icmp ult i32 %v43, 2
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb16, label %bb14
bb14:
  %v77 = zext i32 %v43 to i64
  %v78 = getelementptr inbounds i32, ptr addrspace(3) @__shared_mem_15, i64 %v77
  br label %bb15
bb15:
  store i32 0, ptr addrspace(3) %v78, align 4
  br label %bb16
bb16:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb17
bb17:
  %v80 = add i32 %v33, 16
  %v81 = sub i32 %v80, 1
  %v82 = udiv i32 %v81, 16
  br label %bb18
bb18:
  %v83 = phi float [ 0.0, %bb17 ], [ %v323, %bb92 ]
  %v84 = phi float [ 0.0, %bb17 ], [ %v324, %bb92 ]
  %v85 = phi float [ 0.0, %bb17 ], [ %v325, %bb92 ]
  %v86 = phi float [ 0.0, %bb17 ], [ %v326, %bb92 ]
  %v87 = phi float [ 0.0, %bb17 ], [ %v327, %bb92 ]
  %v88 = phi float [ 0.0, %bb17 ], [ %v328, %bb92 ]
  %v89 = phi i1 [ 0, %bb17 ], [ %v329, %bb92 ]
  %v90 = phi i32 [ 0, %bb17 ], [ %v331, %bb92 ]
  %v91 = add i32 %v82, %v72
  %v92 = icmp ult i32 %v90, %v91
  %v93 = xor i1 %v92, 1
  br i1 %v93, label %bb93, label %bb19
bb19:
  %v94 = icmp uge i32 %v90, %v72
  %v95 = xor i1 %v94, 1
  br i1 %v95, label %bb21, label %bb20
bb20:
  %v96 = sub i32 %v90, %v72
  br label %bb22
bb21:
  br label %bb22
bb22:
  %v97 = phi i32 [ %v96, %bb20 ], [ 4294967295, %bb21 ]
  %v98 = xor i1 %v73, 1
  br i1 %v98, label %bb47, label %bb23
bb23:
  %v99 = icmp ult i32 %v90, %v82
  %v100 = xor i1 %v99, 1
  br i1 %v100, label %bb47, label %bb24
bb24:
  %v101 = icmp eq i32 %v72, 0
  %v102 = xor i1 %v101, 1
  br i1 %v102, label %bb25, label %bb117
bb25:
  %v103 = urem i32 %v90, %v72
  %v104 = zext i32 %v103 to i64
  %v105 = mul i32 %v90, 16
  %v106 = add i32 %v105, 16
  %v107 = icmp ugt i32 %v106, %v33
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb27, label %bb26
bb26:
  br label %bb28
bb27:
  br label %bb28
bb28:
  %v109 = phi i32 [ %v33, %bb26 ], [ %v106, %bb27 ]
  %v110 = sub i32 %v109, %v105
  %v111 = zext i32 %v110 to i64
  %v112 = mul i64 %v104, 16
  %v113 = mul i64 %v112, 128
  br label %bb29
bb29:
  %v114 = phi i64 [ %v45, %bb28 ], [ %v188, %bb36 ]
  %v115 = mul i64 %v111, %v52
  %v116 = icmp ult i64 %v114, %v115
  %v117 = xor i1 %v116, 1
  br i1 %v117, label %bb37, label %bb30
bb30:
  %v118 = icmp eq i64 %v52, 0
  %v119 = xor i1 %v118, 1
  br i1 %v119, label %bb31, label %bb118
bb31:
  %v120 = udiv i64 %v114, %v52
  %v121 = urem i64 %v114, %v52
  %v122 = zext i32 %v105 to i64
  %v123 = add i64 %v122, %v120
  %v124 = icmp eq i64 %v62, 0
  %v125 = xor i1 %v124, 1
  br i1 %v125, label %bb32, label %bb119
bb32:
  %v126 = udiv i64 %v123, %v62
  %v127 = urem i64 %v123, %v62
  %v128 = zext i32 %v35 to i64
  %v129 = zext i32 %v36 to i64
  %v130 = mul i64 %v128, %v129
  %v131 = add i64 %v130, %v126
  %v132 = extractvalue { ptr, i64 } %v29, 1
  %v133 = icmp ult i64 %v131, %v132
  %v134 = extractvalue { ptr, i64 } %v29, 0
  %v135 = getelementptr inbounds i32, ptr %v134, i64 %v131
  %v136 = load i32, ptr %v135, align 4
  %v137 = zext i32 %v136 to i64
  %v138 = mul i64 %v137, %v63
  %v139 = mul i64 %v127, %v65
  %v140 = add i64 %v138, %v139
  %v141 = mul i64 %v61, %v52
  %v142 = mul i64 %v141, 2
  %v143 = add i64 %v140, %v142
  %v144 = add i64 %v138, %v66
  %v145 = add i64 %v144, %v139
  %v146 = add i64 %v145, %v142
  %v147 = mul i64 %v121, 2
  %v148 = add i64 %v143, %v147
  %v149 = extractvalue { ptr, i64 } %v28, 1
  %v150 = icmp ult i64 %v148, %v149
  %v151 = extractvalue { ptr, i64 } %v28, 0
  %v152 = getelementptr inbounds i8, ptr %v151, i64 %v148
  %v153 = load i8, ptr %v152, align 1
  %v154 = zext i8 %v153 to i16
  %v155 = add i64 %v148, 1
  %v156 = icmp ult i64 %v155, %v149
  %v157 = extractvalue { ptr, i64 } %v28, 0
  %v158 = getelementptr inbounds i8, ptr %v157, i64 %v155
  %v159 = load i8, ptr %v158, align 1
  %v160 = zext i8 %v159 to i16
  %v161 = trunc i32 8 to i16
  %v162 = and i16 %v161, 15
  %v163 = shl i16 %v160, %v162
  %v164 = or i16 %v154, %v163
  %v165 = add i64 %v146, %v147
  %v166 = icmp ult i64 %v165, %v149
  %v167 = extractvalue { ptr, i64 } %v28, 0
  %v168 = getelementptr inbounds i8, ptr %v167, i64 %v165
  %v169 = load i8, ptr %v168, align 1
  %v170 = zext i8 %v169 to i16
  %v171 = add i64 %v165, 1
  %v172 = icmp ult i64 %v171, %v149
  %v173 = extractvalue { ptr, i64 } %v28, 0
  %v174 = getelementptr inbounds i8, ptr %v173, i64 %v171
  %v175 = load i8, ptr %v174, align 1
  %v176 = zext i8 %v175 to i16
  %v177 = trunc i32 8 to i16
  %v178 = and i16 %v177, 15
  %v179 = shl i16 %v176, %v178
  %v180 = or i16 %v170, %v179
  %v181 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v164) #0
  br label %bb33
bb33:
  %v182 = mul i64 %v120, 128
  %v183 = add i64 %v113, %v182
  %v184 = add i64 %v183, %v121
  %v185 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_16, i64 %v184
  br label %bb34
bb34:
  store float %v181, ptr addrspace(3) %v185, align 4
  %v186 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v180) #0
  br label %bb35
bb35:
  %v187 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_17, i64 %v184
  br label %bb36
bb36:
  store float %v186, ptr addrspace(3) %v187, align 4
  %v188 = add i64 %v114, 32
  br label %bb29
bb37:
  %v189 = add i64 %v115, %v45
  br label %bb38
bb38:
  %v190 = phi i64 [ %v189, %bb37 ], [ %v203, %bb42 ]
  %v191 = mul i64 16, %v52
  %v192 = icmp ult i64 %v190, %v191
  %v193 = xor i1 %v192, 1
  br i1 %v193, label %bb43, label %bb39
bb39:
  %v194 = icmp eq i64 %v52, 0
  %v195 = xor i1 %v194, 1
  br i1 %v195, label %bb40, label %bb120
bb40:
  %v196 = udiv i64 %v190, %v52
  %v197 = urem i64 %v190, %v52
  %v198 = mul i64 %v196, 128
  %v199 = add i64 %v113, %v198
  %v200 = add i64 %v199, %v197
  %v201 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_16, i64 %v200
  br label %bb41
bb41:
  store float 0.0, ptr addrspace(3) %v201, align 4
  %v202 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_17, i64 %v200
  br label %bb42
bb42:
  store float 0.0, ptr addrspace(3) %v202, align 4
  %v203 = add i64 %v190, 32
  br label %bb38
bb43:
  %v204 = icmp eq i64 %v45, 0
  br i1 %v204, label %bb44, label %bb46
bb44:
  %v205 = getelementptr inbounds i32, ptr addrspace(3) @__shared_mem_15, i64 %v104
  br label %bb45
bb45:
  store i32 1, ptr addrspace(3) %v205, align 4
  br label %bb46
bb46:
  br label %bb47
bb47:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb48
bb48:
  %v207 = xor i1 %v74, 1
  br i1 %v207, label %bb91, label %bb49
bb49:
  %v208 = icmp ult i32 %v97, %v82
  %v209 = xor i1 %v208, 1
  br i1 %v209, label %bb90, label %bb50
bb50:
  %v210 = icmp eq i32 %v46, 1
  br i1 %v210, label %bb51, label %bb91
bb51:
  %v211 = icmp eq i32 %v72, 0
  %v212 = xor i1 %v211, 1
  br i1 %v212, label %bb52, label %bb121
bb52:
  %v213 = urem i32 %v97, %v72
  %v214 = zext i32 %v213 to i64
  %v215 = mul i32 %v97, 16
  %v216 = add i32 %v215, 16
  %v217 = icmp ugt i32 %v216, %v33
  %v218 = xor i1 %v217, 1
  br i1 %v218, label %bb54, label %bb53
bb53:
  br label %bb55
bb54:
  br label %bb55
bb55:
  %v219 = phi i32 [ %v33, %bb53 ], [ %v216, %bb54 ]
  %v220 = sub i32 %v219, %v215
  %v221 = zext i32 %v220 to i64
  %v222 = mul i64 %v214, 16
  %v223 = mul i64 %v222, 128
  br label %bb56
bb56:
  %v224 = phi float [ %v83, %bb55 ], [ %v277, %bb85 ]
  %v225 = phi float [ %v84, %bb55 ], [ %v291, %bb85 ]
  %v226 = phi float [ %v85, %bb55 ], [ %v305, %bb85 ]
  %v227 = phi float [ %v86, %bb55 ], [ %v319, %bb85 ]
  %v228 = phi float [ %v87, %bb55 ], [ %v264, %bb85 ]
  %v229 = phi float [ %v88, %bb55 ], [ %v372, %bb85 ]
  %v230 = phi i1 [ %v89, %bb55 ], [ 1, %bb85 ]
  %v231 = phi i64 [ 0, %bb55 ], [ %v320, %bb85 ]
  %v232 = icmp ult i64 %v231, %v221
  %v233 = xor i1 %v232, 1
  br i1 %v233, label %bb86, label %bb57
bb57:
  br label %bb58
bb58:
  %v234 = phi float [ 0.0, %bb57 ], [ %v251, %bb60 ]
  %v235 = phi i64 [ %v45, %bb57 ], [ %v252, %bb60 ]
  %v236 = icmp ult i64 %v235, %v52
  %v237 = xor i1 %v236, 1
  br i1 %v237, label %bb61, label %bb59
bb59:
  %v238 = bitcast ptr addrspace(3) @__shared_mem_16 to ptr addrspace(3)
  %v239 = mul i64 %v231, 128
  %v240 = add i64 %v223, %v239
  %v241 = add i64 %v240, %v235
  %v242 = getelementptr inbounds float, ptr addrspace(3) %v238, i64 %v241
  br label %bb60
bb60:
  %v243 = load float, ptr addrspace(3) %v242, align 4
  %v244 = add i64 %v68, %v235
  %v245 = extractvalue { ptr, i64 } %v27, 1
  %v246 = icmp ult i64 %v244, %v245
  %v247 = extractvalue { ptr, i64 } %v27, 0
  %v248 = getelementptr inbounds float, ptr %v247, i64 %v244
  %v249 = load float, ptr %v248, align 4
  %v250 = fmul contract float %v249, %v243
  %v251 = fadd contract float %v234, %v250
  %v252 = add i64 %v235, 32
  br label %bb58
bb61:
  br label %bb62
bb62:
  %v253 = phi float [ %v234, %bb61 ], [ %v367, %bb111 ]
  %v254 = phi i32 [ 16, %bb61 ], [ %v368, %bb111 ]
  %v255 = icmp eq i32 %v254, 0
  br i1 %v255, label %bb64, label %bb63
bb63:
  %v256 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v253, i32 %v254, i32 31) #0
  br label %bb111
bb64:
  %v257 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 4294967295, float %v253, i32 0, i32 31) #0
  br label %bb112
bb65:
  br label %bb70
bb66:
  %v258 = fcmp ogt float %v369, %v228
  %v259 = xor i1 %v258, 1
  br i1 %v259, label %bb68, label %bb67
bb67:
  %v260 = fsub contract float %v228, %v369
  %v261 = call float @__nv_expf(float %v260) #0
  br label %bb113
bb68:
  br label %bb69
bb69:
  %v262 = phi float [ %v228, %bb68 ], [ %v369, %bb113 ]
  %v263 = phi float [ 1.0, %bb68 ], [ %v261, %bb113 ]
  br label %bb70
bb70:
  %v264 = phi float [ %v369, %bb65 ], [ %v262, %bb69 ]
  %v265 = phi float [ 0.0, %bb65 ], [ %v263, %bb69 ]
  %v266 = fsub contract float %v369, %v264
  %v267 = call float @__nv_expf(float %v266) #0
  br label %bb114
bb71:
  %v268 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v269 = mul i64 %v231, 128
  %v270 = add i64 %v223, %v269
  %v271 = add i64 %v270, %v45
  %v272 = getelementptr inbounds float, ptr addrspace(3) %v268, i64 %v271
  br label %bb72
bb72:
  %v273 = load float, ptr addrspace(3) %v272, align 4
  %v274 = fmul contract float %v224, %v265
  %v275 = fmul contract float %v267, %v273
  %v276 = fadd contract float %v274, %v275
  br label %bb73
bb73:
  %v277 = phi float [ %v276, %bb72 ], [ %v224, %bb114 ]
  %v278 = add i64 %v45, 32
  %v279 = icmp ult i64 %v278, %v52
  %v280 = xor i1 %v279, 1
  br i1 %v280, label %bb76, label %bb74
bb74:
  %v281 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v282 = mul i64 %v231, 128
  %v283 = add i64 %v223, %v282
  %v284 = add i64 %v283, %v45
  %v285 = add i64 %v284, 32
  %v286 = getelementptr inbounds float, ptr addrspace(3) %v281, i64 %v285
  br label %bb75
bb75:
  %v287 = load float, ptr addrspace(3) %v286, align 4
  %v288 = fmul contract float %v225, %v265
  %v289 = fmul contract float %v267, %v287
  %v290 = fadd contract float %v288, %v289
  br label %bb77
bb76:
  br label %bb77
bb77:
  %v291 = phi float [ %v290, %bb75 ], [ %v225, %bb76 ]
  %v292 = add i64 %v45, 64
  %v293 = icmp ult i64 %v292, %v52
  %v294 = xor i1 %v293, 1
  br i1 %v294, label %bb80, label %bb78
bb78:
  %v295 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v296 = mul i64 %v231, 128
  %v297 = add i64 %v223, %v296
  %v298 = add i64 %v297, %v45
  %v299 = add i64 %v298, 64
  %v300 = getelementptr inbounds float, ptr addrspace(3) %v295, i64 %v299
  br label %bb79
bb79:
  %v301 = load float, ptr addrspace(3) %v300, align 4
  %v302 = fmul contract float %v226, %v265
  %v303 = fmul contract float %v267, %v301
  %v304 = fadd contract float %v302, %v303
  br label %bb81
bb80:
  br label %bb81
bb81:
  %v305 = phi float [ %v304, %bb79 ], [ %v226, %bb80 ]
  %v306 = add i64 %v45, 96
  %v307 = icmp ult i64 %v306, %v52
  %v308 = xor i1 %v307, 1
  br i1 %v308, label %bb84, label %bb82
bb82:
  %v309 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v310 = mul i64 %v231, 128
  %v311 = add i64 %v223, %v310
  %v312 = add i64 %v311, %v45
  %v313 = add i64 %v312, 96
  %v314 = getelementptr inbounds float, ptr addrspace(3) %v309, i64 %v313
  br label %bb83
bb83:
  %v315 = load float, ptr addrspace(3) %v314, align 4
  %v316 = fmul contract float %v227, %v265
  %v317 = fmul contract float %v267, %v315
  %v318 = fadd contract float %v316, %v317
  br label %bb85
bb84:
  br label %bb85
bb85:
  %v319 = phi float [ %v318, %bb83 ], [ %v227, %bb84 ]
  %v320 = add i64 %v231, 1
  br label %bb56
bb86:
  %v321 = icmp eq i64 %v45, 0
  br i1 %v321, label %bb87, label %bb89
bb87:
  %v322 = getelementptr inbounds i32, ptr addrspace(3) @__shared_mem_15, i64 %v214
  br label %bb88
bb88:
  store i32 0, ptr addrspace(3) %v322, align 4
  br label %bb89
bb89:
  br label %bb91
bb90:
  br label %bb91
bb91:
  %v323 = phi float [ %v83, %bb48 ], [ %v83, %bb50 ], [ %v224, %bb89 ], [ %v83, %bb90 ]
  %v324 = phi float [ %v84, %bb48 ], [ %v84, %bb50 ], [ %v225, %bb89 ], [ %v84, %bb90 ]
  %v325 = phi float [ %v85, %bb48 ], [ %v85, %bb50 ], [ %v226, %bb89 ], [ %v85, %bb90 ]
  %v326 = phi float [ %v86, %bb48 ], [ %v86, %bb50 ], [ %v227, %bb89 ], [ %v86, %bb90 ]
  %v327 = phi float [ %v87, %bb48 ], [ %v87, %bb50 ], [ %v228, %bb89 ], [ %v87, %bb90 ]
  %v328 = phi float [ %v88, %bb48 ], [ %v88, %bb50 ], [ %v229, %bb89 ], [ %v88, %bb90 ]
  %v329 = phi i1 [ %v89, %bb48 ], [ %v89, %bb50 ], [ %v230, %bb89 ], [ %v89, %bb90 ]
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb92
bb92:
  %v331 = add i32 %v90, 1
  br label %bb18
bb93:
  %v332 = icmp eq i32 %v46, 1
  br i1 %v332, label %bb94, label %bb109
bb94:
  %v333 = xor i1 %v89, 1
  br i1 %v333, label %bb109, label %bb95
bb95:
  %v334 = fcmp ogt float %v88, 0.0
  %v335 = xor i1 %v334, 1
  br i1 %v335, label %bb108, label %bb96
bb96:
  %v336 = fdiv contract float 1.0, %v88
  %v337 = icmp ult i64 %v45, %v52
  %v338 = xor i1 %v337, 1
  br i1 %v338, label %bb98, label %bb97
bb97:
  %v339 = add i64 %v68, %v45
  %v340 = extractvalue { ptr, i64 } %v41, 0
  %v341 = getelementptr inbounds float, ptr %v340, i64 %v339
  %v342 = fmul contract float %v83, %v336
  store float %v342, ptr %v341, align 4
  br label %bb98
bb98:
  %v343 = add i64 %v45, 32
  %v344 = icmp ult i64 %v343, %v52
  %v345 = xor i1 %v344, 1
  br i1 %v345, label %bb100, label %bb99
bb99:
  %v346 = add i64 %v68, %v45
  %v347 = add i64 %v346, 32
  %v348 = extractvalue { ptr, i64 } %v41, 0
  %v349 = getelementptr inbounds float, ptr %v348, i64 %v347
  %v350 = fmul contract float %v84, %v336
  store float %v350, ptr %v349, align 4
  br label %bb101
bb100:
  br label %bb101
bb101:
  %v351 = add i64 %v45, 64
  %v352 = icmp ult i64 %v351, %v52
  %v353 = xor i1 %v352, 1
  br i1 %v353, label %bb103, label %bb102
bb102:
  %v354 = add i64 %v68, %v45
  %v355 = add i64 %v354, 64
  %v356 = extractvalue { ptr, i64 } %v41, 0
  %v357 = getelementptr inbounds float, ptr %v356, i64 %v355
  %v358 = fmul contract float %v85, %v336
  store float %v358, ptr %v357, align 4
  br label %bb104
bb103:
  br label %bb104
bb104:
  %v359 = add i64 %v45, 96
  %v360 = icmp ult i64 %v359, %v52
  %v361 = xor i1 %v360, 1
  br i1 %v361, label %bb106, label %bb105
bb105:
  %v362 = add i64 %v68, %v45
  %v363 = add i64 %v362, 96
  %v364 = extractvalue { ptr, i64 } %v41, 0
  %v365 = getelementptr inbounds float, ptr %v364, i64 %v363
  %v366 = fmul contract float %v86, %v336
  store float %v366, ptr %v365, align 4
  br label %bb107
bb106:
  br label %bb107
bb107:
  br label %bb109
bb108:
  br label %bb109
bb109:
  br label %bb110
bb110:
  ret void
bb111:
  %v367 = fadd contract float %v253, %v256
  %v368 = udiv i32 %v254, 2
  br label %bb62
bb112:
  %v369 = fmul contract float %v257, %v34
  %v370 = xor i1 %v230, 1
  br i1 %v370, label %bb65, label %bb66
bb113:
  br label %bb69
bb114:
  %v371 = fmul contract float %v229, %v265
  %v372 = fadd contract float %v371, %v267
  %v373 = icmp ult i64 %v45, %v52
  %v374 = xor i1 %v373, 1
  br i1 %v374, label %bb73, label %bb71
bb115:
  call void @llvm.trap() #0
  unreachable
bb116:
  call void @llvm.trap() #0
  unreachable
bb117:
  call void @llvm.trap() #0
  unreachable
bb118:
  call void @llvm.trap() #0
  unreachable
bb119:
  call void @llvm.trap() #0
  unreachable
bb120:
  call void @llvm.trap() #0
  unreachable
bb121:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @attention_fa2_decode_paged(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, ptr %v16, i64 %v17) #0 {
entry:
  %v18 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v1, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v3, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v5, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v17, 1
  br label %bb0
bb0:
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi { ptr, i64 } [ %v23, %entry ]
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
  %v39 = phi { ptr, i64 } [ %v25, %entry ]
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v42 = urem i32 %v41, 32
  %v43 = zext i32 %v42 to i64
  %v44 = icmp uge i32 %v40, %v29
  %v45 = xor i1 %v44, 1
  br i1 %v45, label %bb3, label %bb5
bb3:
  %v46 = icmp uge i64 %v43, 32
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb4, label %bb5
bb4:
  %v48 = icmp ugt i32 %v31, 128
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb6, label %bb5
bb5:
  br label %bb74
bb6:
  %v50 = zext i32 %v31 to i64
  %v51 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v30, i32 1) #0
  br label %bb7
bb7:
  %v52 = icmp eq i32 %v51, 0
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb8, label %bb79
bb8:
  %v54 = udiv i32 %v29, %v51
  %v55 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v54, i32 1) #0
  br label %bb9
bb9:
  %v56 = icmp eq i32 %v55, 0
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb10, label %bb80
bb10:
  %v58 = udiv i32 %v40, %v55
  %v59 = zext i32 %v58 to i64
  %v60 = zext i32 %v36 to i64
  %v61 = zext i32 %v37 to i64
  %v62 = zext i32 %v38 to i64
  %v63 = mul i64 %v62, 2
  %v64 = mul i64 %v60, %v63
  %v65 = zext i32 %v40 to i64
  %v66 = mul i64 %v65, %v50
  br label %bb11
bb11:
  %v67 = phi float [ 0.0, %bb10 ], [ %v160, %bb57 ]
  %v68 = phi float [ 0.0, %bb10 ], [ %v161, %bb57 ]
  %v69 = phi float [ 0.0, %bb10 ], [ %v162, %bb57 ]
  %v70 = phi float [ 0.0, %bb10 ], [ %v163, %bb57 ]
  %v71 = phi float [ 0.0, %bb10 ], [ %v164, %bb57 ]
  %v72 = phi float [ 0.0, %bb10 ], [ %v165, %bb57 ]
  %v73 = phi i1 [ 0, %bb10 ], [ %v166, %bb57 ]
  %v74 = phi i32 [ 0, %bb10 ], [ %v81, %bb57 ]
  %v75 = icmp ult i32 %v74, %v32
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb58, label %bb12
bb12:
  %v77 = add i32 %v74, 16
  %v78 = icmp ugt i32 %v77, %v32
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb14, label %bb13
bb13:
  br label %bb15
bb14:
  %v80 = add i32 %v74, 16
  br label %bb15
bb15:
  %v81 = phi i32 [ %v32, %bb13 ], [ %v80, %bb14 ]
  %v82 = sub i32 %v81, %v74
  %v83 = zext i32 %v82 to i64
  %v84 = zext i32 %v41 to i64
  br label %bb16
bb16:
  %v85 = phi i64 [ %v84, %bb15 ], [ %v158, %bb23 ]
  %v86 = mul i64 %v83, %v50
  %v87 = icmp ult i64 %v85, %v86
  %v88 = xor i1 %v87, 1
  br i1 %v88, label %bb24, label %bb17
bb17:
  %v89 = icmp eq i64 %v50, 0
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb18, label %bb81
bb18:
  %v91 = udiv i64 %v85, %v50
  %v92 = urem i64 %v85, %v50
  %v93 = zext i32 %v74 to i64
  %v94 = add i64 %v93, %v91
  %v95 = icmp eq i64 %v60, 0
  %v96 = xor i1 %v95, 1
  br i1 %v96, label %bb19, label %bb82
bb19:
  %v97 = udiv i64 %v94, %v60
  %v98 = urem i64 %v94, %v60
  %v99 = zext i32 %v34 to i64
  %v100 = zext i32 %v35 to i64
  %v101 = mul i64 %v99, %v100
  %v102 = add i64 %v101, %v97
  %v103 = extractvalue { ptr, i64 } %v28, 1
  %v104 = icmp ult i64 %v102, %v103
  %v105 = extractvalue { ptr, i64 } %v28, 0
  %v106 = getelementptr inbounds i32, ptr %v105, i64 %v102
  %v107 = load i32, ptr %v106, align 4
  %v108 = zext i32 %v107 to i64
  %v109 = mul i64 %v108, %v61
  %v110 = mul i64 %v98, %v63
  %v111 = add i64 %v109, %v110
  %v112 = mul i64 %v59, %v50
  %v113 = mul i64 %v112, 2
  %v114 = add i64 %v111, %v113
  %v115 = add i64 %v109, %v64
  %v116 = add i64 %v115, %v110
  %v117 = add i64 %v116, %v113
  %v118 = mul i64 %v92, 2
  %v119 = add i64 %v114, %v118
  %v120 = extractvalue { ptr, i64 } %v27, 1
  %v121 = icmp ult i64 %v119, %v120
  %v122 = extractvalue { ptr, i64 } %v27, 0
  %v123 = getelementptr inbounds i8, ptr %v122, i64 %v119
  %v124 = load i8, ptr %v123, align 1
  %v125 = zext i8 %v124 to i16
  %v126 = add i64 %v119, 1
  %v127 = icmp ult i64 %v126, %v120
  %v128 = extractvalue { ptr, i64 } %v27, 0
  %v129 = getelementptr inbounds i8, ptr %v128, i64 %v126
  %v130 = load i8, ptr %v129, align 1
  %v131 = zext i8 %v130 to i16
  %v132 = trunc i32 8 to i16
  %v133 = and i16 %v132, 15
  %v134 = shl i16 %v131, %v133
  %v135 = or i16 %v125, %v134
  %v136 = add i64 %v117, %v118
  %v137 = icmp ult i64 %v136, %v120
  %v138 = extractvalue { ptr, i64 } %v27, 0
  %v139 = getelementptr inbounds i8, ptr %v138, i64 %v136
  %v140 = load i8, ptr %v139, align 1
  %v141 = zext i8 %v140 to i16
  %v142 = add i64 %v136, 1
  %v143 = icmp ult i64 %v142, %v120
  %v144 = extractvalue { ptr, i64 } %v27, 0
  %v145 = getelementptr inbounds i8, ptr %v144, i64 %v142
  %v146 = load i8, ptr %v145, align 1
  %v147 = zext i8 %v146 to i16
  %v148 = trunc i32 8 to i16
  %v149 = and i16 %v148, 15
  %v150 = shl i16 %v147, %v149
  %v151 = or i16 %v141, %v150
  %v152 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v135) #0
  br label %bb20
bb20:
  %v153 = mul i64 %v91, 128
  %v154 = add i64 %v153, %v92
  %v155 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_18, i64 %v154
  br label %bb21
bb21:
  store float %v152, ptr addrspace(3) %v155, align 4
  %v156 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v151) #0
  br label %bb22
bb22:
  %v157 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_19, i64 %v154
  br label %bb23
bb23:
  store float %v156, ptr addrspace(3) %v157, align 4
  %v158 = add i64 %v85, 32
  br label %bb16
bb24:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb25
bb25:
  br label %bb26
bb26:
  %v160 = phi float [ %v67, %bb25 ], [ %v211, %bb55 ]
  %v161 = phi float [ %v68, %bb25 ], [ %v224, %bb55 ]
  %v162 = phi float [ %v69, %bb25 ], [ %v237, %bb55 ]
  %v163 = phi float [ %v70, %bb25 ], [ %v250, %bb55 ]
  %v164 = phi float [ %v71, %bb25 ], [ %v199, %bb55 ]
  %v165 = phi float [ %v72, %bb25 ], [ %v292, %bb55 ]
  %v166 = phi i1 [ %v73, %bb25 ], [ 1, %bb55 ]
  %v167 = phi i64 [ 0, %bb25 ], [ %v251, %bb55 ]
  %v168 = icmp ult i64 %v167, %v83
  %v169 = xor i1 %v168, 1
  br i1 %v169, label %bb56, label %bb27
bb27:
  br label %bb28
bb28:
  %v170 = phi float [ 0.0, %bb27 ], [ %v186, %bb30 ]
  %v171 = phi i64 [ %v43, %bb27 ], [ %v187, %bb30 ]
  %v172 = icmp ult i64 %v171, %v50
  %v173 = xor i1 %v172, 1
  br i1 %v173, label %bb31, label %bb29
bb29:
  %v174 = bitcast ptr addrspace(3) @__shared_mem_18 to ptr addrspace(3)
  %v175 = mul i64 %v167, 128
  %v176 = add i64 %v175, %v171
  %v177 = getelementptr inbounds float, ptr addrspace(3) %v174, i64 %v176
  br label %bb30
bb30:
  %v178 = load float, ptr addrspace(3) %v177, align 4
  %v179 = add i64 %v66, %v171
  %v180 = extractvalue { ptr, i64 } %v26, 1
  %v181 = icmp ult i64 %v179, %v180
  %v182 = extractvalue { ptr, i64 } %v26, 0
  %v183 = getelementptr inbounds float, ptr %v182, i64 %v179
  %v184 = load float, ptr %v183, align 4
  %v185 = fmul contract float %v184, %v178
  %v186 = fadd contract float %v170, %v185
  %v187 = add i64 %v171, 32
  br label %bb28
bb31:
  br label %bb32
bb32:
  %v188 = phi float [ %v170, %bb31 ], [ %v287, %bb75 ]
  %v189 = phi i32 [ 16, %bb31 ], [ %v288, %bb75 ]
  %v190 = icmp eq i32 %v189, 0
  br i1 %v190, label %bb34, label %bb33
bb33:
  %v191 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v188, i32 %v189, i32 31) #0
  br label %bb75
bb34:
  %v192 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 4294967295, float %v188, i32 0, i32 31) #0
  br label %bb76
bb35:
  br label %bb40
bb36:
  %v193 = fcmp ogt float %v289, %v164
  %v194 = xor i1 %v193, 1
  br i1 %v194, label %bb38, label %bb37
bb37:
  %v195 = fsub contract float %v164, %v289
  %v196 = call float @__nv_expf(float %v195) #0
  br label %bb77
bb38:
  br label %bb39
bb39:
  %v197 = phi float [ %v164, %bb38 ], [ %v289, %bb77 ]
  %v198 = phi float [ 1.0, %bb38 ], [ %v196, %bb77 ]
  br label %bb40
bb40:
  %v199 = phi float [ %v289, %bb35 ], [ %v197, %bb39 ]
  %v200 = phi float [ 0.0, %bb35 ], [ %v198, %bb39 ]
  %v201 = fsub contract float %v289, %v199
  %v202 = call float @__nv_expf(float %v201) #0
  br label %bb78
bb41:
  %v203 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v204 = mul i64 %v167, 128
  %v205 = add i64 %v204, %v43
  %v206 = getelementptr inbounds float, ptr addrspace(3) %v203, i64 %v205
  br label %bb42
bb42:
  %v207 = load float, ptr addrspace(3) %v206, align 4
  %v208 = fmul contract float %v160, %v200
  %v209 = fmul contract float %v202, %v207
  %v210 = fadd contract float %v208, %v209
  br label %bb43
bb43:
  %v211 = phi float [ %v210, %bb42 ], [ %v160, %bb78 ]
  %v212 = add i64 %v43, 32
  %v213 = icmp ult i64 %v212, %v50
  %v214 = xor i1 %v213, 1
  br i1 %v214, label %bb46, label %bb44
bb44:
  %v215 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v216 = mul i64 %v167, 128
  %v217 = add i64 %v216, %v43
  %v218 = add i64 %v217, 32
  %v219 = getelementptr inbounds float, ptr addrspace(3) %v215, i64 %v218
  br label %bb45
bb45:
  %v220 = load float, ptr addrspace(3) %v219, align 4
  %v221 = fmul contract float %v161, %v200
  %v222 = fmul contract float %v202, %v220
  %v223 = fadd contract float %v221, %v222
  br label %bb47
bb46:
  br label %bb47
bb47:
  %v224 = phi float [ %v223, %bb45 ], [ %v161, %bb46 ]
  %v225 = add i64 %v43, 64
  %v226 = icmp ult i64 %v225, %v50
  %v227 = xor i1 %v226, 1
  br i1 %v227, label %bb50, label %bb48
bb48:
  %v228 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v229 = mul i64 %v167, 128
  %v230 = add i64 %v229, %v43
  %v231 = add i64 %v230, 64
  %v232 = getelementptr inbounds float, ptr addrspace(3) %v228, i64 %v231
  br label %bb49
bb49:
  %v233 = load float, ptr addrspace(3) %v232, align 4
  %v234 = fmul contract float %v162, %v200
  %v235 = fmul contract float %v202, %v233
  %v236 = fadd contract float %v234, %v235
  br label %bb51
bb50:
  br label %bb51
bb51:
  %v237 = phi float [ %v236, %bb49 ], [ %v162, %bb50 ]
  %v238 = add i64 %v43, 96
  %v239 = icmp ult i64 %v238, %v50
  %v240 = xor i1 %v239, 1
  br i1 %v240, label %bb54, label %bb52
bb52:
  %v241 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v242 = mul i64 %v167, 128
  %v243 = add i64 %v242, %v43
  %v244 = add i64 %v243, 96
  %v245 = getelementptr inbounds float, ptr addrspace(3) %v241, i64 %v244
  br label %bb53
bb53:
  %v246 = load float, ptr addrspace(3) %v245, align 4
  %v247 = fmul contract float %v163, %v200
  %v248 = fmul contract float %v202, %v246
  %v249 = fadd contract float %v247, %v248
  br label %bb55
bb54:
  br label %bb55
bb55:
  %v250 = phi float [ %v249, %bb53 ], [ %v163, %bb54 ]
  %v251 = add i64 %v167, 1
  br label %bb26
bb56:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb57
bb57:
  br label %bb11
bb58:
  %v253 = xor i1 %v73, 1
  br i1 %v253, label %bb73, label %bb59
bb59:
  %v254 = fcmp ogt float %v72, 0.0
  %v255 = xor i1 %v254, 1
  br i1 %v255, label %bb72, label %bb60
bb60:
  %v256 = fdiv contract float 1.0, %v72
  %v257 = icmp ult i64 %v43, %v50
  %v258 = xor i1 %v257, 1
  br i1 %v258, label %bb62, label %bb61
bb61:
  %v259 = add i64 %v66, %v43
  %v260 = extractvalue { ptr, i64 } %v39, 0
  %v261 = getelementptr inbounds float, ptr %v260, i64 %v259
  %v262 = fmul contract float %v67, %v256
  store float %v262, ptr %v261, align 4
  br label %bb62
bb62:
  %v263 = add i64 %v43, 32
  %v264 = icmp ult i64 %v263, %v50
  %v265 = xor i1 %v264, 1
  br i1 %v265, label %bb64, label %bb63
bb63:
  %v266 = add i64 %v66, %v43
  %v267 = add i64 %v266, 32
  %v268 = extractvalue { ptr, i64 } %v39, 0
  %v269 = getelementptr inbounds float, ptr %v268, i64 %v267
  %v270 = fmul contract float %v68, %v256
  store float %v270, ptr %v269, align 4
  br label %bb65
bb64:
  br label %bb65
bb65:
  %v271 = add i64 %v43, 64
  %v272 = icmp ult i64 %v271, %v50
  %v273 = xor i1 %v272, 1
  br i1 %v273, label %bb67, label %bb66
bb66:
  %v274 = add i64 %v66, %v43
  %v275 = add i64 %v274, 64
  %v276 = extractvalue { ptr, i64 } %v39, 0
  %v277 = getelementptr inbounds float, ptr %v276, i64 %v275
  %v278 = fmul contract float %v69, %v256
  store float %v278, ptr %v277, align 4
  br label %bb68
bb67:
  br label %bb68
bb68:
  %v279 = add i64 %v43, 96
  %v280 = icmp ult i64 %v279, %v50
  %v281 = xor i1 %v280, 1
  br i1 %v281, label %bb70, label %bb69
bb69:
  %v282 = add i64 %v66, %v43
  %v283 = add i64 %v282, 96
  %v284 = extractvalue { ptr, i64 } %v39, 0
  %v285 = getelementptr inbounds float, ptr %v284, i64 %v283
  %v286 = fmul contract float %v70, %v256
  store float %v286, ptr %v285, align 4
  br label %bb71
bb70:
  br label %bb71
bb71:
  br label %bb73
bb72:
  br label %bb73
bb73:
  br label %bb74
bb74:
  ret void
bb75:
  %v287 = fadd contract float %v188, %v191
  %v288 = udiv i32 %v189, 2
  br label %bb32
bb76:
  %v289 = fmul contract float %v192, %v33
  %v290 = xor i1 %v166, 1
  br i1 %v290, label %bb35, label %bb36
bb77:
  br label %bb39
bb78:
  %v291 = fmul contract float %v165, %v200
  %v292 = fadd contract float %v291, %v202
  %v293 = icmp ult i64 %v43, %v50
  %v294 = xor i1 %v293, 1
  br i1 %v294, label %bb43, label %bb41
bb79:
  call void @llvm.trap() #0
  unreachable
bb80:
  call void @llvm.trap() #0
  unreachable
bb81:
  call void @llvm.trap() #0
  unreachable
bb82:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @add_in_place_f32(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = alloca {  }, align 1
  %v11 = bitcast ptr %v10 to ptr
  %v12 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v11) #0
  br label %bb1
bb1:
  %v13 = icmp eq i64 %v12, 18446744073709551615
  br i1 %v13, label %bb9, label %bb6
bb2:
  %v14 = extractvalue { ptr } %v33, 0
  %v15 = extractvalue { ptr, i64 } %v8, 1
  %v16 = icmp ult i64 %v12, %v15
  br i1 %v16, label %bb3, label %bb13
bb3:
  %v17 = extractvalue { ptr, i64 } %v8, 0
  %v18 = getelementptr inbounds float, ptr %v17, i64 %v12
  %v19 = load float, ptr %v18, align 4
  %v20 = load float, ptr %v14, align 4
  %v21 = fadd contract float %v20, %v19
  store float %v21, ptr %v14, align 4
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v22 = extractvalue { ptr, i64 } %v9, 1
  %v23 = icmp ult i64 %v12, %v22
  %v24 = xor i1 %v23, 1
  br i1 %v24, label %bb8, label %bb7
bb7:
  %v25 = extractvalue { ptr, i64 } %v9, 0
  %v26 = getelementptr inbounds float, ptr %v25, i64 %v12
  %v27 = insertvalue { ptr } undef, ptr %v26, 0
  %v28 = extractvalue { ptr } %v27, 0
  br label %bb10
bb8:
  br label %bb9
bb9:
  %v29 = inttoptr i64 0 to ptr
  %v30 = insertvalue { ptr } undef, ptr %v29, 0
  %v31 = extractvalue { ptr } %v30, 0
  br label %bb10
bb10:
  %v32 = phi ptr [ %v28, %bb7 ], [ %v31, %bb9 ]
  %v33 = insertvalue { ptr } undef, ptr %v32, 0
  %v34 = extractvalue { ptr } %v33, 0
  %v35 = ptrtoint ptr %v34 to i64
  %v36 = sub i64 %v35, 0
  %v37 = icmp ule i64 %v36, 0
  %v38 = add i64 %v36, 0
  %v39 = select i1 %v37, i64 %v38, i64 1
  %v40 = icmp eq i64 %v39, 1
  br i1 %v40, label %bb2, label %bb11
bb11:
  %v41 = icmp eq i64 %v39, 0
  br i1 %v41, label %bb4, label %bb12
bb12:
  unreachable
bb13:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_q6k_project(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr %v10, i64 %v11) #0 {
entry:
  %v12 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v1, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v3, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v5, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v11, 1
  br label %bb0
bb0:
  %v20 = phi { ptr, i64 } [ %v13, %entry ]
  %v21 = phi { ptr, i64 } [ %v15, %entry ]
  %v22 = phi { ptr, i64 } [ %v17, %entry ]
  %v23 = phi i32 [ %v6, %entry ]
  %v24 = phi i32 [ %v7, %entry ]
  %v25 = phi i32 [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi { ptr, i64 } [ %v19, %entry ]
  %v28 = alloca {  }, align 1
  %v29 = bitcast ptr %v28 to ptr
  %v30 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v29) #0
  br label %bb1
bb1:
  %v31 = zext i32 %v23 to i64
  %v32 = zext i32 %v24 to i64
  %v33 = mul i64 %v31, %v32
  %v34 = zext i32 %v25 to i64
  %v35 = mul i64 %v33, %v34
  %v36 = icmp uge i64 %v30, %v35
  %v37 = xor i1 %v36, 1
  br i1 %v37, label %bb3, label %bb2
bb2:
  br label %bb10
bb3:
  %v38 = icmp eq i64 %v34, 0
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb4, label %bb18
bb4:
  %v40 = urem i64 %v30, %v34
  %v41 = udiv i64 %v30, %v34
  %v42 = extractvalue { ptr, i64 } %v22, 1
  %v43 = icmp ult i64 %v41, %v42
  br i1 %v43, label %bb5, label %bb19
bb5:
  %v44 = extractvalue { ptr, i64 } %v22, 0
  %v45 = getelementptr inbounds i32, ptr %v44, i64 %v41
  %v46 = load i32, ptr %v45, align 4
  %v47 = zext i32 %v46 to i64
  %v48 = udiv i32 %v26, 256
  %v49 = zext i32 %v48 to i64
  %v50 = mul i64 %v49, 210
  %v51 = mul i64 %v34, %v50
  %v52 = mul i64 %v47, %v51
  %v53 = mul i64 %v40, %v50
  %v54 = add i64 %v52, %v53
  %v55 = zext i32 %v26 to i64
  %v56 = mul i64 %v41, %v55
  %v57 = extractvalue { ptr, i64 } %v20, 0
  %v58 = extractvalue { ptr, i64 } %v20, 1
  %v59 = extractvalue { ptr, i64 } %v21, 0
  %v60 = extractvalue { ptr, i64 } %v21, 1
  %v61 = call float @cuda_kernels__oxide_kernels__kernels__dot_q6k(ptr %v57, i64 %v58, i64 %v54, ptr %v59, i64 %v60, i64 %v56, i32 %v48) #0
  br label %bb6
bb6:
  %v62 = bitcast ptr %v28 to ptr
  %v63 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v62) #0
  br label %bb7
bb7:
  %v64 = icmp eq i64 %v63, 18446744073709551615
  br i1 %v64, label %bb14, label %bb11
bb8:
  %v65 = extractvalue { ptr } %v77, 0
  store float %v61, ptr %v65, align 4
  br label %bb10
bb9:
  br label %bb10
bb10:
  ret void
bb11:
  %v66 = extractvalue { ptr, i64 } %v27, 1
  %v67 = icmp ult i64 %v63, %v66
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb13, label %bb12
bb12:
  %v69 = extractvalue { ptr, i64 } %v27, 0
  %v70 = getelementptr inbounds float, ptr %v69, i64 %v63
  %v71 = insertvalue { ptr } undef, ptr %v70, 0
  %v72 = extractvalue { ptr } %v71, 0
  br label %bb15
bb13:
  br label %bb14
bb14:
  %v73 = inttoptr i64 0 to ptr
  %v74 = insertvalue { ptr } undef, ptr %v73, 0
  %v75 = extractvalue { ptr } %v74, 0
  br label %bb15
bb15:
  %v76 = phi ptr [ %v72, %bb12 ], [ %v75, %bb14 ]
  %v77 = insertvalue { ptr } undef, ptr %v76, 0
  %v78 = extractvalue { ptr } %v77, 0
  %v79 = ptrtoint ptr %v78 to i64
  %v80 = sub i64 %v79, 0
  %v81 = icmp ule i64 %v80, 0
  %v82 = add i64 %v80, 0
  %v83 = select i1 %v81, i64 %v82, i64 1
  %v84 = icmp eq i64 %v83, 1
  br i1 %v84, label %bb8, label %bb16
bb16:
  %v85 = icmp eq i64 %v83, 0
  br i1 %v85, label %bb9, label %bb17
bb17:
  unreachable
bb18:
  call void @llvm.trap() #0
  unreachable
bb19:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_route_topk(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, float %v12, ptr %v13, i64 %v14, ptr %v15, i64 %v16) #0 {
entry:
  %v17 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v1, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v3, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v5, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v14, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v15, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v16, 1
  br label %bb0
bb0:
  %v27 = phi { ptr, i64 } [ %v18, %entry ]
  %v28 = phi { ptr, i64 } [ %v20, %entry ]
  %v29 = phi { ptr, i64 } [ %v22, %entry ]
  %v30 = phi i32 [ %v6, %entry ]
  %v31 = phi i32 [ %v7, %entry ]
  %v32 = phi i32 [ %v8, %entry ]
  %v33 = phi i32 [ %v9, %entry ]
  %v34 = phi i32 [ %v10, %entry ]
  %v35 = phi i32 [ %v11, %entry ]
  %v36 = phi float [ %v12, %entry ]
  %v37 = phi { ptr, i64 } [ %v24, %entry ]
  %v38 = phi { ptr, i64 } [ %v26, %entry ]
  %v39 = alloca {  }, align 1
  %v40 = bitcast ptr %v39 to ptr
  %v41 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v40) #0
  br label %bb1
bb1:
  %v42 = zext i32 %v30 to i64
  %v43 = icmp uge i64 %v41, %v42
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb3, label %bb2
bb2:
  br label %bb70
bb3:
  %v45 = zext i32 %v31 to i64
  %v46 = mul i64 %v41, %v45
  br label %bb4
bb4:
  %v47 = phi float [ -340282346638528860000000000000000000000.0, %bb3 ], [ %v82, %bb17 ]
  %v48 = phi i64 [ 0, %bb3 ], [ %v83, %bb17 ]
  %v49 = zext i32 %v32 to i64
  %v50 = icmp ult i64 %v48, %v49
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb18, label %bb5
bb5:
  %v52 = extractvalue { ptr, i64 } %v29, 1
  %v53 = icmp uge i64 %v52, %v49
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb8, label %bb6
bb6:
  %v55 = icmp ult i64 %v48, %v52
  br i1 %v55, label %bb7, label %bb74
bb7:
  %v56 = extractvalue { ptr, i64 } %v29, 0
  %v57 = getelementptr inbounds float, ptr %v56, i64 %v48
  %v58 = load float, ptr %v57, align 4
  br label %bb9
bb8:
  br label %bb9
bb9:
  %v59 = phi float [ %v58, %bb7 ], [ 0.0, %bb8 ]
  br label %bb10
bb10:
  %v60 = phi float [ %v59, %bb9 ], [ %v78, %bb13 ]
  %v61 = phi i64 [ 0, %bb9 ], [ %v79, %bb13 ]
  %v62 = icmp ult i64 %v61, %v45
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb14, label %bb11
bb11:
  %v64 = mul i64 %v48, %v45
  %v65 = add i64 %v64, %v61
  %v66 = extractvalue { ptr, i64 } %v28, 1
  %v67 = icmp ult i64 %v65, %v66
  br i1 %v67, label %bb12, label %bb75
bb12:
  %v68 = extractvalue { ptr, i64 } %v28, 0
  %v69 = getelementptr inbounds float, ptr %v68, i64 %v65
  %v70 = load float, ptr %v69, align 4
  %v71 = add i64 %v46, %v61
  %v72 = extractvalue { ptr, i64 } %v27, 1
  %v73 = icmp ult i64 %v71, %v72
  br i1 %v73, label %bb13, label %bb76
bb13:
  %v74 = extractvalue { ptr, i64 } %v27, 0
  %v75 = getelementptr inbounds float, ptr %v74, i64 %v71
  %v76 = load float, ptr %v75, align 4
  %v77 = fmul contract float %v70, %v76
  %v78 = fadd contract float %v60, %v77
  %v79 = add i64 %v61, 1
  br label %bb10
bb14:
  %v80 = fcmp ogt float %v60, %v47
  %v81 = xor i1 %v80, 1
  br i1 %v81, label %bb16, label %bb15
bb15:
  br label %bb17
bb16:
  br label %bb17
bb17:
  %v82 = phi float [ %v60, %bb15 ], [ %v47, %bb16 ]
  %v83 = add i64 %v48, 1
  br label %bb4
bb18:
  %v84 = icmp eq i32 %v34, 2
  br i1 %v84, label %bb32, label %bb19
bb19:
  br label %bb20
bb20:
  %v85 = phi float [ 0.0, %bb19 ], [ %v211, %bb71 ]
  %v86 = phi i64 [ 0, %bb19 ], [ %v212, %bb71 ]
  %v87 = icmp ult i64 %v86, %v49
  %v88 = xor i1 %v87, 1
  br i1 %v88, label %bb31, label %bb21
bb21:
  %v89 = extractvalue { ptr, i64 } %v29, 1
  %v90 = icmp uge i64 %v89, %v49
  %v91 = xor i1 %v90, 1
  br i1 %v91, label %bb24, label %bb22
bb22:
  %v92 = icmp ult i64 %v86, %v89
  br i1 %v92, label %bb23, label %bb77
bb23:
  %v93 = extractvalue { ptr, i64 } %v29, 0
  %v94 = getelementptr inbounds float, ptr %v93, i64 %v86
  %v95 = load float, ptr %v94, align 4
  br label %bb25
bb24:
  br label %bb25
bb25:
  %v96 = phi float [ %v95, %bb23 ], [ 0.0, %bb24 ]
  br label %bb26
bb26:
  %v97 = phi float [ %v96, %bb25 ], [ %v115, %bb29 ]
  %v98 = phi i64 [ 0, %bb25 ], [ %v116, %bb29 ]
  %v99 = icmp ult i64 %v98, %v45
  %v100 = xor i1 %v99, 1
  br i1 %v100, label %bb30, label %bb27
bb27:
  %v101 = mul i64 %v86, %v45
  %v102 = add i64 %v101, %v98
  %v103 = extractvalue { ptr, i64 } %v28, 1
  %v104 = icmp ult i64 %v102, %v103
  br i1 %v104, label %bb28, label %bb78
bb28:
  %v105 = extractvalue { ptr, i64 } %v28, 0
  %v106 = getelementptr inbounds float, ptr %v105, i64 %v102
  %v107 = load float, ptr %v106, align 4
  %v108 = add i64 %v46, %v98
  %v109 = extractvalue { ptr, i64 } %v27, 1
  %v110 = icmp ult i64 %v108, %v109
  br i1 %v110, label %bb29, label %bb79
bb29:
  %v111 = extractvalue { ptr, i64 } %v27, 0
  %v112 = getelementptr inbounds float, ptr %v111, i64 %v108
  %v113 = load float, ptr %v112, align 4
  %v114 = fmul contract float %v107, %v113
  %v115 = fadd contract float %v97, %v114
  %v116 = add i64 %v98, 1
  br label %bb26
bb30:
  %v117 = fsub contract float %v97, %v47
  %v118 = call float @__nv_expf(float %v117) #0
  br label %bb71
bb31:
  br label %bb32
bb32:
  %v119 = phi float [ 0.0, %bb18 ], [ %v85, %bb31 ]
  br label %bb33
bb33:
  %v120 = phi float [ 0.0, %bb32 ], [ %v194, %bb60 ]
  %v121 = phi i64 [ 0, %bb32 ], [ %v195, %bb60 ]
  %v122 = zext i32 %v33 to i64
  %v123 = icmp ult i64 %v121, %v122
  %v124 = xor i1 %v123, 1
  br i1 %v124, label %bb61, label %bb34
bb34:
  br label %bb35
bb35:
  %v125 = phi i64 [ 0, %bb34 ], [ %v185, %bb59 ]
  %v126 = phi float [ -340282346638528860000000000000000000000.0, %bb34 ], [ %v183, %bb59 ]
  %v127 = phi i64 [ 0, %bb34 ], [ %v184, %bb59 ]
  %v128 = icmp ult i64 %v125, %v49
  %v129 = xor i1 %v128, 1
  br i1 %v129, label %bb60, label %bb36
bb36:
  br label %bb37
bb37:
  %v130 = phi i1 [ 0, %bb36 ], [ %v142, %bb41 ]
  %v131 = phi i64 [ 0, %bb36 ], [ %v143, %bb41 ]
  %v132 = icmp ult i64 %v131, %v121
  %v133 = xor i1 %v132, 1
  br i1 %v133, label %bb42, label %bb38
bb38:
  %v134 = mul i64 %v41, %v122
  %v135 = add i64 %v134, %v131
  %v136 = extractvalue { ptr, i64 } %v37, 0
  %v137 = getelementptr inbounds i32, ptr %v136, i64 %v135
  %v138 = load i32, ptr %v137, align 4
  %v139 = zext i32 %v138 to i64
  %v140 = icmp eq i64 %v139, %v125
  %v141 = xor i1 %v140, 1
  br i1 %v141, label %bb40, label %bb39
bb39:
  br label %bb41
bb40:
  br label %bb41
bb41:
  %v142 = phi i1 [ 1, %bb39 ], [ %v130, %bb40 ]
  %v143 = add i64 %v131, 1
  br label %bb37
bb42:
  %v144 = xor i1 %v130, 1
  br i1 %v144, label %bb43, label %bb59
bb43:
  %v145 = extractvalue { ptr, i64 } %v29, 1
  %v146 = icmp uge i64 %v145, %v49
  %v147 = xor i1 %v146, 1
  br i1 %v147, label %bb46, label %bb44
bb44:
  %v148 = icmp ult i64 %v125, %v145
  br i1 %v148, label %bb45, label %bb80
bb45:
  %v149 = extractvalue { ptr, i64 } %v29, 0
  %v150 = getelementptr inbounds float, ptr %v149, i64 %v125
  %v151 = load float, ptr %v150, align 4
  br label %bb47
bb46:
  br label %bb47
bb47:
  %v152 = phi float [ %v151, %bb45 ], [ 0.0, %bb46 ]
  br label %bb48
bb48:
  %v153 = phi float [ %v152, %bb47 ], [ %v171, %bb51 ]
  %v154 = phi i64 [ 0, %bb47 ], [ %v172, %bb51 ]
  %v155 = icmp ult i64 %v154, %v45
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb52, label %bb49
bb49:
  %v157 = mul i64 %v125, %v45
  %v158 = add i64 %v157, %v154
  %v159 = extractvalue { ptr, i64 } %v28, 1
  %v160 = icmp ult i64 %v158, %v159
  br i1 %v160, label %bb50, label %bb81
bb50:
  %v161 = extractvalue { ptr, i64 } %v28, 0
  %v162 = getelementptr inbounds float, ptr %v161, i64 %v158
  %v163 = load float, ptr %v162, align 4
  %v164 = add i64 %v46, %v154
  %v165 = extractvalue { ptr, i64 } %v27, 1
  %v166 = icmp ult i64 %v164, %v165
  br i1 %v166, label %bb51, label %bb82
bb51:
  %v167 = extractvalue { ptr, i64 } %v27, 0
  %v168 = getelementptr inbounds float, ptr %v167, i64 %v164
  %v169 = load float, ptr %v168, align 4
  %v170 = fmul contract float %v163, %v169
  %v171 = fadd contract float %v153, %v170
  %v172 = add i64 %v154, 1
  br label %bb48
bb52:
  %v173 = icmp eq i32 %v34, 2
  br i1 %v173, label %bb53, label %bb54
bb53:
  %v174 = fneg float %v153
  %v175 = call float @__nv_expf(float %v174) #0
  br label %bb72
bb54:
  %v176 = fsub contract float %v153, %v47
  %v177 = call float @__nv_expf(float %v176) #0
  br label %bb73
bb55:
  %v178 = phi float [ %v214, %bb72 ], [ %v215, %bb73 ]
  %v179 = fcmp ogt float %v178, %v126
  %v180 = xor i1 %v179, 1
  br i1 %v180, label %bb57, label %bb56
bb56:
  br label %bb58
bb57:
  br label %bb58
bb58:
  %v181 = phi float [ %v178, %bb56 ], [ %v126, %bb57 ]
  %v182 = phi i64 [ %v125, %bb56 ], [ %v127, %bb57 ]
  br label %bb59
bb59:
  %v183 = phi float [ %v126, %bb42 ], [ %v181, %bb58 ]
  %v184 = phi i64 [ %v127, %bb42 ], [ %v182, %bb58 ]
  %v185 = add i64 %v125, 1
  br label %bb35
bb60:
  %v186 = mul i64 %v41, %v122
  %v187 = add i64 %v186, %v121
  %v188 = extractvalue { ptr, i64 } %v37, 0
  %v189 = getelementptr inbounds i32, ptr %v188, i64 %v187
  %v190 = trunc i64 %v127 to i32
  store i32 %v190, ptr %v189, align 4
  %v191 = add i64 %v186, %v121
  %v192 = extractvalue { ptr, i64 } %v38, 0
  %v193 = getelementptr inbounds float, ptr %v192, i64 %v191
  store float %v126, ptr %v193, align 4
  %v194 = fadd contract float %v120, %v126
  %v195 = add i64 %v121, 1
  br label %bb33
bb61:
  br label %bb62
bb62:
  %v196 = phi i64 [ 0, %bb61 ], [ %v210, %bb68 ]
  %v197 = icmp ult i64 %v196, %v122
  %v198 = xor i1 %v197, 1
  br i1 %v198, label %bb69, label %bb63
bb63:
  %v199 = mul i64 %v41, %v122
  %v200 = add i64 %v199, %v196
  %v201 = extractvalue { ptr, i64 } %v38, 0
  %v202 = getelementptr inbounds float, ptr %v201, i64 %v200
  %v203 = load float, ptr %v202, align 4
  %v204 = icmp eq i32 %v35, 0
  br i1 %v204, label %bb67, label %bb64
bb64:
  %v205 = fcmp ogt float %v120, 0.0
  %v206 = xor i1 %v205, 1
  br i1 %v206, label %bb66, label %bb65
bb65:
  %v207 = fdiv contract float %v203, %v120
  br label %bb68
bb66:
  br label %bb67
bb67:
  br label %bb68
bb68:
  %v208 = phi float [ %v207, %bb65 ], [ %v203, %bb67 ]
  %v209 = fmul contract float %v208, %v36
  store float %v209, ptr %v202, align 4
  %v210 = add i64 %v196, 1
  br label %bb62
bb69:
  br label %bb70
bb70:
  ret void
bb71:
  %v211 = fadd contract float %v85, %v118
  %v212 = add i64 %v86, 1
  br label %bb20
bb72:
  %v213 = fadd contract float 1.0, %v175
  %v214 = fdiv contract float 1.0, %v213
  br label %bb55
bb73:
  %v215 = fdiv contract float %v177, %v119
  br label %bb55
bb74:
  call void @llvm.trap() #0
  unreachable
bb75:
  call void @llvm.trap() #0
  unreachable
bb76:
  call void @llvm.trap() #0
  unreachable
bb77:
  call void @llvm.trap() #0
  unreachable
bb78:
  call void @llvm.trap() #0
  unreachable
bb79:
  call void @llvm.trap() #0
  unreachable
bb80:
  call void @llvm.trap() #0
  unreachable
bb81:
  call void @llvm.trap() #0
  unreachable
bb82:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @attention_paged_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, ptr %v16, i64 %v17) #0 {
entry:
  %v18 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v1, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v3, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v5, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v17, 1
  br label %bb0
bb0:
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi { ptr, i64 } [ %v23, %entry ]
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
  %v39 = phi { ptr, i64 } [ %v25, %entry ]
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v41 = zext i32 %v40 to i64
  %v42 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v43 = icmp uge i32 %v42, %v29
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb3, label %bb5
bb3:
  %v45 = icmp uge i64 %v41, 32
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb4, label %bb5
bb4:
  %v47 = icmp ugt i32 %v31, 128
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb6, label %bb5
bb5:
  br label %bb51
bb6:
  %v49 = zext i32 %v31 to i64
  %v50 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v30, i32 1) #0
  br label %bb7
bb7:
  %v51 = icmp eq i32 %v50, 0
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb8, label %bb56
bb8:
  %v53 = udiv i32 %v29, %v50
  %v54 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v53, i32 1) #0
  br label %bb9
bb9:
  %v55 = icmp eq i32 %v54, 0
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb10, label %bb57
bb10:
  %v57 = udiv i32 %v42, %v54
  %v58 = zext i32 %v57 to i64
  %v59 = zext i32 %v38 to i64
  %v60 = zext i32 %v36 to i64
  %v61 = zext i32 %v37 to i64
  %v62 = mul i64 %v59, 2
  %v63 = mul i64 %v60, %v62
  %v64 = zext i32 %v42 to i64
  %v65 = mul i64 %v64, %v49
  br label %bb11
bb11:
  %v66 = phi float [ 0.0, %bb10 ], [ %v167, %bb38 ]
  %v67 = phi float [ 0.0, %bb10 ], [ %v193, %bb38 ]
  %v68 = phi float [ 0.0, %bb10 ], [ %v219, %bb38 ]
  %v69 = phi float [ 0.0, %bb10 ], [ %v245, %bb38 ]
  %v70 = phi float [ 0.0, %bb10 ], [ %v141, %bb38 ]
  %v71 = phi float [ 0.0, %bb10 ], [ %v283, %bb38 ]
  %v72 = phi i64 [ 0, %bb10 ], [ %v246, %bb38 ]
  %v73 = zext i32 %v32 to i64
  %v74 = icmp ult i64 %v72, %v73
  %v75 = xor i1 %v74, 1
  br i1 %v75, label %bb39, label %bb12
bb12:
  %v76 = icmp eq i64 %v60, 0
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb13, label %bb58
bb13:
  %v78 = udiv i64 %v72, %v60
  %v79 = urem i64 %v72, %v60
  %v80 = zext i32 %v34 to i64
  %v81 = zext i32 %v35 to i64
  %v82 = mul i64 %v80, %v81
  %v83 = add i64 %v82, %v78
  %v84 = extractvalue { ptr, i64 } %v28, 1
  %v85 = icmp ult i64 %v83, %v84
  %v86 = extractvalue { ptr, i64 } %v28, 0
  %v87 = getelementptr inbounds i32, ptr %v86, i64 %v83
  %v88 = load i32, ptr %v87, align 4
  %v89 = zext i32 %v88 to i64
  %v90 = mul i64 %v89, %v61
  %v91 = mul i64 %v79, %v62
  %v92 = add i64 %v90, %v91
  %v93 = mul i64 %v58, %v49
  %v94 = mul i64 %v93, 2
  %v95 = add i64 %v92, %v94
  br label %bb14
bb14:
  %v96 = phi float [ 0.0, %bb13 ], [ %v128, %bb16 ]
  %v97 = phi i64 [ %v41, %bb13 ], [ %v129, %bb16 ]
  %v98 = icmp ult i64 %v97, %v49
  %v99 = xor i1 %v98, 1
  br i1 %v99, label %bb17, label %bb15
bb15:
  %v100 = mul i64 %v97, 2
  %v101 = add i64 %v95, %v100
  %v102 = extractvalue { ptr, i64 } %v27, 1
  %v103 = icmp ult i64 %v101, %v102
  %v104 = extractvalue { ptr, i64 } %v27, 0
  %v105 = getelementptr inbounds i8, ptr %v104, i64 %v101
  %v106 = load i8, ptr %v105, align 1
  %v107 = zext i8 %v106 to i16
  %v108 = mul i64 %v97, 2
  %v109 = add i64 %v95, %v108
  %v110 = add i64 %v109, 1
  %v111 = icmp ult i64 %v110, %v102
  %v112 = extractvalue { ptr, i64 } %v27, 0
  %v113 = getelementptr inbounds i8, ptr %v112, i64 %v110
  %v114 = load i8, ptr %v113, align 1
  %v115 = zext i8 %v114 to i16
  %v116 = trunc i32 8 to i16
  %v117 = and i16 %v116, 15
  %v118 = shl i16 %v115, %v117
  %v119 = or i16 %v107, %v118
  %v120 = add i64 %v65, %v97
  %v121 = extractvalue { ptr, i64 } %v26, 1
  %v122 = icmp ult i64 %v120, %v121
  %v123 = extractvalue { ptr, i64 } %v26, 0
  %v124 = getelementptr inbounds float, ptr %v123, i64 %v120
  %v125 = load float, ptr %v124, align 4
  %v126 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v119) #0
  br label %bb16
bb16:
  %v127 = fmul contract float %v125, %v126
  %v128 = fadd contract float %v96, %v127
  %v129 = add i64 %v97, 32
  br label %bb14
bb17:
  br label %bb18
bb18:
  %v130 = phi float [ %v96, %bb17 ], [ %v278, %bb52 ]
  %v131 = phi i32 [ 16, %bb17 ], [ %v279, %bb52 ]
  %v132 = icmp eq i32 %v131, 0
  br i1 %v132, label %bb20, label %bb19
bb19:
  %v133 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v130, i32 %v131, i32 31) #0
  br label %bb52
bb20:
  %v134 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 4294967295, float %v130, i32 0, i32 31) #0
  br label %bb53
bb21:
  br label %bb26
bb22:
  %v135 = fcmp ogt float %v280, %v70
  %v136 = xor i1 %v135, 1
  br i1 %v136, label %bb24, label %bb23
bb23:
  %v137 = fsub contract float %v70, %v280
  %v138 = call float @__nv_expf(float %v137) #0
  br label %bb54
bb24:
  br label %bb25
bb25:
  %v139 = phi float [ %v70, %bb24 ], [ %v280, %bb54 ]
  %v140 = phi float [ 1.0, %bb24 ], [ %v138, %bb54 ]
  br label %bb26
bb26:
  %v141 = phi float [ %v280, %bb21 ], [ %v139, %bb25 ]
  %v142 = phi float [ 0.0, %bb21 ], [ %v140, %bb25 ]
  %v143 = fsub contract float %v280, %v141
  %v144 = call float @__nv_expf(float %v143) #0
  br label %bb55
bb27:
  %v145 = mul i64 %v41, 2
  %v146 = add i64 %v286, %v145
  %v147 = extractvalue { ptr, i64 } %v27, 1
  %v148 = icmp ult i64 %v146, %v147
  %v149 = extractvalue { ptr, i64 } %v27, 0
  %v150 = getelementptr inbounds i8, ptr %v149, i64 %v146
  %v151 = load i8, ptr %v150, align 1
  %v152 = zext i8 %v151 to i16
  %v153 = add i64 %v146, 1
  %v154 = icmp ult i64 %v153, %v147
  %v155 = extractvalue { ptr, i64 } %v27, 0
  %v156 = getelementptr inbounds i8, ptr %v155, i64 %v153
  %v157 = load i8, ptr %v156, align 1
  %v158 = zext i8 %v157 to i16
  %v159 = trunc i32 8 to i16
  %v160 = and i16 %v159, 15
  %v161 = shl i16 %v158, %v160
  %v162 = or i16 %v152, %v161
  %v163 = fmul contract float %v66, %v142
  %v164 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v162) #0
  br label %bb28
bb28:
  %v165 = fmul contract float %v144, %v164
  %v166 = fadd contract float %v163, %v165
  br label %bb29
bb29:
  %v167 = phi float [ %v166, %bb28 ], [ %v66, %bb55 ]
  %v168 = add i64 %v41, 32
  %v169 = icmp ult i64 %v168, %v49
  %v170 = xor i1 %v169, 1
  br i1 %v170, label %bb32, label %bb30
bb30:
  %v171 = mul i64 %v168, 2
  %v172 = add i64 %v286, %v171
  %v173 = extractvalue { ptr, i64 } %v27, 1
  %v174 = icmp ult i64 %v172, %v173
  %v175 = extractvalue { ptr, i64 } %v27, 0
  %v176 = getelementptr inbounds i8, ptr %v175, i64 %v172
  %v177 = load i8, ptr %v176, align 1
  %v178 = zext i8 %v177 to i16
  %v179 = add i64 %v172, 1
  %v180 = icmp ult i64 %v179, %v173
  %v181 = extractvalue { ptr, i64 } %v27, 0
  %v182 = getelementptr inbounds i8, ptr %v181, i64 %v179
  %v183 = load i8, ptr %v182, align 1
  %v184 = zext i8 %v183 to i16
  %v185 = trunc i32 8 to i16
  %v186 = and i16 %v185, 15
  %v187 = shl i16 %v184, %v186
  %v188 = or i16 %v178, %v187
  %v189 = fmul contract float %v67, %v142
  %v190 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v188) #0
  br label %bb31
bb31:
  %v191 = fmul contract float %v144, %v190
  %v192 = fadd contract float %v189, %v191
  br label %bb32
bb32:
  %v193 = phi float [ %v67, %bb29 ], [ %v192, %bb31 ]
  %v194 = add i64 %v41, 64
  %v195 = icmp ult i64 %v194, %v49
  %v196 = xor i1 %v195, 1
  br i1 %v196, label %bb35, label %bb33
bb33:
  %v197 = mul i64 %v194, 2
  %v198 = add i64 %v286, %v197
  %v199 = extractvalue { ptr, i64 } %v27, 1
  %v200 = icmp ult i64 %v198, %v199
  %v201 = extractvalue { ptr, i64 } %v27, 0
  %v202 = getelementptr inbounds i8, ptr %v201, i64 %v198
  %v203 = load i8, ptr %v202, align 1
  %v204 = zext i8 %v203 to i16
  %v205 = add i64 %v198, 1
  %v206 = icmp ult i64 %v205, %v199
  %v207 = extractvalue { ptr, i64 } %v27, 0
  %v208 = getelementptr inbounds i8, ptr %v207, i64 %v205
  %v209 = load i8, ptr %v208, align 1
  %v210 = zext i8 %v209 to i16
  %v211 = trunc i32 8 to i16
  %v212 = and i16 %v211, 15
  %v213 = shl i16 %v210, %v212
  %v214 = or i16 %v204, %v213
  %v215 = fmul contract float %v68, %v142
  %v216 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v214) #0
  br label %bb34
bb34:
  %v217 = fmul contract float %v144, %v216
  %v218 = fadd contract float %v215, %v217
  br label %bb35
bb35:
  %v219 = phi float [ %v68, %bb32 ], [ %v218, %bb34 ]
  %v220 = add i64 %v41, 96
  %v221 = icmp ult i64 %v220, %v49
  %v222 = xor i1 %v221, 1
  br i1 %v222, label %bb38, label %bb36
bb36:
  %v223 = mul i64 %v220, 2
  %v224 = add i64 %v286, %v223
  %v225 = extractvalue { ptr, i64 } %v27, 1
  %v226 = icmp ult i64 %v224, %v225
  %v227 = extractvalue { ptr, i64 } %v27, 0
  %v228 = getelementptr inbounds i8, ptr %v227, i64 %v224
  %v229 = load i8, ptr %v228, align 1
  %v230 = zext i8 %v229 to i16
  %v231 = add i64 %v224, 1
  %v232 = icmp ult i64 %v231, %v225
  %v233 = extractvalue { ptr, i64 } %v27, 0
  %v234 = getelementptr inbounds i8, ptr %v233, i64 %v231
  %v235 = load i8, ptr %v234, align 1
  %v236 = zext i8 %v235 to i16
  %v237 = trunc i32 8 to i16
  %v238 = and i16 %v237, 15
  %v239 = shl i16 %v236, %v238
  %v240 = or i16 %v230, %v239
  %v241 = fmul contract float %v69, %v142
  %v242 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v240) #0
  br label %bb37
bb37:
  %v243 = fmul contract float %v144, %v242
  %v244 = fadd contract float %v241, %v243
  br label %bb38
bb38:
  %v245 = phi float [ %v69, %bb35 ], [ %v244, %bb37 ]
  %v246 = add i64 %v72, 1
  br label %bb11
bb39:
  %v247 = fdiv contract float 1.0, %v71
  %v248 = icmp ult i64 %v41, %v49
  %v249 = xor i1 %v248, 1
  br i1 %v249, label %bb41, label %bb40
bb40:
  %v250 = add i64 %v65, %v41
  %v251 = extractvalue { ptr, i64 } %v39, 0
  %v252 = getelementptr inbounds float, ptr %v251, i64 %v250
  %v253 = fmul contract float %v66, %v247
  store float %v253, ptr %v252, align 4
  br label %bb41
bb41:
  %v254 = add i64 %v41, 32
  %v255 = icmp ult i64 %v254, %v49
  %v256 = xor i1 %v255, 1
  br i1 %v256, label %bb43, label %bb42
bb42:
  %v257 = add i64 %v65, %v41
  %v258 = add i64 %v257, 32
  %v259 = extractvalue { ptr, i64 } %v39, 0
  %v260 = getelementptr inbounds float, ptr %v259, i64 %v258
  %v261 = fmul contract float %v67, %v247
  store float %v261, ptr %v260, align 4
  br label %bb44
bb43:
  br label %bb44
bb44:
  %v262 = add i64 %v41, 64
  %v263 = icmp ult i64 %v262, %v49
  %v264 = xor i1 %v263, 1
  br i1 %v264, label %bb46, label %bb45
bb45:
  %v265 = add i64 %v65, %v41
  %v266 = add i64 %v265, 64
  %v267 = extractvalue { ptr, i64 } %v39, 0
  %v268 = getelementptr inbounds float, ptr %v267, i64 %v266
  %v269 = fmul contract float %v68, %v247
  store float %v269, ptr %v268, align 4
  br label %bb47
bb46:
  br label %bb47
bb47:
  %v270 = add i64 %v41, 96
  %v271 = icmp ult i64 %v270, %v49
  %v272 = xor i1 %v271, 1
  br i1 %v272, label %bb49, label %bb48
bb48:
  %v273 = add i64 %v65, %v41
  %v274 = add i64 %v273, 96
  %v275 = extractvalue { ptr, i64 } %v39, 0
  %v276 = getelementptr inbounds float, ptr %v275, i64 %v274
  %v277 = fmul contract float %v69, %v247
  store float %v277, ptr %v276, align 4
  br label %bb50
bb49:
  br label %bb50
bb50:
  br label %bb51
bb51:
  ret void
bb52:
  %v278 = fadd contract float %v130, %v133
  %v279 = udiv i32 %v131, 2
  br label %bb18
bb53:
  %v280 = fmul contract float %v134, %v33
  %v281 = icmp eq i64 %v72, 0
  br i1 %v281, label %bb21, label %bb22
bb54:
  br label %bb25
bb55:
  %v282 = fmul contract float %v71, %v142
  %v283 = fadd contract float %v282, %v144
  %v284 = add i64 %v90, %v63
  %v285 = add i64 %v284, %v91
  %v286 = add i64 %v285, %v94
  %v287 = icmp ult i64 %v41, %v49
  %v288 = xor i1 %v287, 1
  br i1 %v288, label %bb29, label %bb27
bb56:
  call void @llvm.trap() #0
  unreachable
bb57:
  call void @llvm.trap() #0
  unreachable
bb58:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @attention_heads(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, ptr %v11, i64 %v12) #0 {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v11, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v12, 1
  br label %bb0
bb0:
  %v21 = phi { ptr, i64 } [ %v14, %entry ]
  %v22 = phi { ptr, i64 } [ %v16, %entry ]
  %v23 = phi { ptr, i64 } [ %v18, %entry ]
  %v24 = phi i32 [ %v6, %entry ]
  %v25 = phi i32 [ %v7, %entry ]
  %v26 = phi i32 [ %v8, %entry ]
  %v27 = phi i32 [ %v9, %entry ]
  %v28 = phi float [ %v10, %entry ]
  %v29 = phi { ptr, i64 } [ %v20, %entry ]
  %v30 = alloca {  }, align 1
  %v31 = bitcast ptr %v30 to ptr
  %v32 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v31) #0
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
  %v38 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v25, i32 1) #0
  br label %bb4
bb4:
  %v39 = icmp eq i32 %v38, 0
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb5, label %bb38
bb5:
  %v41 = udiv i32 %v24, %v38
  %v42 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v41, i32 1) #0
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
  %v53 = extractvalue { ptr, i64 } %v29, 0
  %v54 = getelementptr inbounds float, ptr %v53, i64 %v52
  store float 0.0, ptr %v54, align 4
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
  %v72 = extractvalue { ptr, i64 } %v21, 1
  %v73 = icmp ult i64 %v71, %v72
  br i1 %v73, label %bb15, label %bb40
bb15:
  %v74 = extractvalue { ptr, i64 } %v21, 0
  %v75 = getelementptr inbounds float, ptr %v74, i64 %v71
  %v76 = load float, ptr %v75, align 4
  %v77 = add i64 %v66, %v68
  %v78 = extractvalue { ptr, i64 } %v22, 1
  %v79 = icmp ult i64 %v77, %v78
  br i1 %v79, label %bb16, label %bb41
bb16:
  %v80 = extractvalue { ptr, i64 } %v22, 0
  %v81 = getelementptr inbounds float, ptr %v80, i64 %v77
  %v82 = load float, ptr %v81, align 4
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
  %v102 = extractvalue { ptr, i64 } %v29, 0
  %v103 = getelementptr inbounds float, ptr %v102, i64 %v101
  %v104 = load float, ptr %v103, align 4
  %v105 = fmul contract float %v104, %v95
  %v106 = add i64 %v132, %v98
  %v107 = extractvalue { ptr, i64 } %v23, 1
  %v108 = icmp ult i64 %v106, %v107
  br i1 %v108, label %bb26, label %bb42
bb26:
  %v109 = extractvalue { ptr, i64 } %v23, 0
  %v110 = getelementptr inbounds float, ptr %v109, i64 %v106
  %v111 = load float, ptr %v110, align 4
  %v112 = fmul contract float %v97, %v111
  %v113 = fadd contract float %v105, %v112
  store float %v113, ptr %v103, align 4
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
  %v123 = extractvalue { ptr, i64 } %v29, 0
  %v124 = getelementptr inbounds float, ptr %v123, i64 %v122
  %v125 = load float, ptr %v124, align 4
  %v126 = fmul contract float %v125, %v118
  store float %v126, ptr %v124, align 4
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
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @kv_write_row(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  br label %bb0
bb0:
  %v20 = phi { ptr, i64 } [ %v15, %entry ]
  %v21 = phi { ptr, i64 } [ %v17, %entry ]
  %v22 = phi { ptr, i64 } [ %v19, %entry ]
  %v23 = phi i32 [ %v6, %entry ]
  %v24 = phi i32 [ %v7, %entry ]
  %v25 = phi i32 [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi i32 [ %v10, %entry ]
  %v28 = phi i32 [ %v11, %entry ]
  %v29 = phi i32 [ %v12, %entry ]
  %v30 = phi i32 [ %v13, %entry ]
  %v31 = alloca {  }, align 1
  %v32 = bitcast ptr %v31 to ptr
  %v33 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v32) #0
  br label %bb1
bb1:
  %v34 = trunc i64 %v33 to i32
  %v35 = icmp uge i32 %v34, %v28
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb3, label %bb2
bb2:
  br label %bb11
bb3:
  %v37 = icmp eq i32 %v29, 0
  %v38 = xor i1 %v37, 1
  br i1 %v38, label %bb4, label %bb12
bb4:
  %v39 = udiv i32 %v24, %v29
  %v40 = urem i32 %v24, %v29
  %v41 = mul i32 %v23, %v26
  %v42 = add i32 %v41, %v39
  %v43 = zext i32 %v42 to i64
  %v44 = extractvalue { ptr, i64 } %v22, 1
  %v45 = icmp ult i64 %v43, %v44
  br i1 %v45, label %bb5, label %bb13
bb5:
  %v46 = extractvalue { ptr, i64 } %v22, 0
  %v47 = getelementptr inbounds i32, ptr %v46, i64 %v43
  %v48 = load i32, ptr %v47, align 4
  %v49 = zext i32 %v48 to i64
  %v50 = zext i32 %v27 to i64
  %v51 = mul i64 %v50, 2
  %v52 = icmp eq i32 %v25, 0
  br i1 %v52, label %bb7, label %bb6
bb6:
  %v53 = zext i32 %v29 to i64
  %v54 = mul i64 %v53, %v51
  br label %bb8
bb7:
  br label %bb8
bb8:
  %v55 = phi i64 [ %v54, %bb6 ], [ 0, %bb7 ]
  %v56 = zext i32 %v30 to i64
  %v57 = mul i64 %v49, %v56
  %v58 = add i64 %v57, %v55
  %v59 = zext i32 %v40 to i64
  %v60 = mul i64 %v59, %v51
  %v61 = add i64 %v58, %v60
  %v62 = zext i32 %v34 to i64
  %v63 = extractvalue { ptr, i64 } %v20, 1
  %v64 = icmp ult i64 %v62, %v63
  br i1 %v64, label %bb9, label %bb14
bb9:
  %v65 = extractvalue { ptr, i64 } %v20, 0
  %v66 = getelementptr inbounds float, ptr %v65, i64 %v62
  %v67 = load float, ptr %v66, align 4
  %v68 = call i16 @cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits(float %v67) #0
  br label %bb10
bb10:
  %v69 = and i16 %v68, 255
  %v70 = trunc i16 %v69 to i8
  %v71 = trunc i32 8 to i16
  %v72 = and i16 %v71, 15
  %v73 = lshr i16 %v68, %v72
  %v74 = trunc i16 %v73 to i8
  %v75 = mul i64 %v62, 2
  %v76 = add i64 %v61, %v75
  %v77 = extractvalue { ptr, i64 } %v21, 0
  %v78 = getelementptr inbounds i8, ptr %v77, i64 %v76
  store i8 %v70, ptr %v78, align 1
  %v79 = add i64 %v76, 1
  %v80 = getelementptr inbounds i8, ptr %v77, i64 %v79
  store i8 %v74, ptr %v80, align 1
  br label %bb11
bb11:
  ret void
bb12:
  call void @llvm.trap() #0
  unreachable
bb13:
  call void @llvm.trap() #0
  unreachable
bb14:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @embedding_f32(ptr %v0, i64 %v1, i32 %v2, i32 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi i32 [ %v2, %entry ]
  %v12 = phi i32 [ %v3, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = alloca {  }, align 1
  %v15 = bitcast ptr %v14 to ptr
  %v16 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v15) #0
  br label %bb1
bb1:
  %v17 = zext i32 %v12 to i64
  %v18 = icmp uge i64 %v16, %v17
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb3, label %bb2
bb2:
  br label %bb8
bb3:
  %v20 = zext i32 %v11 to i64
  %v21 = mul i64 %v20, %v17
  %v22 = add i64 %v21, %v16
  %v23 = icmp eq i64 %v16, 18446744073709551615
  br i1 %v23, label %bb12, label %bb9
bb4:
  %v24 = extractvalue { ptr } %v41, 0
  %v25 = extractvalue { ptr, i64 } %v10, 1
  %v26 = icmp ult i64 %v22, %v25
  br i1 %v26, label %bb5, label %bb16
bb5:
  %v27 = extractvalue { ptr, i64 } %v10, 0
  %v28 = getelementptr inbounds float, ptr %v27, i64 %v22
  %v29 = load float, ptr %v28, align 4
  store float %v29, ptr %v24, align 4
  br label %bb7
bb6:
  br label %bb7
bb7:
  br label %bb8
bb8:
  ret void
bb9:
  %v30 = extractvalue { ptr, i64 } %v13, 1
  %v31 = icmp ult i64 %v16, %v30
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb11, label %bb10
bb10:
  %v33 = extractvalue { ptr, i64 } %v13, 0
  %v34 = getelementptr inbounds float, ptr %v33, i64 %v16
  %v35 = insertvalue { ptr } undef, ptr %v34, 0
  %v36 = extractvalue { ptr } %v35, 0
  br label %bb13
bb11:
  br label %bb12
bb12:
  %v37 = inttoptr i64 0 to ptr
  %v38 = insertvalue { ptr } undef, ptr %v37, 0
  %v39 = extractvalue { ptr } %v38, 0
  br label %bb13
bb13:
  %v40 = phi ptr [ %v36, %bb10 ], [ %v39, %bb12 ]
  %v41 = insertvalue { ptr } undef, ptr %v40, 0
  %v42 = extractvalue { ptr } %v41, 0
  %v43 = ptrtoint ptr %v42 to i64
  %v44 = sub i64 %v43, 0
  %v45 = icmp ule i64 %v44, 0
  %v46 = add i64 %v44, 0
  %v47 = select i1 %v45, i64 %v46, i64 1
  %v48 = icmp eq i64 %v47, 1
  br i1 %v48, label %bb4, label %bb14
bb14:
  %v49 = icmp eq i64 %v47, 0
  br i1 %v49, label %bb6, label %bb15
bb15:
  unreachable
bb16:
  call void @llvm.trap() #0
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.usub.sat.i32(i32, i32)

define ptx_kernel void @attention_fa2_prefill_paged(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, float %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, i32 %v16, i32 %v17, i32 %v18, i32 %v19, ptr %v20, i64 %v21) #0 {
entry:
  %v22 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v1, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v3, 1
  %v26 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v27 = insertvalue { ptr, i64 } %v26, i64 %v5, 1
  %v28 = insertvalue { ptr, i64 } undef, ptr %v20, 0
  %v29 = insertvalue { ptr, i64 } %v28, i64 %v21, 1
  br label %bb0
bb0:
  %v30 = phi { ptr, i64 } [ %v23, %entry ]
  %v31 = phi { ptr, i64 } [ %v25, %entry ]
  %v32 = phi { ptr, i64 } [ %v27, %entry ]
  %v33 = phi i32 [ %v6, %entry ]
  %v34 = phi i32 [ %v7, %entry ]
  %v35 = phi i32 [ %v8, %entry ]
  %v36 = phi i32 [ %v9, %entry ]
  %v37 = phi i32 [ %v10, %entry ]
  %v38 = phi float [ %v11, %entry ]
  %v39 = phi i32 [ %v12, %entry ]
  %v40 = phi i32 [ %v13, %entry ]
  %v41 = phi i32 [ %v14, %entry ]
  %v42 = phi i32 [ %v15, %entry ]
  %v43 = phi i32 [ %v16, %entry ]
  %v44 = phi i32 [ %v17, %entry ]
  %v45 = phi i32 [ %v18, %entry ]
  %v46 = phi i32 [ %v19, %entry ]
  %v47 = phi { ptr, i64 } [ %v29, %entry ]
  %v48 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb1
bb1:
  %v49 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v50 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v51 = icmp uge i32 %v48, %v33
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb5, label %bb4
bb4:
  br label %bb96
bb5:
  %v53 = zext i32 %v35 to i64
  %v54 = icmp eq i64 %v53, 0
  br i1 %v54, label %bb8, label %bb6
bb6:
  %v55 = icmp ugt i64 %v53, 128
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb7, label %bb8
bb7:
  %v57 = mul i32 %v49, 4
  %v58 = icmp uge i32 %v57, %v36
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb10, label %bb9
bb8:
  br label %bb96
bb9:
  br label %bb96
bb10:
  %v60 = add i32 %v57, 4
  %v61 = icmp ugt i32 %v60, %v36
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb12, label %bb11
bb11:
  br label %bb13
bb12:
  br label %bb13
bb13:
  %v63 = phi i32 [ %v36, %bb11 ], [ %v60, %bb12 ]
  %v64 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v34, i32 1) #0
  br label %bb14
bb14:
  %v65 = icmp eq i32 %v64, 0
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb15, label %bb102
bb15:
  %v67 = udiv i32 %v33, %v64
  %v68 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v67, i32 1) #0
  br label %bb16
bb16:
  %v69 = icmp eq i32 %v68, 0
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb17, label %bb103
bb17:
  %v71 = udiv i32 %v48, %v68
  %v72 = zext i32 %v71 to i64
  %v73 = zext i32 %v41 to i64
  %v74 = zext i32 %v42 to i64
  %v75 = zext i32 %v43 to i64
  %v76 = mul i64 %v75, 2
  %v77 = mul i64 %v73, %v76
  %v78 = udiv i32 %v50, 32
  %v79 = urem i32 %v50, 32
  %v80 = zext i32 %v79 to i64
  %v81 = zext i32 %v78 to i64
  %v82 = sub i32 %v63, %v57
  %v83 = zext i32 %v82 to i64
  %v84 = icmp ult i64 %v81, %v83
  %v85 = xor i1 %v84, 1
  br i1 %v85, label %bb19, label %bb18
bb18:
  %v86 = zext i32 %v57 to i64
  %v87 = add i64 %v86, %v81
  %v88 = zext i32 %v48 to i64
  %v89 = zext i32 %v36 to i64
  %v90 = mul i64 %v88, %v89
  %v91 = add i64 %v90, %v87
  %v92 = mul i64 %v91, %v53
  br label %bb19
bb19:
  %v93 = phi i64 [ 0, %bb17 ], [ %v92, %bb18 ]
  %v94 = phi i64 [ 0, %bb17 ], [ %v87, %bb18 ]
  br label %bb20
bb20:
  %v95 = phi float [ 0.0, %bb19 ], [ %v301, %bb78 ]
  %v96 = phi float [ 0.0, %bb19 ], [ %v302, %bb78 ]
  %v97 = phi float [ 0.0, %bb19 ], [ %v303, %bb78 ]
  %v98 = phi float [ 0.0, %bb19 ], [ %v304, %bb78 ]
  %v99 = phi float [ 0.0, %bb19 ], [ %v305, %bb78 ]
  %v100 = phi float [ 0.0, %bb19 ], [ %v306, %bb78 ]
  %v101 = phi i1 [ 0, %bb19 ], [ %v307, %bb78 ]
  %v102 = phi i32 [ 0, %bb19 ], [ %v109, %bb78 ]
  %v103 = icmp ult i32 %v102, %v37
  %v104 = xor i1 %v103, 1
  br i1 %v104, label %bb79, label %bb21
bb21:
  %v105 = add i32 %v102, 16
  %v106 = icmp ugt i32 %v105, %v37
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb23, label %bb22
bb22:
  br label %bb24
bb23:
  %v108 = add i32 %v102, 16
  br label %bb24
bb24:
  %v109 = phi i32 [ %v37, %bb22 ], [ %v108, %bb23 ]
  %v110 = sub i32 %v109, %v102
  %v111 = zext i32 %v110 to i64
  %v112 = zext i32 %v50 to i64
  br label %bb25
bb25:
  %v113 = phi i64 [ %v112, %bb24 ], [ %v184, %bb31 ]
  %v114 = mul i64 %v111, %v53
  %v115 = icmp ult i64 %v113, %v114
  %v116 = xor i1 %v115, 1
  br i1 %v116, label %bb32, label %bb26
bb26:
  %v117 = udiv i64 %v113, %v53
  %v118 = urem i64 %v113, %v53
  %v119 = zext i32 %v102 to i64
  %v120 = add i64 %v119, %v117
  %v121 = icmp eq i64 %v73, 0
  %v122 = xor i1 %v121, 1
  br i1 %v122, label %bb27, label %bb104
bb27:
  %v123 = udiv i64 %v120, %v73
  %v124 = urem i64 %v120, %v73
  %v125 = zext i32 %v39 to i64
  %v126 = zext i32 %v40 to i64
  %v127 = mul i64 %v125, %v126
  %v128 = add i64 %v127, %v123
  %v129 = extractvalue { ptr, i64 } %v32, 1
  %v130 = icmp ult i64 %v128, %v129
  %v131 = extractvalue { ptr, i64 } %v32, 0
  %v132 = getelementptr inbounds i32, ptr %v131, i64 %v128
  %v133 = load i32, ptr %v132, align 4
  %v134 = zext i32 %v133 to i64
  %v135 = mul i64 %v134, %v74
  %v136 = mul i64 %v124, %v76
  %v137 = add i64 %v135, %v136
  %v138 = mul i64 %v72, %v53
  %v139 = mul i64 %v138, 2
  %v140 = add i64 %v137, %v139
  %v141 = add i64 %v135, %v77
  %v142 = add i64 %v141, %v136
  %v143 = add i64 %v142, %v139
  %v144 = mul i64 %v118, 2
  %v145 = add i64 %v140, %v144
  %v146 = extractvalue { ptr, i64 } %v31, 1
  %v147 = icmp ult i64 %v145, %v146
  %v148 = extractvalue { ptr, i64 } %v31, 0
  %v149 = getelementptr inbounds i8, ptr %v148, i64 %v145
  %v150 = load i8, ptr %v149, align 1
  %v151 = zext i8 %v150 to i16
  %v152 = add i64 %v145, 1
  %v153 = icmp ult i64 %v152, %v146
  %v154 = extractvalue { ptr, i64 } %v31, 0
  %v155 = getelementptr inbounds i8, ptr %v154, i64 %v152
  %v156 = load i8, ptr %v155, align 1
  %v157 = zext i8 %v156 to i16
  %v158 = trunc i32 8 to i16
  %v159 = and i16 %v158, 15
  %v160 = shl i16 %v157, %v159
  %v161 = or i16 %v151, %v160
  %v162 = add i64 %v143, %v144
  %v163 = icmp ult i64 %v162, %v146
  %v164 = extractvalue { ptr, i64 } %v31, 0
  %v165 = getelementptr inbounds i8, ptr %v164, i64 %v162
  %v166 = load i8, ptr %v165, align 1
  %v167 = zext i8 %v166 to i16
  %v168 = add i64 %v162, 1
  %v169 = icmp ult i64 %v168, %v146
  %v170 = extractvalue { ptr, i64 } %v31, 0
  %v171 = getelementptr inbounds i8, ptr %v170, i64 %v168
  %v172 = load i8, ptr %v171, align 1
  %v173 = zext i8 %v172 to i16
  %v174 = trunc i32 8 to i16
  %v175 = and i16 %v174, 15
  %v176 = shl i16 %v173, %v175
  %v177 = or i16 %v167, %v176
  %v178 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v161) #0
  br label %bb28
bb28:
  %v179 = mul i64 %v117, 128
  %v180 = add i64 %v179, %v118
  %v181 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_20, i64 %v180
  br label %bb29
bb29:
  store float %v178, ptr addrspace(3) %v181, align 4
  %v182 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v177) #0
  br label %bb30
bb30:
  %v183 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_21, i64 %v180
  br label %bb31
bb31:
  store float %v182, ptr addrspace(3) %v183, align 4
  %v184 = add i64 %v113, 128
  br label %bb25
bb32:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb33
bb33:
  %v186 = xor i1 %v84, 1
  br i1 %v186, label %bb77, label %bb34
bb34:
  br label %bb35
bb35:
  %v187 = phi float [ %v95, %bb34 ], [ %v293, %bb75 ]
  %v188 = phi float [ %v96, %bb34 ], [ %v294, %bb75 ]
  %v189 = phi float [ %v97, %bb34 ], [ %v295, %bb75 ]
  %v190 = phi float [ %v98, %bb34 ], [ %v296, %bb75 ]
  %v191 = phi float [ %v99, %bb34 ], [ %v297, %bb75 ]
  %v192 = phi float [ %v100, %bb34 ], [ %v298, %bb75 ]
  %v193 = phi i1 [ %v101, %bb34 ], [ %v299, %bb75 ]
  %v194 = phi i64 [ 0, %bb34 ], [ %v300, %bb75 ]
  %v195 = icmp ult i64 %v194, %v111
  %v196 = xor i1 %v195, 1
  br i1 %v196, label %bb76, label %bb36
bb36:
  %v197 = zext i32 %v102 to i64
  %v198 = add i64 %v197, %v194
  %v199 = icmp eq i32 %v44, 0
  br i1 %v199, label %bb45, label %bb37
bb37:
  %v200 = sub i32 %v37, %v36
  %v201 = trunc i64 %v94 to i32
  %v202 = add i32 %v200, %v201
  %v203 = trunc i64 %v198 to i32
  %v204 = icmp ugt i32 %v203, %v202
  %v205 = xor i1 %v204, 1
  br i1 %v205, label %bb39, label %bb38
bb38:
  br label %bb44
bb39:
  %v206 = icmp eq i32 %v45, 0
  br i1 %v206, label %bb43, label %bb40
bb40:
  %v207 = call i32 @llvm.usub.sat.i32(i32 %v202, i32 %v203) #0
  br label %bb97
bb41:
  br label %bb43
bb42:
  br label %bb43
bb43:
  %v208 = phi i1 [ 1, %bb39 ], [ 0, %bb41 ], [ 1, %bb42 ]
  br label %bb44
bb44:
  %v209 = phi i1 [ 0, %bb38 ], [ %v208, %bb43 ]
  br label %bb45
bb45:
  %v210 = phi i1 [ 1, %bb36 ], [ %v209, %bb44 ]
  %v211 = xor i1 %v210, 1
  br i1 %v211, label %bb75, label %bb46
bb46:
  br label %bb47
bb47:
  %v212 = phi float [ 0.0, %bb46 ], [ %v228, %bb49 ]
  %v213 = phi i64 [ %v80, %bb46 ], [ %v229, %bb49 ]
  %v214 = icmp ult i64 %v213, %v53
  %v215 = xor i1 %v214, 1
  br i1 %v215, label %bb50, label %bb48
bb48:
  %v216 = bitcast ptr addrspace(3) @__shared_mem_20 to ptr addrspace(3)
  %v217 = mul i64 %v194, 128
  %v218 = add i64 %v217, %v213
  %v219 = getelementptr inbounds float, ptr addrspace(3) %v216, i64 %v218
  br label %bb49
bb49:
  %v220 = load float, ptr addrspace(3) %v219, align 4
  %v221 = add i64 %v93, %v213
  %v222 = extractvalue { ptr, i64 } %v30, 1
  %v223 = icmp ult i64 %v221, %v222
  %v224 = extractvalue { ptr, i64 } %v30, 0
  %v225 = getelementptr inbounds float, ptr %v224, i64 %v221
  %v226 = load float, ptr %v225, align 4
  %v227 = fmul contract float %v226, %v220
  %v228 = fadd contract float %v212, %v227
  %v229 = add i64 %v213, 32
  br label %bb47
bb50:
  br label %bb51
bb51:
  %v230 = phi float [ %v212, %bb50 ], [ %v346, %bb98 ]
  %v231 = phi i32 [ 16, %bb50 ], [ %v347, %bb98 ]
  %v232 = icmp eq i32 %v231, 0
  br i1 %v232, label %bb53, label %bb52
bb52:
  %v233 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v230, i32 %v231, i32 31) #0
  br label %bb98
bb53:
  %v234 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 4294967295, float %v230, i32 0, i32 31) #0
  br label %bb99
bb54:
  br label %bb59
bb55:
  %v235 = fcmp ogt float %v348, %v191
  %v236 = xor i1 %v235, 1
  br i1 %v236, label %bb57, label %bb56
bb56:
  %v237 = fsub contract float %v191, %v348
  %v238 = call float @__nv_expf(float %v237) #0
  br label %bb100
bb57:
  br label %bb58
bb58:
  %v239 = phi float [ %v191, %bb57 ], [ %v348, %bb100 ]
  %v240 = phi float [ 1.0, %bb57 ], [ %v238, %bb100 ]
  br label %bb59
bb59:
  %v241 = phi float [ %v348, %bb54 ], [ %v239, %bb58 ]
  %v242 = phi float [ 0.0, %bb54 ], [ %v240, %bb58 ]
  %v243 = fsub contract float %v348, %v241
  %v244 = call float @__nv_expf(float %v243) #0
  br label %bb101
bb60:
  %v245 = bitcast ptr addrspace(3) @__shared_mem_21 to ptr addrspace(3)
  %v246 = mul i64 %v194, 128
  %v247 = add i64 %v246, %v80
  %v248 = getelementptr inbounds float, ptr addrspace(3) %v245, i64 %v247
  br label %bb61
bb61:
  %v249 = load float, ptr addrspace(3) %v248, align 4
  %v250 = fmul contract float %v187, %v242
  %v251 = fmul contract float %v244, %v249
  %v252 = fadd contract float %v250, %v251
  br label %bb62
bb62:
  %v253 = phi float [ %v252, %bb61 ], [ %v187, %bb101 ]
  %v254 = add i64 %v80, 32
  %v255 = icmp ult i64 %v254, %v53
  %v256 = xor i1 %v255, 1
  br i1 %v256, label %bb65, label %bb63
bb63:
  %v257 = bitcast ptr addrspace(3) @__shared_mem_21 to ptr addrspace(3)
  %v258 = mul i64 %v194, 128
  %v259 = add i64 %v258, %v80
  %v260 = add i64 %v259, 32
  %v261 = getelementptr inbounds float, ptr addrspace(3) %v257, i64 %v260
  br label %bb64
bb64:
  %v262 = load float, ptr addrspace(3) %v261, align 4
  %v263 = fmul contract float %v188, %v242
  %v264 = fmul contract float %v244, %v262
  %v265 = fadd contract float %v263, %v264
  br label %bb66
bb65:
  br label %bb66
bb66:
  %v266 = phi float [ %v265, %bb64 ], [ %v188, %bb65 ]
  %v267 = add i64 %v80, 64
  %v268 = icmp ult i64 %v267, %v53
  %v269 = xor i1 %v268, 1
  br i1 %v269, label %bb69, label %bb67
bb67:
  %v270 = bitcast ptr addrspace(3) @__shared_mem_21 to ptr addrspace(3)
  %v271 = mul i64 %v194, 128
  %v272 = add i64 %v271, %v80
  %v273 = add i64 %v272, 64
  %v274 = getelementptr inbounds float, ptr addrspace(3) %v270, i64 %v273
  br label %bb68
bb68:
  %v275 = load float, ptr addrspace(3) %v274, align 4
  %v276 = fmul contract float %v189, %v242
  %v277 = fmul contract float %v244, %v275
  %v278 = fadd contract float %v276, %v277
  br label %bb70
bb69:
  br label %bb70
bb70:
  %v279 = phi float [ %v278, %bb68 ], [ %v189, %bb69 ]
  %v280 = add i64 %v80, 96
  %v281 = icmp ult i64 %v280, %v53
  %v282 = xor i1 %v281, 1
  br i1 %v282, label %bb73, label %bb71
bb71:
  %v283 = bitcast ptr addrspace(3) @__shared_mem_21 to ptr addrspace(3)
  %v284 = mul i64 %v194, 128
  %v285 = add i64 %v284, %v80
  %v286 = add i64 %v285, 96
  %v287 = getelementptr inbounds float, ptr addrspace(3) %v283, i64 %v286
  br label %bb72
bb72:
  %v288 = load float, ptr addrspace(3) %v287, align 4
  %v289 = fmul contract float %v190, %v242
  %v290 = fmul contract float %v244, %v288
  %v291 = fadd contract float %v289, %v290
  br label %bb74
bb73:
  br label %bb74
bb74:
  %v292 = phi float [ %v291, %bb72 ], [ %v190, %bb73 ]
  br label %bb75
bb75:
  %v293 = phi float [ %v187, %bb45 ], [ %v253, %bb74 ]
  %v294 = phi float [ %v188, %bb45 ], [ %v266, %bb74 ]
  %v295 = phi float [ %v189, %bb45 ], [ %v279, %bb74 ]
  %v296 = phi float [ %v190, %bb45 ], [ %v292, %bb74 ]
  %v297 = phi float [ %v191, %bb45 ], [ %v241, %bb74 ]
  %v298 = phi float [ %v192, %bb45 ], [ %v351, %bb74 ]
  %v299 = phi i1 [ %v193, %bb45 ], [ 1, %bb74 ]
  %v300 = add i64 %v194, 1
  br label %bb35
bb76:
  br label %bb77
bb77:
  %v301 = phi float [ %v95, %bb33 ], [ %v187, %bb76 ]
  %v302 = phi float [ %v96, %bb33 ], [ %v188, %bb76 ]
  %v303 = phi float [ %v97, %bb33 ], [ %v189, %bb76 ]
  %v304 = phi float [ %v98, %bb33 ], [ %v190, %bb76 ]
  %v305 = phi float [ %v99, %bb33 ], [ %v191, %bb76 ]
  %v306 = phi float [ %v100, %bb33 ], [ %v192, %bb76 ]
  %v307 = phi i1 [ %v101, %bb33 ], [ %v193, %bb76 ]
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb78
bb78:
  br label %bb20
bb79:
  %v309 = xor i1 %v84, 1
  br i1 %v309, label %bb95, label %bb80
bb80:
  %v310 = xor i1 %v101, 1
  br i1 %v310, label %bb95, label %bb81
bb81:
  %v311 = fcmp ogt float %v100, 0.0
  %v312 = xor i1 %v311, 1
  br i1 %v312, label %bb94, label %bb82
bb82:
  %v313 = fdiv contract float 1.0, %v100
  %v314 = icmp ult i64 %v80, %v53
  %v315 = xor i1 %v314, 1
  br i1 %v315, label %bb84, label %bb83
bb83:
  %v316 = add i64 %v93, %v80
  %v317 = extractvalue { ptr, i64 } %v47, 0
  %v318 = getelementptr inbounds float, ptr %v317, i64 %v316
  %v319 = fmul contract float %v95, %v313
  store float %v319, ptr %v318, align 4
  br label %bb84
bb84:
  %v320 = add i64 %v80, 32
  %v321 = icmp ult i64 %v320, %v53
  %v322 = xor i1 %v321, 1
  br i1 %v322, label %bb86, label %bb85
bb85:
  %v323 = add i64 %v93, %v80
  %v324 = add i64 %v323, 32
  %v325 = extractvalue { ptr, i64 } %v47, 0
  %v326 = getelementptr inbounds float, ptr %v325, i64 %v324
  %v327 = fmul contract float %v96, %v313
  store float %v327, ptr %v326, align 4
  br label %bb87
bb86:
  br label %bb87
bb87:
  %v328 = add i64 %v80, 64
  %v329 = icmp ult i64 %v328, %v53
  %v330 = xor i1 %v329, 1
  br i1 %v330, label %bb89, label %bb88
bb88:
  %v331 = add i64 %v93, %v80
  %v332 = add i64 %v331, 64
  %v333 = extractvalue { ptr, i64 } %v47, 0
  %v334 = getelementptr inbounds float, ptr %v333, i64 %v332
  %v335 = fmul contract float %v97, %v313
  store float %v335, ptr %v334, align 4
  br label %bb90
bb89:
  br label %bb90
bb90:
  %v336 = add i64 %v80, 96
  %v337 = icmp ult i64 %v336, %v53
  %v338 = xor i1 %v337, 1
  br i1 %v338, label %bb92, label %bb91
bb91:
  %v339 = add i64 %v93, %v80
  %v340 = add i64 %v339, 96
  %v341 = extractvalue { ptr, i64 } %v47, 0
  %v342 = getelementptr inbounds float, ptr %v341, i64 %v340
  %v343 = fmul contract float %v98, %v313
  store float %v343, ptr %v342, align 4
  br label %bb93
bb92:
  br label %bb93
bb93:
  br label %bb95
bb94:
  br label %bb95
bb95:
  br label %bb96
bb96:
  ret void
bb97:
  %v344 = icmp uge i32 %v207, %v45
  %v345 = xor i1 %v344, 1
  br i1 %v345, label %bb42, label %bb41
bb98:
  %v346 = fadd contract float %v230, %v233
  %v347 = udiv i32 %v231, 2
  br label %bb51
bb99:
  %v348 = fmul contract float %v234, %v38
  %v349 = xor i1 %v193, 1
  br i1 %v349, label %bb54, label %bb55
bb100:
  br label %bb58
bb101:
  %v350 = fmul contract float %v192, %v242
  %v351 = fadd contract float %v350, %v244
  %v352 = icmp ult i64 %v80, %v53
  %v353 = xor i1 %v352, 1
  br i1 %v353, label %bb62, label %bb60
bb102:
  call void @llvm.trap() #0
  unreachable
bb103:
  call void @llvm.trap() #0
  unreachable
bb104:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_count_assignments(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = alloca {  }, align 1
  %v11 = bitcast ptr %v10 to ptr
  %v12 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v11) #0
  br label %bb1
bb1:
  %v13 = extractvalue { ptr, i64 } %v8, 1
  %v14 = icmp ult i64 %v12, %v13
  %v15 = xor i1 %v14, 1
  br i1 %v15, label %bb5, label %bb2
bb2:
  %v16 = extractvalue { ptr, i64 } %v8, 0
  %v17 = getelementptr inbounds i32, ptr %v16, i64 %v12
  %v18 = load i32, ptr %v17, align 4
  %v19 = zext i32 %v18 to i64
  %v20 = extractvalue { ptr, i64 } %v9, 1
  %v21 = icmp ult i64 %v19, %v20
  br i1 %v21, label %bb3, label %bb6
bb3:
  %v22 = extractvalue { ptr, i64 } %v9, 0
  %v23 = getelementptr inbounds { { i32 } }, ptr %v22, i64 %v19
  %v24 = atomicrmw add ptr %v23, i32 1 syncscope("device") monotonic
  br label %bb4
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @q4k_gate_up_swiglu_multiwarp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, ptr %v10, i64 %v11) #0 {
entry:
  %v12 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v1, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v3, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v5, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v7, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v11, 1
  br label %bb0
bb0:
  %v22 = phi { ptr, i64 } [ %v13, %entry ]
  %v23 = phi { ptr, i64 } [ %v15, %entry ]
  %v24 = phi { ptr, i64 } [ %v17, %entry ]
  %v25 = phi { ptr, i64 } [ %v19, %entry ]
  %v26 = phi i32 [ %v8, %entry ]
  %v27 = phi i32 [ %v9, %entry ]
  %v28 = phi { ptr, i64 } [ %v21, %entry ]
  %v29 = alloca { [0 x i32], { { i32 } } }, align 4
  %v30 = alloca { [0 x i32], { { i32 } } }, align 4
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v32 = zext i32 %v31 to i64
  %v33 = and i64 %v32, 31
  %v34 = zext i32 5 to i64
  %v35 = and i64 %v34, 63
  %v36 = lshr i64 %v32, %v35
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v38 = zext i32 %v37 to i64
  %v39 = zext i32 %v26 to i64
  %v40 = icmp uge i64 %v38, %v39
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb4, label %bb3
bb3:
  br label %bb5
bb4:
  %v42 = icmp uge i64 %v36, 4
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb6, label %bb5
bb5:
  br label %bb38
bb6:
  %v44 = zext i32 %v27 to i64
  %v45 = mul i64 %v44, 144
  br label %bb7
bb7:
  %v46 = phi float [ 0.0, %bb6 ], [ %v54, %bb13 ]
  %v47 = phi float [ 0.0, %bb6 ], [ %v55, %bb13 ]
  %v48 = phi i64 [ %v36, %bb6 ], [ %v109, %bb13 ]
  %v49 = icmp ult i64 %v48, %v44
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb14, label %bb8
bb8:
  %v51 = mul i64 %v38, %v45
  %v52 = mul i64 %v48, 144
  %v53 = add i64 %v51, %v52
  br label %bb9
bb9:
  %v54 = phi float [ %v46, %bb8 ], [ %v103, %bb12 ]
  %v55 = phi float [ %v47, %bb8 ], [ %v107, %bb12 ]
  %v56 = phi i64 [ 0, %bb8 ], [ %v108, %bb12 ]
  %v57 = icmp ult i64 %v56, 2
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb13, label %bb10
bb10:
  %v59 = mul i64 %v56, 128
  %v60 = mul i64 %v33, 4
  %v61 = add i64 %v59, %v60
  %v62 = mul i64 %v48, 256
  %v63 = add i64 %v62, %v61
  %v64 = extractvalue { ptr, i64 } %v24, 0
  %v65 = getelementptr inbounds i8, ptr %v64, i64 %v63
  store { [0 x i32], { { i32 } } } undef, ptr %v29, align 4
  %v66 = bitcast ptr %v65 to ptr
  %v67 = bitcast ptr %v29 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %v67, ptr %v66, i64 4, i1 0) #0
  %v69 = load { [0 x i32], { { i32 } } }, ptr %v29, align 4
  store { [0 x i32], { { i32 } } } %v69, ptr %v30, align 4
  %v70 = getelementptr inbounds i8, ptr %v30, i32 0
  %v71 = bitcast ptr %v70 to ptr
  %v72 = load i32, ptr %v71, align 4
  %v73 = trunc i32 %v72 to i8
  %v74 = bitcast i8 %v73 to i8
  %v75 = and i32 8, 31
  %v76 = lshr i32 %v72, %v75
  %v77 = trunc i32 %v76 to i8
  %v78 = bitcast i8 %v77 to i8
  %v79 = and i32 16, 31
  %v80 = lshr i32 %v72, %v79
  %v81 = trunc i32 %v80 to i8
  %v82 = bitcast i8 %v81 to i8
  %v83 = and i32 24, 31
  %v84 = lshr i32 %v72, %v83
  %v85 = trunc i32 %v84 to i8
  %v86 = bitcast i8 %v85 to i8
  %v87 = sext i8 %v74 to i32
  %v88 = sext i8 %v78 to i32
  %v89 = add i32 %v87, %v88
  %v90 = sext i8 %v82 to i32
  %v91 = add i32 %v89, %v90
  %v92 = sext i8 %v86 to i32
  %v93 = add i32 %v91, %v92
  %v94 = udiv i64 %v63, 32
  %v95 = extractvalue { ptr, i64 } %v25, 1
  %v96 = icmp ult i64 %v94, %v95
  %v97 = extractvalue { ptr, i64 } %v25, 0
  %v98 = getelementptr inbounds float, ptr %v97, i64 %v94
  %v99 = load float, ptr %v98, align 4
  %v100 = extractvalue { ptr, i64 } %v22, 0
  %v101 = extractvalue { ptr, i64 } %v22, 1
  %v102 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v100, i64 %v101, i64 %v53, i64 %v61, i32 %v72, i32 %v93, float %v99) #0
  br label %bb11
bb11:
  %v103 = fadd contract float %v54, %v102
  %v104 = extractvalue { ptr, i64 } %v23, 0
  %v105 = extractvalue { ptr, i64 } %v23, 1
  %v106 = call float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v104, i64 %v105, i64 %v53, i64 %v61, i32 %v72, i32 %v93, float %v99) #0
  br label %bb12
bb12:
  %v107 = fadd contract float %v55, %v106
  %v108 = add i64 %v56, 1
  br label %bb9
bb13:
  %v109 = add i64 %v48, 4
  br label %bb7
bb14:
  br label %bb15
bb15:
  %v110 = phi float [ %v46, %bb14 ], [ %v142, %bb40 ]
  %v111 = phi float [ %v47, %bb14 ], [ %v144, %bb40 ]
  %v112 = phi i32 [ 16, %bb14 ], [ %v145, %bb40 ]
  %v113 = icmp ugt i32 %v112, 0
  %v114 = xor i1 %v113, 1
  br i1 %v114, label %bb17, label %bb16
bb16:
  %v115 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v110, i32 %v112, i32 31) #0
  br label %bb39
bb17:
  %v116 = icmp eq i64 %v33, 0
  %v117 = icmp eq i64 %v33, 0
  br i1 %v117, label %bb18, label %bb21
bb18:
  %v118 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_22, i64 %v36
  br label %bb19
bb19:
  store float %v110, ptr addrspace(3) %v118, align 4
  %v119 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_23, i64 %v36
  br label %bb20
bb20:
  store float %v111, ptr addrspace(3) %v119, align 4
  br label %bb21
bb21:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0
  br label %bb22
bb22:
  %v121 = icmp eq i64 %v36, 0
  br i1 %v121, label %bb23, label %bb37
bb23:
  %v122 = icmp ult i64 %v33, 4
  %v123 = xor i1 %v122, 1
  br i1 %v123, label %bb26, label %bb24
bb24:
  %v124 = bitcast ptr addrspace(3) @__shared_mem_22 to ptr addrspace(3)
  %v125 = getelementptr inbounds float, ptr addrspace(3) %v124, i64 %v33
  br label %bb25
bb25:
  %v126 = load float, ptr addrspace(3) %v125, align 4
  br label %bb27
bb26:
  br label %bb27
bb27:
  %v127 = phi float [ %v126, %bb25 ], [ 0.0, %bb26 ]
  %v128 = xor i1 %v122, 1
  br i1 %v128, label %bb30, label %bb28
bb28:
  %v129 = bitcast ptr addrspace(3) @__shared_mem_23 to ptr addrspace(3)
  %v130 = getelementptr inbounds float, ptr addrspace(3) %v129, i64 %v33
  br label %bb29
bb29:
  %v131 = load float, ptr addrspace(3) %v130, align 4
  br label %bb31
bb30:
  br label %bb31
bb31:
  %v132 = phi float [ %v131, %bb29 ], [ 0.0, %bb30 ]
  br label %bb32
bb32:
  %v133 = phi float [ %v127, %bb31 ], [ %v146, %bb42 ]
  %v134 = phi float [ %v132, %bb31 ], [ %v148, %bb42 ]
  %v135 = phi i32 [ 16, %bb31 ], [ %v149, %bb42 ]
  %v136 = icmp ugt i32 %v135, 0
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb34, label %bb33
bb33:
  %v138 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v133, i32 %v135, i32 31) #0
  br label %bb41
bb34:
  %v139 = xor i1 %v116, 1
  br i1 %v139, label %bb36, label %bb35
bb35:
  %v140 = fneg float %v133
  %v141 = call float @__nv_expf(float %v140) #0
  br label %bb43
bb36:
  br label %bb37
bb37:
  br label %bb38
bb38:
  ret void
bb39:
  %v142 = fadd contract float %v110, %v115
  %v143 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v111, i32 %v112, i32 31) #0
  br label %bb40
bb40:
  %v144 = fadd contract float %v111, %v143
  %v145 = udiv i32 %v112, 2
  br label %bb15
bb41:
  %v146 = fadd contract float %v133, %v138
  %v147 = call float @llvm.nvvm.shfl.sync.down.f32(i32 4294967295, float %v134, i32 %v135, i32 31) #0
  br label %bb42
bb42:
  %v148 = fadd contract float %v134, %v147
  %v149 = udiv i32 %v135, 2
  br label %bb32
bb43:
  %v150 = fadd contract float 1.0, %v141
  %v151 = fdiv contract float %v133, %v150
  %v152 = extractvalue { ptr, i64 } %v28, 0
  %v153 = getelementptr inbounds float, ptr %v152, i64 %v38
  %v154 = fmul contract float %v151, %v134
  store float %v154, ptr %v153, align 4
  br label %bb36
}

define ptx_kernel void @embedding_q4k_rows(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = alloca [2 x i8], align 1
  %v23 = alloca [2 x i8], align 1
  %v24 = alloca [8 x i8], align 1
  %v25 = alloca [8 x i8], align 1
  %v26 = bitcast ptr %v21 to ptr
  %v27 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v26) #0
  br label %bb1
bb1:
  %v28 = zext i32 %v17 to i64
  %v29 = zext i32 %v18 to i64
  %v30 = mul i64 %v28, %v29
  %v31 = icmp uge i64 %v27, %v30
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb3, label %bb2
bb2:
  br label %bb34
bb3:
  %v33 = icmp eq i64 %v29, 0
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb4, label %bb42
bb4:
  %v35 = udiv i64 %v27, %v29
  %v36 = urem i64 %v27, %v29
  %v37 = zext i32 %v19 to i64
  %v38 = mul i64 %v37, 144
  %v39 = udiv i64 %v36, 256
  %v40 = urem i64 %v36, 256
  %v41 = extractvalue { ptr, i64 } %v16, 1
  %v42 = icmp ult i64 %v35, %v41
  br i1 %v42, label %bb5, label %bb43
bb5:
  %v43 = extractvalue { ptr, i64 } %v16, 0
  %v44 = getelementptr inbounds i32, ptr %v43, i64 %v35
  %v45 = load i32, ptr %v44, align 4
  %v46 = zext i32 %v45 to i64
  %v47 = mul i64 %v46, %v38
  %v48 = mul i64 %v39, 144
  %v49 = add i64 %v47, %v48
  %v50 = extractvalue { ptr, i64 } %v15, 1
  %v51 = icmp ult i64 %v49, %v50
  br i1 %v51, label %bb6, label %bb44
bb6:
  %v52 = extractvalue { ptr, i64 } %v15, 0
  %v53 = getelementptr inbounds i8, ptr %v52, i64 %v49
  %v54 = load i8, ptr %v53, align 1
  %v55 = add i64 %v49, 1
  %v56 = icmp ult i64 %v55, %v50
  br i1 %v56, label %bb7, label %bb45
bb7:
  %v57 = extractvalue { ptr, i64 } %v15, 0
  %v58 = getelementptr inbounds i8, ptr %v57, i64 %v55
  %v59 = load i8, ptr %v58, align 1
  %v60 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 0
  store i8 %v54, ptr %v60, align 1
  %v61 = getelementptr inbounds [2 x i8], ptr %v22, i32 0, i64 1
  store i8 %v59, ptr %v61, align 1
  %v62 = load [2 x i8], ptr %v22, align 1
  %v63 = alloca [2 x i8], align 2
  store [2 x i8] %v62, ptr %v63, align 2
  %v64 = load i16, ptr %v63, align 2
  %v65 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v64) #0
  br label %bb8
bb8:
  %v66 = add i64 %v49, 2
  %v67 = icmp ult i64 %v66, %v50
  br i1 %v67, label %bb9, label %bb46
bb9:
  %v68 = extractvalue { ptr, i64 } %v15, 0
  %v69 = getelementptr inbounds i8, ptr %v68, i64 %v66
  %v70 = load i8, ptr %v69, align 1
  %v71 = add i64 %v49, 3
  %v72 = icmp ult i64 %v71, %v50
  br i1 %v72, label %bb10, label %bb47
bb10:
  %v73 = extractvalue { ptr, i64 } %v15, 0
  %v74 = getelementptr inbounds i8, ptr %v73, i64 %v71
  %v75 = load i8, ptr %v74, align 1
  %v76 = getelementptr inbounds [2 x i8], ptr %v23, i32 0, i64 0
  store i8 %v70, ptr %v76, align 1
  %v77 = getelementptr inbounds [2 x i8], ptr %v23, i32 0, i64 1
  store i8 %v75, ptr %v77, align 1
  %v78 = load [2 x i8], ptr %v23, align 1
  %v79 = alloca [2 x i8], align 2
  store [2 x i8] %v78, ptr %v79, align 2
  %v80 = load i16, ptr %v79, align 2
  %v81 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v80) #0
  br label %bb11
bb11:
  %v82 = add i64 %v49, 4
  %v83 = icmp ult i64 %v82, %v50
  br i1 %v83, label %bb12, label %bb48
bb12:
  %v84 = extractvalue { ptr, i64 } %v15, 0
  %v85 = getelementptr inbounds i8, ptr %v84, i64 %v82
  %v86 = load i8, ptr %v85, align 1
  %v87 = add i64 %v49, 5
  %v88 = icmp ult i64 %v87, %v50
  br i1 %v88, label %bb13, label %bb49
bb13:
  %v89 = extractvalue { ptr, i64 } %v15, 0
  %v90 = getelementptr inbounds i8, ptr %v89, i64 %v87
  %v91 = load i8, ptr %v90, align 1
  %v92 = add i64 %v49, 6
  %v93 = icmp ult i64 %v92, %v50
  br i1 %v93, label %bb14, label %bb50
bb14:
  %v94 = extractvalue { ptr, i64 } %v15, 0
  %v95 = getelementptr inbounds i8, ptr %v94, i64 %v92
  %v96 = load i8, ptr %v95, align 1
  %v97 = add i64 %v49, 7
  %v98 = icmp ult i64 %v97, %v50
  br i1 %v98, label %bb15, label %bb51
bb15:
  %v99 = extractvalue { ptr, i64 } %v15, 0
  %v100 = getelementptr inbounds i8, ptr %v99, i64 %v97
  %v101 = load i8, ptr %v100, align 1
  %v102 = add i64 %v49, 8
  %v103 = icmp ult i64 %v102, %v50
  br i1 %v103, label %bb16, label %bb52
bb16:
  %v104 = extractvalue { ptr, i64 } %v15, 0
  %v105 = getelementptr inbounds i8, ptr %v104, i64 %v102
  %v106 = load i8, ptr %v105, align 1
  %v107 = add i64 %v49, 9
  %v108 = icmp ult i64 %v107, %v50
  br i1 %v108, label %bb17, label %bb53
bb17:
  %v109 = extractvalue { ptr, i64 } %v15, 0
  %v110 = getelementptr inbounds i8, ptr %v109, i64 %v107
  %v111 = load i8, ptr %v110, align 1
  %v112 = add i64 %v49, 10
  %v113 = icmp ult i64 %v112, %v50
  br i1 %v113, label %bb18, label %bb54
bb18:
  %v114 = extractvalue { ptr, i64 } %v15, 0
  %v115 = getelementptr inbounds i8, ptr %v114, i64 %v112
  %v116 = load i8, ptr %v115, align 1
  %v117 = add i64 %v49, 11
  %v118 = icmp ult i64 %v117, %v50
  br i1 %v118, label %bb19, label %bb55
bb19:
  %v119 = extractvalue { ptr, i64 } %v15, 0
  %v120 = getelementptr inbounds i8, ptr %v119, i64 %v117
  %v121 = load i8, ptr %v120, align 1
  %v122 = add i64 %v49, 12
  %v123 = icmp ult i64 %v122, %v50
  br i1 %v123, label %bb20, label %bb56
bb20:
  %v124 = extractvalue { ptr, i64 } %v15, 0
  %v125 = getelementptr inbounds i8, ptr %v124, i64 %v122
  %v126 = load i8, ptr %v125, align 1
  %v127 = add i64 %v49, 13
  %v128 = icmp ult i64 %v127, %v50
  br i1 %v128, label %bb21, label %bb57
bb21:
  %v129 = extractvalue { ptr, i64 } %v15, 0
  %v130 = getelementptr inbounds i8, ptr %v129, i64 %v127
  %v131 = load i8, ptr %v130, align 1
  %v132 = add i64 %v49, 14
  %v133 = icmp ult i64 %v132, %v50
  br i1 %v133, label %bb22, label %bb58
bb22:
  %v134 = extractvalue { ptr, i64 } %v15, 0
  %v135 = getelementptr inbounds i8, ptr %v134, i64 %v132
  %v136 = load i8, ptr %v135, align 1
  %v137 = add i64 %v49, 15
  %v138 = icmp ult i64 %v137, %v50
  br i1 %v138, label %bb23, label %bb59
bb23:
  %v139 = extractvalue { ptr, i64 } %v15, 0
  %v140 = getelementptr inbounds i8, ptr %v139, i64 %v137
  %v141 = load i8, ptr %v140, align 1
  %v142 = call { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v86, i8 %v91, i8 %v96, i8 %v101, i8 %v106, i8 %v111, i8 %v116, i8 %v121, i8 %v126, i8 %v131, i8 %v136, i8 %v141) #0
  br label %bb24
bb24:
  %v143 = extractvalue { [8 x i8], [8 x i8] } %v142, 0
  store [8 x i8] %v143, ptr %v24, align 1
  %v144 = extractvalue { [8 x i8], [8 x i8] } %v142, 1
  store [8 x i8] %v144, ptr %v25, align 1
  %v145 = udiv i64 %v40, 32
  %v146 = urem i64 %v40, 32
  %v147 = add i64 %v49, 16
  %v148 = udiv i64 %v40, 64
  %v149 = mul i64 %v148, 32
  %v150 = add i64 %v147, %v149
  %v151 = urem i64 %v40, 64
  %v152 = icmp ult i64 %v151, 32
  %v153 = xor i1 %v152, 1
  br i1 %v153, label %bb27, label %bb25
bb25:
  %v154 = add i64 %v150, %v146
  %v155 = icmp ult i64 %v154, %v50
  br i1 %v155, label %bb26, label %bb60
bb26:
  %v156 = extractvalue { ptr, i64 } %v15, 0
  %v157 = getelementptr inbounds i8, ptr %v156, i64 %v154
  %v158 = load i8, ptr %v157, align 1
  %v159 = and i8 %v158, 15
  %v160 = uitofp i8 %v159 to float
  br label %bb29
bb27:
  %v161 = add i64 %v150, %v146
  %v162 = icmp ult i64 %v161, %v50
  br i1 %v162, label %bb28, label %bb61
bb28:
  %v163 = extractvalue { ptr, i64 } %v15, 0
  %v164 = getelementptr inbounds i8, ptr %v163, i64 %v161
  %v165 = load i8, ptr %v164, align 1
  %v166 = trunc i32 4 to i8
  %v167 = and i8 %v166, 7
  %v168 = lshr i8 %v165, %v167
  %v169 = uitofp i8 %v168 to float
  br label %bb29
bb29:
  %v170 = phi float [ %v160, %bb26 ], [ %v169, %bb28 ]
  %v171 = icmp eq i64 %v27, 18446744073709551615
  br i1 %v171, label %bb38, label %bb35
bb30:
  %v172 = extractvalue { ptr } %v195, 0
  %v173 = icmp ult i64 %v145, 8
  br i1 %v173, label %bb31, label %bb62
bb31:
  %v174 = getelementptr inbounds [8 x i8], ptr %v24, i32 0, i64 %v145
  %v175 = load i8, ptr %v174, align 1
  %v176 = uitofp i8 %v175 to float
  %v177 = fmul contract float %v65, %v176
  %v178 = fmul contract float %v177, %v170
  %v179 = getelementptr inbounds [8 x i8], ptr %v25, i32 0, i64 %v145
  %v180 = load i8, ptr %v179, align 1
  %v181 = uitofp i8 %v180 to float
  %v182 = fmul contract float %v81, %v181
  %v183 = fsub contract float %v178, %v182
  store float %v183, ptr %v172, align 4
  br label %bb33
bb32:
  br label %bb33
bb33:
  br label %bb34
bb34:
  ret void
bb35:
  %v184 = extractvalue { ptr, i64 } %v20, 1
  %v185 = icmp ult i64 %v27, %v184
  %v186 = xor i1 %v185, 1
  br i1 %v186, label %bb37, label %bb36
bb36:
  %v187 = extractvalue { ptr, i64 } %v20, 0
  %v188 = getelementptr inbounds float, ptr %v187, i64 %v27
  %v189 = insertvalue { ptr } undef, ptr %v188, 0
  %v190 = extractvalue { ptr } %v189, 0
  br label %bb39
bb37:
  br label %bb38
bb38:
  %v191 = inttoptr i64 0 to ptr
  %v192 = insertvalue { ptr } undef, ptr %v191, 0
  %v193 = extractvalue { ptr } %v192, 0
  br label %bb39
bb39:
  %v194 = phi ptr [ %v190, %bb36 ], [ %v193, %bb38 ]
  %v195 = insertvalue { ptr } undef, ptr %v194, 0
  %v196 = extractvalue { ptr } %v195, 0
  %v197 = ptrtoint ptr %v196 to i64
  %v198 = sub i64 %v197, 0
  %v199 = icmp ule i64 %v198, 0
  %v200 = add i64 %v198, 0
  %v201 = select i1 %v199, i64 %v200, i64 1
  %v202 = icmp eq i64 %v201, 1
  br i1 %v202, label %bb30, label %bb40
bb40:
  %v203 = icmp eq i64 %v201, 0
  br i1 %v203, label %bb32, label %bb41
bb41:
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
bb57:
  call void @llvm.trap() #0
  unreachable
bb58:
  call void @llvm.trap() #0
  unreachable
bb59:
  call void @llvm.trap() #0
  unreachable
bb60:
  call void @llvm.trap() #0
  unreachable
bb61:
  call void @llvm.trap() #0
  unreachable
bb62:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @attention_canvas_heads(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, float %v15, ptr %v16, i64 %v17) #0 {
entry:
  %v18 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v1, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v3, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v5, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v7, 1
  %v26 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v27 = insertvalue { ptr, i64 } %v26, i64 %v9, 1
  %v28 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v29 = insertvalue { ptr, i64 } %v28, i64 %v17, 1
  br label %bb0
bb0:
  %v30 = phi { ptr, i64 } [ %v19, %entry ]
  %v31 = phi { ptr, i64 } [ %v21, %entry ]
  %v32 = phi { ptr, i64 } [ %v23, %entry ]
  %v33 = phi { ptr, i64 } [ %v25, %entry ]
  %v34 = phi { ptr, i64 } [ %v27, %entry ]
  %v35 = phi i32 [ %v10, %entry ]
  %v36 = phi i32 [ %v11, %entry ]
  %v37 = phi i32 [ %v12, %entry ]
  %v38 = phi i32 [ %v13, %entry ]
  %v39 = phi i32 [ %v14, %entry ]
  %v40 = phi float [ %v15, %entry ]
  %v41 = phi { ptr, i64 } [ %v29, %entry ]
  %v42 = alloca {  }, align 1
  %v43 = bitcast ptr %v42 to ptr
  %v44 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v43) #0
  br label %bb1
bb1:
  %v45 = zext i32 %v35 to i64
  %v46 = zext i32 %v37 to i64
  %v47 = mul i64 %v45, %v46
  %v48 = icmp uge i64 %v44, %v47
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb3, label %bb2
bb2:
  br label %bb39
bb3:
  %v50 = icmp eq i64 %v46, 0
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb4, label %bb42
bb4:
  %v52 = udiv i64 %v44, %v46
  %v53 = urem i64 %v44, %v46
  %v54 = zext i32 %v39 to i64
  %v55 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v38, i32 1) #0
  br label %bb5
bb5:
  %v56 = zext i32 %v55 to i64
  %v57 = icmp eq i64 %v56, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb6, label %bb43
bb6:
  %v59 = udiv i64 %v46, %v56
  %v60 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i64 %v59, i64 1) #0
  br label %bb7
bb7:
  %v61 = icmp eq i64 %v60, 0
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb8, label %bb44
bb8:
  %v63 = udiv i64 %v53, %v60
  %v64 = mul i64 %v52, %v46
  %v65 = mul i64 %v64, %v54
  %v66 = mul i64 %v53, %v54
  %v67 = add i64 %v65, %v66
  br label %bb9
bb9:
  %v68 = phi i64 [ 0, %bb8 ], [ %v74, %bb10 ]
  %v69 = icmp ult i64 %v68, %v54
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb11, label %bb10
bb10:
  %v71 = add i64 %v67, %v68
  %v72 = extractvalue { ptr, i64 } %v41, 0
  %v73 = getelementptr inbounds float, ptr %v72, i64 %v71
  store float 0.0, ptr %v73, align 4
  %v74 = add i64 %v68, 1
  br label %bb9
bb11:
  %v75 = zext i32 %v36 to i64
  %v76 = add i64 %v75, %v45
  br label %bb12
bb12:
  %v77 = phi float [ 0.0, %bb11 ], [ %v138, %bb31 ]
  %v78 = phi float [ 0.0, %bb11 ], [ %v172, %bb31 ]
  %v79 = phi i1 [ 0, %bb11 ], [ 1, %bb31 ]
  %v80 = phi i64 [ 0, %bb11 ], [ %v159, %bb31 ]
  %v81 = icmp ult i64 %v80, %v76
  %v82 = xor i1 %v81, 1
  br i1 %v82, label %bb32, label %bb13
bb13:
  %v83 = icmp ult i64 %v80, %v75
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb15, label %bb14
bb14:
  %v85 = mul i64 %v80, %v56
  %v86 = mul i64 %v85, %v54
  %v87 = mul i64 %v63, %v54
  %v88 = add i64 %v86, %v87
  %v89 = extractvalue { ptr, i64 } %v31, 0
  %v90 = extractvalue { ptr, i64 } %v31, 1
  %v91 = extractvalue { ptr, i64 } %v32, 0
  %v92 = extractvalue { ptr, i64 } %v32, 1
  br label %bb16
bb15:
  %v93 = sub i64 %v80, %v75
  %v94 = mul i64 %v93, %v56
  %v95 = mul i64 %v94, %v54
  %v96 = mul i64 %v63, %v54
  %v97 = add i64 %v95, %v96
  %v98 = extractvalue { ptr, i64 } %v33, 0
  %v99 = extractvalue { ptr, i64 } %v33, 1
  %v100 = extractvalue { ptr, i64 } %v34, 0
  %v101 = extractvalue { ptr, i64 } %v34, 1
  br label %bb16
bb16:
  %v102 = phi i64 [ %v88, %bb14 ], [ %v97, %bb15 ]
  %v103 = phi ptr [ %v89, %bb14 ], [ %v98, %bb15 ]
  %v104 = phi i64 [ %v90, %bb14 ], [ %v99, %bb15 ]
  %v105 = phi ptr [ %v91, %bb14 ], [ %v100, %bb15 ]
  %v106 = phi i64 [ %v92, %bb14 ], [ %v101, %bb15 ]
  %v107 = insertvalue { ptr, i64 } undef, ptr %v103, 0
  %v108 = insertvalue { ptr, i64 } %v107, i64 %v104, 1
  %v109 = insertvalue { ptr, i64 } undef, ptr %v105, 0
  %v110 = insertvalue { ptr, i64 } %v109, i64 %v106, 1
  br label %bb17
bb17:
  %v111 = phi i64 [ 0, %bb16 ], [ %v129, %bb20 ]
  %v112 = phi float [ 0.0, %bb16 ], [ %v128, %bb20 ]
  %v113 = icmp ult i64 %v111, %v54
  %v114 = xor i1 %v113, 1
  br i1 %v114, label %bb21, label %bb18
bb18:
  %v115 = add i64 %v67, %v111
  %v116 = extractvalue { ptr, i64 } %v30, 1
  %v117 = icmp ult i64 %v115, %v116
  br i1 %v117, label %bb19, label %bb45
bb19:
  %v118 = extractvalue { ptr, i64 } %v30, 0
  %v119 = getelementptr inbounds float, ptr %v118, i64 %v115
  %v120 = load float, ptr %v119, align 4
  %v121 = add i64 %v102, %v111
  %v122 = extractvalue { ptr, i64 } %v108, 1
  %v123 = icmp ult i64 %v121, %v122
  br i1 %v123, label %bb20, label %bb46
bb20:
  %v124 = extractvalue { ptr, i64 } %v108, 0
  %v125 = getelementptr inbounds float, ptr %v124, i64 %v121
  %v126 = load float, ptr %v125, align 4
  %v127 = fmul contract float %v120, %v126
  %v128 = fadd contract float %v112, %v127
  %v129 = add i64 %v111, 1
  br label %bb17
bb21:
  %v130 = fmul contract float %v112, %v40
  %v131 = xor i1 %v79, 1
  br i1 %v131, label %bb23, label %bb22
bb22:
  %v132 = fcmp ogt float %v130, %v77
  %v133 = xor i1 %v132, 1
  br i1 %v133, label %bb25, label %bb24
bb23:
  br label %bb27
bb24:
  %v134 = fsub contract float %v77, %v130
  %v135 = call float @__nv_expf(float %v134) #0
  br label %bb40
bb25:
  br label %bb26
bb26:
  %v136 = phi float [ %v77, %bb25 ], [ %v130, %bb40 ]
  %v137 = phi float [ 1.0, %bb25 ], [ %v135, %bb40 ]
  br label %bb27
bb27:
  %v138 = phi float [ %v130, %bb23 ], [ %v136, %bb26 ]
  %v139 = phi float [ 0.0, %bb23 ], [ %v137, %bb26 ]
  %v140 = fsub contract float %v130, %v138
  %v141 = call float @__nv_expf(float %v140) #0
  br label %bb41
bb28:
  %v142 = phi i64 [ %v158, %bb30 ], [ 0, %bb41 ]
  %v143 = icmp ult i64 %v142, %v54
  %v144 = xor i1 %v143, 1
  br i1 %v144, label %bb31, label %bb29
bb29:
  %v145 = add i64 %v67, %v142
  %v146 = extractvalue { ptr, i64 } %v41, 0
  %v147 = getelementptr inbounds float, ptr %v146, i64 %v145
  %v148 = load float, ptr %v147, align 4
  %v149 = fmul contract float %v148, %v139
  %v150 = add i64 %v102, %v142
  %v151 = extractvalue { ptr, i64 } %v110, 1
  %v152 = icmp ult i64 %v150, %v151
  br i1 %v152, label %bb30, label %bb47
bb30:
  %v153 = extractvalue { ptr, i64 } %v110, 0
  %v154 = getelementptr inbounds float, ptr %v153, i64 %v150
  %v155 = load float, ptr %v154, align 4
  %v156 = fmul contract float %v141, %v155
  %v157 = fadd contract float %v149, %v156
  store float %v157, ptr %v147, align 4
  %v158 = add i64 %v142, 1
  br label %bb28
bb31:
  %v159 = add i64 %v80, 1
  br label %bb12
bb32:
  %v160 = fcmp ogt float %v78, 0.0
  %v161 = xor i1 %v160, 1
  br i1 %v161, label %bb34, label %bb33
bb33:
  br label %bb35
bb34:
  br label %bb38
bb35:
  %v162 = phi i64 [ 0, %bb33 ], [ %v170, %bb36 ]
  %v163 = icmp ult i64 %v162, %v54
  %v164 = xor i1 %v163, 1
  br i1 %v164, label %bb37, label %bb36
bb36:
  %v165 = add i64 %v67, %v162
  %v166 = extractvalue { ptr, i64 } %v41, 0
  %v167 = getelementptr inbounds float, ptr %v166, i64 %v165
  %v168 = load float, ptr %v167, align 4
  %v169 = fdiv contract float %v168, %v78
  store float %v169, ptr %v167, align 4
  %v170 = add i64 %v162, 1
  br label %bb35
bb37:
  br label %bb38
bb38:
  br label %bb39
bb39:
  ret void
bb40:
  br label %bb26
bb41:
  %v171 = fmul contract float %v78, %v139
  %v172 = fadd contract float %v171, %v141
  br label %bb28
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @fill_u32(i32 %v0, ptr %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { ptr, i64 } undef, ptr %v1, 0
  %v4 = insertvalue { ptr, i64 } %v3, i64 %v2, 1
  br label %bb0
bb0:
  %v5 = phi i32 [ %v0, %entry ]
  %v6 = phi { ptr, i64 } [ %v4, %entry ]
  %v7 = alloca {  }, align 1
  %v8 = bitcast ptr %v7 to ptr
  %v9 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v8) #0
  br label %bb1
bb1:
  %v10 = icmp eq i64 %v9, 18446744073709551615
  br i1 %v10, label %bb8, label %bb5
bb2:
  %v11 = extractvalue { ptr } %v23, 0
  store i32 %v5, ptr %v11, align 4
  br label %bb4
bb3:
  br label %bb4
bb4:
  ret void
bb5:
  %v12 = extractvalue { ptr, i64 } %v6, 1
  %v13 = icmp ult i64 %v9, %v12
  %v14 = xor i1 %v13, 1
  br i1 %v14, label %bb7, label %bb6
bb6:
  %v15 = extractvalue { ptr, i64 } %v6, 0
  %v16 = getelementptr inbounds i32, ptr %v15, i64 %v9
  %v17 = insertvalue { ptr } undef, ptr %v16, 0
  %v18 = extractvalue { ptr } %v17, 0
  br label %bb9
bb7:
  br label %bb8
bb8:
  %v19 = inttoptr i64 0 to ptr
  %v20 = insertvalue { ptr } undef, ptr %v19, 0
  %v21 = extractvalue { ptr } %v20, 0
  br label %bb9
bb9:
  %v22 = phi ptr [ %v18, %bb6 ], [ %v21, %bb8 ]
  %v23 = insertvalue { ptr } undef, ptr %v22, 0
  %v24 = extractvalue { ptr } %v23, 0
  %v25 = ptrtoint ptr %v24 to i64
  %v26 = sub i64 %v25, 0
  %v27 = icmp ule i64 %v26, 0
  %v28 = add i64 %v26, 0
  %v29 = select i1 %v27, i64 %v28, i64 1
  %v30 = icmp eq i64 %v29, 1
  br i1 %v30, label %bb2, label %bb10
bb10:
  %v31 = icmp eq i64 %v29, 0
  br i1 %v31, label %bb3, label %bb11
bb11:
  unreachable
}

define ptx_kernel void @moe_q5_0_project(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr %v10, i64 %v11) #0 {
entry:
  %v12 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v1, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v3, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v5, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v11, 1
  br label %bb0
bb0:
  %v20 = phi { ptr, i64 } [ %v13, %entry ]
  %v21 = phi { ptr, i64 } [ %v15, %entry ]
  %v22 = phi { ptr, i64 } [ %v17, %entry ]
  %v23 = phi i32 [ %v6, %entry ]
  %v24 = phi i32 [ %v7, %entry ]
  %v25 = phi i32 [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi { ptr, i64 } [ %v19, %entry ]
  %v28 = alloca {  }, align 1
  %v29 = alloca [2 x i8], align 1
  %v30 = alloca [4 x i8], align 1
  %v31 = bitcast ptr %v28 to ptr
  %v32 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v31) #0
  br label %bb1
bb1:
  %v33 = zext i32 %v23 to i64
  %v34 = zext i32 %v24 to i64
  %v35 = mul i64 %v33, %v34
  %v36 = zext i32 %v25 to i64
  %v37 = mul i64 %v35, %v36
  %v38 = icmp uge i64 %v32, %v37
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb3, label %bb2
bb2:
  br label %bb25
bb3:
  %v40 = icmp eq i64 %v36, 0
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb4, label %bb33
bb4:
  %v42 = urem i64 %v32, %v36
  %v43 = udiv i64 %v32, %v36
  %v44 = extractvalue { ptr, i64 } %v22, 1
  %v45 = icmp ult i64 %v43, %v44
  br i1 %v45, label %bb5, label %bb34
bb5:
  %v46 = extractvalue { ptr, i64 } %v22, 0
  %v47 = getelementptr inbounds i32, ptr %v46, i64 %v43
  %v48 = load i32, ptr %v47, align 4
  %v49 = zext i32 %v48 to i64
  %v50 = zext i32 %v26 to i64
  %v51 = udiv i64 %v50, 32
  %v52 = mul i64 %v51, 22
  %v53 = mul i64 %v49, %v36
  %v54 = add i64 %v53, %v42
  %v55 = mul i64 %v54, %v52
  %v56 = mul i64 %v43, %v50
  br label %bb6
bb6:
  %v57 = phi float [ 0.0, %bb5 ], [ %v108, %bb20 ]
  %v58 = phi i64 [ 0, %bb5 ], [ %v164, %bb20 ]
  %v59 = icmp ult i64 %v58, %v51
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb21, label %bb7
bb7:
  %v61 = mul i64 %v58, 22
  %v62 = add i64 %v55, %v61
  %v63 = extractvalue { ptr, i64 } %v20, 1
  %v64 = icmp ult i64 %v62, %v63
  br i1 %v64, label %bb8, label %bb35
bb8:
  %v65 = extractvalue { ptr, i64 } %v20, 0
  %v66 = getelementptr inbounds i8, ptr %v65, i64 %v62
  %v67 = load i8, ptr %v66, align 1
  %v68 = add i64 %v62, 1
  %v69 = icmp ult i64 %v68, %v63
  br i1 %v69, label %bb9, label %bb36
bb9:
  %v70 = extractvalue { ptr, i64 } %v20, 0
  %v71 = getelementptr inbounds i8, ptr %v70, i64 %v68
  %v72 = load i8, ptr %v71, align 1
  %v73 = getelementptr inbounds [2 x i8], ptr %v29, i32 0, i64 0
  store i8 %v67, ptr %v73, align 1
  %v74 = getelementptr inbounds [2 x i8], ptr %v29, i32 0, i64 1
  store i8 %v72, ptr %v74, align 1
  %v75 = load [2 x i8], ptr %v29, align 1
  %v76 = alloca [2 x i8], align 2
  store [2 x i8] %v75, ptr %v76, align 2
  %v77 = load i16, ptr %v76, align 2
  %v78 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v77) #0
  br label %bb10
bb10:
  %v79 = add i64 %v62, 2
  %v80 = icmp ult i64 %v79, %v63
  br i1 %v80, label %bb11, label %bb37
bb11:
  %v81 = extractvalue { ptr, i64 } %v20, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = add i64 %v62, 3
  %v85 = icmp ult i64 %v84, %v63
  br i1 %v85, label %bb12, label %bb38
bb12:
  %v86 = extractvalue { ptr, i64 } %v20, 0
  %v87 = getelementptr inbounds i8, ptr %v86, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = add i64 %v62, 4
  %v90 = icmp ult i64 %v89, %v63
  br i1 %v90, label %bb13, label %bb39
bb13:
  %v91 = extractvalue { ptr, i64 } %v20, 0
  %v92 = getelementptr inbounds i8, ptr %v91, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = add i64 %v62, 5
  %v95 = icmp ult i64 %v94, %v63
  br i1 %v95, label %bb14, label %bb40
bb14:
  %v96 = extractvalue { ptr, i64 } %v20, 0
  %v97 = getelementptr inbounds i8, ptr %v96, i64 %v94
  %v98 = load i8, ptr %v97, align 1
  %v99 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 0
  store i8 %v83, ptr %v99, align 1
  %v100 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 1
  store i8 %v88, ptr %v100, align 1
  %v101 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 2
  store i8 %v93, ptr %v101, align 1
  %v102 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 3
  store i8 %v98, ptr %v102, align 1
  %v103 = load [4 x i8], ptr %v30, align 1
  %v104 = alloca [4 x i8], align 4
  store [4 x i8] %v103, ptr %v104, align 4
  %v105 = load i32, ptr %v104, align 4
  %v106 = mul i64 %v58, 32
  %v107 = add i64 %v56, %v106
  br label %bb15
bb15:
  %v108 = phi float [ %v57, %bb14 ], [ %v162, %bb19 ]
  %v109 = phi i64 [ 0, %bb14 ], [ %v163, %bb19 ]
  %v110 = icmp ult i64 %v109, 16
  %v111 = xor i1 %v110, 1
  br i1 %v111, label %bb20, label %bb16
bb16:
  %v112 = add i64 %v62, 6
  %v113 = add i64 %v112, %v109
  %v114 = icmp ult i64 %v113, %v63
  br i1 %v114, label %bb17, label %bb41
bb17:
  %v115 = extractvalue { ptr, i64 } %v20, 0
  %v116 = getelementptr inbounds i8, ptr %v115, i64 %v113
  %v117 = load i8, ptr %v116, align 1
  %v118 = trunc i64 %v109 to i32
  %v119 = and i32 %v118, 31
  %v120 = lshr i32 %v105, %v119
  %v121 = and i32 %v120, 1
  %v122 = bitcast i32 %v121 to i32
  %v123 = and i32 4, 31
  %v124 = shl i32 %v122, %v123
  %v125 = and i8 %v117, 15
  %v126 = zext i8 %v125 to i32
  %v127 = or i32 %v124, %v126
  %v128 = sub i32 %v127, 16
  %v129 = add i64 %v109, 16
  %v130 = trunc i64 %v129 to i32
  %v131 = and i32 %v130, 31
  %v132 = lshr i32 %v105, %v131
  %v133 = and i32 %v132, 1
  %v134 = bitcast i32 %v133 to i32
  %v135 = and i32 4, 31
  %v136 = shl i32 %v134, %v135
  %v137 = trunc i32 4 to i8
  %v138 = and i8 %v137, 7
  %v139 = lshr i8 %v117, %v138
  %v140 = zext i8 %v139 to i32
  %v141 = or i32 %v136, %v140
  %v142 = sub i32 %v141, 16
  %v143 = sitofp i32 %v128 to float
  %v144 = fmul contract float %v78, %v143
  %v145 = add i64 %v107, %v109
  %v146 = extractvalue { ptr, i64 } %v21, 1
  %v147 = icmp ult i64 %v145, %v146
  br i1 %v147, label %bb18, label %bb42
bb18:
  %v148 = extractvalue { ptr, i64 } %v21, 0
  %v149 = getelementptr inbounds float, ptr %v148, i64 %v145
  %v150 = load float, ptr %v149, align 4
  %v151 = fmul contract float %v144, %v150
  %v152 = sitofp i32 %v142 to float
  %v153 = fmul contract float %v78, %v152
  %v154 = add i64 %v107, %v109
  %v155 = add i64 %v154, 16
  %v156 = icmp ult i64 %v155, %v146
  br i1 %v156, label %bb19, label %bb43
bb19:
  %v157 = extractvalue { ptr, i64 } %v21, 0
  %v158 = getelementptr inbounds float, ptr %v157, i64 %v155
  %v159 = load float, ptr %v158, align 4
  %v160 = fmul contract float %v153, %v159
  %v161 = fadd contract float %v151, %v160
  %v162 = fadd contract float %v108, %v161
  %v163 = add i64 %v109, 1
  br label %bb15
bb20:
  %v164 = add i64 %v58, 1
  br label %bb6
bb21:
  %v165 = icmp eq i64 %v32, 18446744073709551615
  br i1 %v165, label %bb29, label %bb26
bb22:
  %v166 = extractvalue { ptr } %v178, 0
  store float %v57, ptr %v166, align 4
  br label %bb24
bb23:
  br label %bb24
bb24:
  br label %bb25
bb25:
  ret void
bb26:
  %v167 = extractvalue { ptr, i64 } %v27, 1
  %v168 = icmp ult i64 %v32, %v167
  %v169 = xor i1 %v168, 1
  br i1 %v169, label %bb28, label %bb27
bb27:
  %v170 = extractvalue { ptr, i64 } %v27, 0
  %v171 = getelementptr inbounds float, ptr %v170, i64 %v32
  %v172 = insertvalue { ptr } undef, ptr %v171, 0
  %v173 = extractvalue { ptr } %v172, 0
  br label %bb30
bb28:
  br label %bb29
bb29:
  %v174 = inttoptr i64 0 to ptr
  %v175 = insertvalue { ptr } undef, ptr %v174, 0
  %v176 = extractvalue { ptr } %v175, 0
  br label %bb30
bb30:
  %v177 = phi ptr [ %v173, %bb27 ], [ %v176, %bb29 ]
  %v178 = insertvalue { ptr } undef, ptr %v177, 0
  %v179 = extractvalue { ptr } %v178, 0
  %v180 = ptrtoint ptr %v179 to i64
  %v181 = sub i64 %v180, 0
  %v182 = icmp ule i64 %v181, 0
  %v183 = add i64 %v181, 0
  %v184 = select i1 %v182, i64 %v183, i64 1
  %v185 = icmp eq i64 %v184, 1
  br i1 %v185, label %bb22, label %bb31
bb31:
  %v186 = icmp eq i64 %v184, 0
  br i1 %v186, label %bb23, label %bb32
bb32:
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @rope(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, float %v7, i32 %v8, ptr %v9, i64 %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v10, 1
  br label %bb0
bb0:
  %v17 = phi { ptr, i64 } [ %v12, %entry ]
  %v18 = phi { ptr, i64 } [ %v14, %entry ]
  %v19 = phi i32 [ %v4, %entry ]
  %v20 = phi i32 [ %v5, %entry ]
  %v21 = phi i32 [ %v6, %entry ]
  %v22 = phi float [ %v7, %entry ]
  %v23 = phi i32 [ %v8, %entry ]
  %v24 = phi { ptr, i64 } [ %v16, %entry ]
  %v25 = alloca {  }, align 1
  %v26 = bitcast ptr %v25 to ptr
  %v27 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v26) #0
  br label %bb1
bb1:
  %v28 = zext i32 %v23 to i64
  %v29 = icmp uge i64 %v27, %v28
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb3, label %bb2
bb2:
  br label %bb14
bb3:
  %v31 = zext i32 %v20 to i64
  %v32 = icmp eq i64 %v31, 0
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb4, label %bb18
bb4:
  %v34 = urem i64 %v27, %v31
  %v35 = sub i64 %v27, %v34
  %v36 = zext i32 %v19 to i64
  %v37 = mul i64 %v36, %v31
  %v38 = icmp eq i64 %v37, 0
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb5, label %bb19
bb5:
  %v40 = udiv i64 %v27, %v37
  %v41 = zext i32 %v21 to i64
  %v42 = icmp uge i64 %v34, %v41
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb8, label %bb6
bb6:
  %v44 = extractvalue { ptr, i64 } %v17, 1
  %v45 = icmp ult i64 %v27, %v44
  br i1 %v45, label %bb7, label %bb20
bb7:
  %v46 = extractvalue { ptr, i64 } %v17, 0
  %v47 = getelementptr inbounds float, ptr %v46, i64 %v27
  %v48 = load float, ptr %v47, align 4
  %v49 = extractvalue { ptr, i64 } %v24, 0
  %v50 = getelementptr inbounds float, ptr %v49, i64 %v27
  store float %v48, ptr %v50, align 4
  br label %bb14
bb8:
  %v51 = urem i64 %v34, 2
  %v52 = icmp eq i64 %v51, 1
  br i1 %v52, label %bb9, label %bb10
bb9:
  br label %bb14
bb10:
  %v53 = udiv i64 %v34, 2
  %v54 = uitofp i64 %v40 to float
  %v55 = fadd contract float %v22, %v54
  %v56 = extractvalue { ptr, i64 } %v18, 1
  %v57 = icmp ult i64 %v53, %v56
  br i1 %v57, label %bb11, label %bb21
bb11:
  %v58 = extractvalue { ptr, i64 } %v18, 0
  %v59 = getelementptr inbounds float, ptr %v58, i64 %v53
  %v60 = load float, ptr %v59, align 4
  %v61 = fmul contract float %v55, %v60
  %v62 = call float @__nv_sinf(float %v61) #0
  br label %bb16
bb12:
  %v63 = extractvalue { ptr, i64 } %v17, 0
  %v64 = getelementptr inbounds float, ptr %v63, i64 %v81
  %v65 = load float, ptr %v64, align 4
  %v66 = add i64 %v81, 1
  %v67 = icmp ult i64 %v66, %v82
  br i1 %v67, label %bb13, label %bb22
bb13:
  %v68 = extractvalue { ptr, i64 } %v17, 0
  %v69 = getelementptr inbounds float, ptr %v68, i64 %v66
  %v70 = load float, ptr %v69, align 4
  %v71 = fmul contract float %v65, %v80
  %v72 = fmul contract float %v70, %v62
  %v73 = extractvalue { ptr, i64 } %v24, 0
  %v74 = getelementptr inbounds float, ptr %v73, i64 %v81
  %v75 = fsub contract float %v71, %v72
  store float %v75, ptr %v74, align 4
  %v76 = fmul contract float %v65, %v62
  %v77 = fmul contract float %v70, %v80
  %v78 = getelementptr inbounds float, ptr %v73, i64 %v66
  %v79 = fadd contract float %v76, %v77
  store float %v79, ptr %v78, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  ret void
bb16:
  %v80 = call float @__nv_cosf(float %v61) #0
  br label %bb17
bb17:
  %v81 = add i64 %v35, %v34
  %v82 = extractvalue { ptr, i64 } %v17, 1
  %v83 = icmp ult i64 %v81, %v82
  br i1 %v83, label %bb12, label %bb23
bb18:
  call void @llvm.trap() #0
  unreachable
bb19:
  call void @llvm.trap() #0
  unreachable
bb20:
  call void @llvm.trap() #0
  unreachable
bb21:
  call void @llvm.trap() #0
  unreachable
bb22:
  call void @llvm.trap() #0
  unreachable
bb23:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @shortconv_mix_rows(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, ptr %v9, i64 %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v5, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v10, 1
  br label %bb0
bb0:
  %v19 = phi { ptr, i64 } [ %v12, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = phi i32 [ %v6, %entry ]
  %v23 = phi i32 [ %v7, %entry ]
  %v24 = phi i32 [ %v8, %entry ]
  %v25 = phi { ptr, i64 } [ %v18, %entry ]
  %v26 = alloca {  }, align 1
  %v27 = bitcast ptr %v26 to ptr
  %v28 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v27) #0
  br label %bb1
bb1:
  %v29 = zext i32 %v22 to i64
  %v30 = zext i32 %v24 to i64
  %v31 = mul i64 %v29, %v30
  %v32 = icmp uge i64 %v28, %v31
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb3, label %bb2
bb2:
  br label %bb4
bb3:
  %v34 = icmp eq i32 %v23, 0
  br i1 %v34, label %bb4, label %bb5
bb4:
  br label %bb24
bb5:
  %v35 = icmp eq i64 %v29, 0
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb6, label %bb32
bb6:
  %v37 = udiv i64 %v28, %v29
  %v38 = urem i64 %v28, %v29
  %v39 = zext i32 %v23 to i64
  %v40 = sub i64 %v39, 1
  %v41 = mul i64 %v37, 3
  %v42 = mul i64 %v41, %v29
  %v43 = mul i64 %v37, %v40
  %v44 = mul i64 %v43, %v29
  %v45 = add i64 %v42, %v38
  %v46 = extractvalue { ptr, i64 } %v19, 1
  %v47 = icmp ult i64 %v45, %v46
  br i1 %v47, label %bb7, label %bb33
bb7:
  %v48 = extractvalue { ptr, i64 } %v19, 0
  %v49 = getelementptr inbounds float, ptr %v48, i64 %v45
  %v50 = load float, ptr %v49, align 4
  %v51 = add i64 %v42, %v29
  %v52 = add i64 %v51, %v38
  %v53 = icmp ult i64 %v52, %v46
  br i1 %v53, label %bb8, label %bb34
bb8:
  %v54 = extractvalue { ptr, i64 } %v19, 0
  %v55 = getelementptr inbounds float, ptr %v54, i64 %v52
  %v56 = load float, ptr %v55, align 4
  %v57 = mul i64 2, %v29
  %v58 = add i64 %v42, %v57
  %v59 = add i64 %v58, %v38
  %v60 = icmp ult i64 %v59, %v46
  br i1 %v60, label %bb9, label %bb35
bb9:
  %v61 = extractvalue { ptr, i64 } %v19, 0
  %v62 = getelementptr inbounds float, ptr %v61, i64 %v59
  %v63 = load float, ptr %v62, align 4
  %v64 = fmul contract float %v50, %v63
  %v65 = mul i64 %v38, %v39
  %v66 = add i64 %v65, %v40
  %v67 = extractvalue { ptr, i64 } %v20, 1
  %v68 = icmp ult i64 %v66, %v67
  br i1 %v68, label %bb10, label %bb36
bb10:
  %v69 = extractvalue { ptr, i64 } %v20, 0
  %v70 = getelementptr inbounds float, ptr %v69, i64 %v66
  %v71 = load float, ptr %v70, align 4
  %v72 = fmul contract float %v64, %v71
  br label %bb11
bb11:
  %v73 = phi float [ %v72, %bb10 ], [ %v89, %bb13 ]
  %v74 = phi i64 [ 0, %bb10 ], [ %v90, %bb13 ]
  %v75 = icmp ult i64 %v74, %v40
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb14, label %bb12
bb12:
  %v77 = mul i64 %v74, %v29
  %v78 = add i64 %v44, %v77
  %v79 = add i64 %v78, %v38
  %v80 = extractvalue { ptr, i64 } %v21, 0
  %v81 = getelementptr inbounds float, ptr %v80, i64 %v79
  %v82 = load float, ptr %v81, align 4
  %v83 = add i64 %v65, %v74
  %v84 = icmp ult i64 %v83, %v67
  br i1 %v84, label %bb13, label %bb37
bb13:
  %v85 = extractvalue { ptr, i64 } %v20, 0
  %v86 = getelementptr inbounds float, ptr %v85, i64 %v83
  %v87 = load float, ptr %v86, align 4
  %v88 = fmul contract float %v82, %v87
  %v89 = fadd contract float %v73, %v88
  %v90 = add i64 %v74, 1
  br label %bb11
bb14:
  %v91 = icmp eq i64 %v28, 18446744073709551615
  br i1 %v91, label %bb28, label %bb25
bb15:
  %v92 = extractvalue { ptr } %v125, 0
  %v93 = fmul contract float %v56, %v73
  store float %v93, ptr %v92, align 4
  br label %bb17
bb16:
  br label %bb17
bb17:
  br label %bb18
bb18:
  %v94 = phi i64 [ 0, %bb17 ], [ %v113, %bb22 ]
  %v95 = icmp ult i64 %v94, %v40
  %v96 = xor i1 %v95, 1
  br i1 %v96, label %bb23, label %bb19
bb19:
  %v97 = add i64 %v94, 1
  %v98 = icmp ult i64 %v97, %v40
  %v99 = xor i1 %v98, 1
  br i1 %v99, label %bb21, label %bb20
bb20:
  %v100 = add i64 %v94, 1
  %v101 = mul i64 %v100, %v29
  %v102 = add i64 %v44, %v101
  %v103 = add i64 %v102, %v38
  %v104 = extractvalue { ptr, i64 } %v21, 0
  %v105 = getelementptr inbounds float, ptr %v104, i64 %v103
  %v106 = load float, ptr %v105, align 4
  br label %bb22
bb21:
  br label %bb22
bb22:
  %v107 = phi float [ %v106, %bb20 ], [ %v64, %bb21 ]
  %v108 = mul i64 %v94, %v29
  %v109 = add i64 %v44, %v108
  %v110 = add i64 %v109, %v38
  %v111 = extractvalue { ptr, i64 } %v21, 0
  %v112 = getelementptr inbounds float, ptr %v111, i64 %v110
  store float %v107, ptr %v112, align 4
  %v113 = add i64 %v94, 1
  br label %bb18
bb23:
  br label %bb24
bb24:
  ret void
bb25:
  %v114 = extractvalue { ptr, i64 } %v25, 1
  %v115 = icmp ult i64 %v28, %v114
  %v116 = xor i1 %v115, 1
  br i1 %v116, label %bb27, label %bb26
bb26:
  %v117 = extractvalue { ptr, i64 } %v25, 0
  %v118 = getelementptr inbounds float, ptr %v117, i64 %v28
  %v119 = insertvalue { ptr } undef, ptr %v118, 0
  %v120 = extractvalue { ptr } %v119, 0
  br label %bb29
bb27:
  br label %bb28
bb28:
  %v121 = inttoptr i64 0 to ptr
  %v122 = insertvalue { ptr } undef, ptr %v121, 0
  %v123 = extractvalue { ptr } %v122, 0
  br label %bb29
bb29:
  %v124 = phi ptr [ %v120, %bb26 ], [ %v123, %bb28 ]
  %v125 = insertvalue { ptr } undef, ptr %v124, 0
  %v126 = extractvalue { ptr } %v125, 0
  %v127 = ptrtoint ptr %v126 to i64
  %v128 = sub i64 %v127, 0
  %v129 = icmp ule i64 %v128, 0
  %v130 = add i64 %v128, 0
  %v131 = select i1 %v129, i64 %v130, i64 1
  %v132 = icmp eq i64 %v131, 1
  br i1 %v132, label %bb15, label %bb30
bb30:
  %v133 = icmp eq i64 %v131, 0
  br i1 %v133, label %bb16, label %bb31
bb31:
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @mul_f32(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = alloca {  }, align 1
  %v16 = bitcast ptr %v15 to ptr
  %v17 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v16) #0
  br label %bb1
bb1:
  %v18 = icmp eq i64 %v17, 18446744073709551615
  br i1 %v18, label %bb10, label %bb7
bb2:
  %v19 = extractvalue { ptr } %v42, 0
  %v20 = extractvalue { ptr, i64 } %v12, 1
  %v21 = icmp ult i64 %v17, %v20
  br i1 %v21, label %bb3, label %bb14
bb3:
  %v22 = extractvalue { ptr, i64 } %v12, 0
  %v23 = getelementptr inbounds float, ptr %v22, i64 %v17
  %v24 = load float, ptr %v23, align 4
  %v25 = extractvalue { ptr, i64 } %v13, 1
  %v26 = icmp ult i64 %v17, %v25
  br i1 %v26, label %bb4, label %bb15
bb4:
  %v27 = extractvalue { ptr, i64 } %v13, 0
  %v28 = getelementptr inbounds float, ptr %v27, i64 %v17
  %v29 = load float, ptr %v28, align 4
  %v30 = fmul contract float %v24, %v29
  store float %v30, ptr %v19, align 4
  br label %bb6
bb5:
  br label %bb6
bb6:
  ret void
bb7:
  %v31 = extractvalue { ptr, i64 } %v14, 1
  %v32 = icmp ult i64 %v17, %v31
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb9, label %bb8
bb8:
  %v34 = extractvalue { ptr, i64 } %v14, 0
  %v35 = getelementptr inbounds float, ptr %v34, i64 %v17
  %v36 = insertvalue { ptr } undef, ptr %v35, 0
  %v37 = extractvalue { ptr } %v36, 0
  br label %bb11
bb9:
  br label %bb10
bb10:
  %v38 = inttoptr i64 0 to ptr
  %v39 = insertvalue { ptr } undef, ptr %v38, 0
  %v40 = extractvalue { ptr } %v39, 0
  br label %bb11
bb11:
  %v41 = phi ptr [ %v37, %bb8 ], [ %v40, %bb10 ]
  %v42 = insertvalue { ptr } undef, ptr %v41, 0
  %v43 = extractvalue { ptr } %v42, 0
  %v44 = ptrtoint ptr %v43 to i64
  %v45 = sub i64 %v44, 0
  %v46 = icmp ule i64 %v45, 0
  %v47 = add i64 %v45, 0
  %v48 = select i1 %v46, i64 %v47, i64 1
  %v49 = icmp eq i64 %v48, 1
  br i1 %v49, label %bb2, label %bb12
bb12:
  %v50 = icmp eq i64 %v48, 0
  br i1 %v50, label %bb5, label %bb13
bb13:
  unreachable
bb14:
  call void @llvm.trap() #0
  unreachable
bb15:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_q8_0_project(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr %v10, i64 %v11) #0 {
entry:
  %v12 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v1, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v3, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v5, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v11, 1
  br label %bb0
bb0:
  %v20 = phi { ptr, i64 } [ %v13, %entry ]
  %v21 = phi { ptr, i64 } [ %v15, %entry ]
  %v22 = phi { ptr, i64 } [ %v17, %entry ]
  %v23 = phi i32 [ %v6, %entry ]
  %v24 = phi i32 [ %v7, %entry ]
  %v25 = phi i32 [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi { ptr, i64 } [ %v19, %entry ]
  %v28 = alloca {  }, align 1
  %v29 = alloca [2 x i8], align 1
  %v30 = bitcast ptr %v28 to ptr
  %v31 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v30) #0
  br label %bb1
bb1:
  %v32 = zext i32 %v23 to i64
  %v33 = zext i32 %v24 to i64
  %v34 = mul i64 %v32, %v33
  %v35 = zext i32 %v25 to i64
  %v36 = mul i64 %v34, %v35
  %v37 = icmp uge i64 %v31, %v36
  %v38 = xor i1 %v37, 1
  br i1 %v38, label %bb3, label %bb2
bb2:
  br label %bb20
bb3:
  %v39 = icmp eq i64 %v35, 0
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb4, label %bb28
bb4:
  %v41 = urem i64 %v31, %v35
  %v42 = udiv i64 %v31, %v35
  %v43 = extractvalue { ptr, i64 } %v22, 1
  %v44 = icmp ult i64 %v42, %v43
  br i1 %v44, label %bb5, label %bb29
bb5:
  %v45 = extractvalue { ptr, i64 } %v22, 0
  %v46 = getelementptr inbounds i32, ptr %v45, i64 %v42
  %v47 = load i32, ptr %v46, align 4
  %v48 = zext i32 %v47 to i64
  %v49 = zext i32 %v26 to i64
  %v50 = udiv i64 %v49, 32
  %v51 = mul i64 %v50, 34
  %v52 = mul i64 %v48, %v35
  %v53 = add i64 %v52, %v41
  %v54 = mul i64 %v53, %v51
  %v55 = mul i64 %v42, %v49
  br label %bb6
bb6:
  %v56 = phi float [ 0.0, %bb5 ], [ %v80, %bb15 ]
  %v57 = phi i64 [ 0, %bb5 ], [ %v102, %bb15 ]
  %v58 = icmp ult i64 %v57, %v50
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb16, label %bb7
bb7:
  %v60 = mul i64 %v57, 34
  %v61 = add i64 %v54, %v60
  %v62 = extractvalue { ptr, i64 } %v20, 1
  %v63 = icmp ult i64 %v61, %v62
  br i1 %v63, label %bb8, label %bb30
bb8:
  %v64 = extractvalue { ptr, i64 } %v20, 0
  %v65 = getelementptr inbounds i8, ptr %v64, i64 %v61
  %v66 = load i8, ptr %v65, align 1
  %v67 = add i64 %v61, 1
  %v68 = icmp ult i64 %v67, %v62
  br i1 %v68, label %bb9, label %bb31
bb9:
  %v69 = extractvalue { ptr, i64 } %v20, 0
  %v70 = getelementptr inbounds i8, ptr %v69, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v72 = getelementptr inbounds [2 x i8], ptr %v29, i32 0, i64 0
  store i8 %v66, ptr %v72, align 1
  %v73 = getelementptr inbounds [2 x i8], ptr %v29, i32 0, i64 1
  store i8 %v71, ptr %v73, align 1
  %v74 = load [2 x i8], ptr %v29, align 1
  %v75 = alloca [2 x i8], align 2
  store [2 x i8] %v74, ptr %v75, align 2
  %v76 = load i16, ptr %v75, align 2
  %v77 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v76) #0
  br label %bb10
bb10:
  %v78 = mul i64 %v57, 32
  %v79 = add i64 %v55, %v78
  br label %bb11
bb11:
  %v80 = phi float [ %v56, %bb10 ], [ %v100, %bb14 ]
  %v81 = phi i64 [ 0, %bb10 ], [ %v101, %bb14 ]
  %v82 = icmp ult i64 %v81, 32
  %v83 = xor i1 %v82, 1
  br i1 %v83, label %bb15, label %bb12
bb12:
  %v84 = add i64 %v61, 2
  %v85 = add i64 %v84, %v81
  %v86 = icmp ult i64 %v85, %v62
  br i1 %v86, label %bb13, label %bb32
bb13:
  %v87 = extractvalue { ptr, i64 } %v20, 0
  %v88 = getelementptr inbounds i8, ptr %v87, i64 %v85
  %v89 = load i8, ptr %v88, align 1
  %v90 = bitcast i8 %v89 to i8
  %v91 = sitofp i8 %v90 to float
  %v92 = fmul contract float %v77, %v91
  %v93 = add i64 %v79, %v81
  %v94 = extractvalue { ptr, i64 } %v21, 1
  %v95 = icmp ult i64 %v93, %v94
  br i1 %v95, label %bb14, label %bb33
bb14:
  %v96 = extractvalue { ptr, i64 } %v21, 0
  %v97 = getelementptr inbounds float, ptr %v96, i64 %v93
  %v98 = load float, ptr %v97, align 4
  %v99 = fmul contract float %v92, %v98
  %v100 = fadd contract float %v80, %v99
  %v101 = add i64 %v81, 1
  br label %bb11
bb15:
  %v102 = add i64 %v57, 1
  br label %bb6
bb16:
  %v103 = icmp eq i64 %v31, 18446744073709551615
  br i1 %v103, label %bb24, label %bb21
bb17:
  %v104 = extractvalue { ptr } %v116, 0
  store float %v56, ptr %v104, align 4
  br label %bb19
bb18:
  br label %bb19
bb19:
  br label %bb20
bb20:
  ret void
bb21:
  %v105 = extractvalue { ptr, i64 } %v27, 1
  %v106 = icmp ult i64 %v31, %v105
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb23, label %bb22
bb22:
  %v108 = extractvalue { ptr, i64 } %v27, 0
  %v109 = getelementptr inbounds float, ptr %v108, i64 %v31
  %v110 = insertvalue { ptr } undef, ptr %v109, 0
  %v111 = extractvalue { ptr } %v110, 0
  br label %bb25
bb23:
  br label %bb24
bb24:
  %v112 = inttoptr i64 0 to ptr
  %v113 = insertvalue { ptr } undef, ptr %v112, 0
  %v114 = extractvalue { ptr } %v113, 0
  br label %bb25
bb25:
  %v115 = phi ptr [ %v111, %bb22 ], [ %v114, %bb24 ]
  %v116 = insertvalue { ptr } undef, ptr %v115, 0
  %v117 = extractvalue { ptr } %v116, 0
  %v118 = ptrtoint ptr %v117 to i64
  %v119 = sub i64 %v118, 0
  %v120 = icmp ule i64 %v119, 0
  %v121 = add i64 %v119, 0
  %v122 = select i1 %v120, i64 %v121, i64 1
  %v123 = icmp eq i64 %v122, 1
  br i1 %v123, label %bb17, label %bb26
bb26:
  %v124 = icmp eq i64 %v122, 0
  br i1 %v124, label %bb18, label %bb27
bb27:
  unreachable
bb28:
  call void @llvm.trap() #0
  unreachable
bb29:
  call void @llvm.trap() #0
  unreachable
bb30:
  call void @llvm.trap() #0
  unreachable
bb31:
  call void @llvm.trap() #0
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
}

define ptx_kernel void @moe_weighted_reduce(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi i32 [ %v4, %entry ]
  %v18 = phi i32 [ %v5, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = alloca {  }, align 1
  %v22 = bitcast ptr %v21 to ptr
  %v23 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v22) #0
  br label %bb1
bb1:
  %v24 = zext i32 %v17 to i64
  %v25 = zext i32 %v19 to i64
  %v26 = mul i64 %v24, %v25
  %v27 = icmp uge i64 %v23, %v26
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb3, label %bb2
bb2:
  br label %bb14
bb3:
  %v29 = icmp eq i64 %v25, 0
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb4, label %bb22
bb4:
  %v31 = udiv i64 %v23, %v25
  %v32 = urem i64 %v23, %v25
  br label %bb5
bb5:
  %v33 = phi float [ 0.0, %bb4 ], [ %v53, %bb8 ]
  %v34 = phi i64 [ 0, %bb4 ], [ %v54, %bb8 ]
  %v35 = zext i32 %v18 to i64
  %v36 = icmp ult i64 %v34, %v35
  %v37 = xor i1 %v36, 1
  br i1 %v37, label %bb9, label %bb6
bb6:
  %v38 = mul i64 %v31, %v35
  %v39 = add i64 %v38, %v34
  %v40 = extractvalue { ptr, i64 } %v16, 1
  %v41 = icmp ult i64 %v39, %v40
  br i1 %v41, label %bb7, label %bb23
bb7:
  %v42 = extractvalue { ptr, i64 } %v16, 0
  %v43 = getelementptr inbounds float, ptr %v42, i64 %v39
  %v44 = load float, ptr %v43, align 4
  %v45 = mul i64 %v39, %v25
  %v46 = add i64 %v45, %v32
  %v47 = extractvalue { ptr, i64 } %v15, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb8, label %bb24
bb8:
  %v49 = extractvalue { ptr, i64 } %v15, 0
  %v50 = getelementptr inbounds float, ptr %v49, i64 %v46
  %v51 = load float, ptr %v50, align 4
  %v52 = fmul contract float %v44, %v51
  %v53 = fadd contract float %v33, %v52
  %v54 = add i64 %v34, 1
  br label %bb5
bb9:
  %v55 = bitcast ptr %v21 to ptr
  %v56 = call i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v55) #0
  br label %bb10
bb10:
  %v57 = icmp eq i64 %v56, 18446744073709551615
  br i1 %v57, label %bb18, label %bb15
bb11:
  %v58 = extractvalue { ptr } %v70, 0
  store float %v33, ptr %v58, align 4
  br label %bb13
bb12:
  br label %bb13
bb13:
  br label %bb14
bb14:
  ret void
bb15:
  %v59 = extractvalue { ptr, i64 } %v20, 1
  %v60 = icmp ult i64 %v56, %v59
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb17, label %bb16
bb16:
  %v62 = extractvalue { ptr, i64 } %v20, 0
  %v63 = getelementptr inbounds float, ptr %v62, i64 %v56
  %v64 = insertvalue { ptr } undef, ptr %v63, 0
  %v65 = extractvalue { ptr } %v64, 0
  br label %bb19
bb17:
  br label %bb18
bb18:
  %v66 = inttoptr i64 0 to ptr
  %v67 = insertvalue { ptr } undef, ptr %v66, 0
  %v68 = extractvalue { ptr } %v67, 0
  br label %bb19
bb19:
  %v69 = phi ptr [ %v65, %bb16 ], [ %v68, %bb18 ]
  %v70 = insertvalue { ptr } undef, ptr %v69, 0
  %v71 = extractvalue { ptr } %v70, 0
  %v72 = ptrtoint ptr %v71 to i64
  %v73 = sub i64 %v72, 0
  %v74 = icmp ule i64 %v73, 0
  %v75 = add i64 %v73, 0
  %v76 = select i1 %v74, i64 %v75, i64 1
  %v77 = icmp eq i64 %v76, 1
  br i1 %v77, label %bb11, label %bb20
bb20:
  %v78 = icmp eq i64 %v76, 0
  br i1 %v78, label %bb12, label %bb21
bb21:
  unreachable
bb22:
  call void @llvm.trap() #0
  unreachable
bb23:
  call void @llvm.trap() #0
  unreachable
bb24:
  call void @llvm.trap() #0
  unreachable
}

define i64 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v5 = zext i32 %v2 to i64
  %v6 = zext i32 %v3 to i64
  %v7 = zext i32 %v4 to i64
  %v8 = icmp eq i64 %v6, 0
  br i1 %v8, label %bb10, label %bb8
bb4:
  %v9 = xor i1 %v20, 1
  br i1 %v9, label %bb6, label %bb5
bb5:
  %v10 = icmp ne i64 %v19, 18446744073709551615
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v11 = phi i1 [ %v10, %bb5 ], [ 0, %bb6 ]
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb14, label %bb13
bb8:
  %v13 = sub i64 18446744073709551615, %v7
  %v14 = udiv i64 %v13, %v6
  %v15 = icmp ugt i64 %v5, %v14
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb11, label %bb9
bb9:
  br label %bb10
bb10:
  br label %bb12
bb11:
  %v17 = mul i64 %v5, %v6
  %v18 = add i64 %v17, %v7
  br label %bb12
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v1) #0
  br label %bb4
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615
  br i1 %v21, label %bb14, label %bb15
bb14:
  br label %bb15
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22
}

define float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v0, i64 %v1, i64 %v2, ptr %v3, i64 %v4, i64 %v5, i32 %v6, i64 %v7) #0 {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v4, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v9, %entry ]
  %v13 = phi i64 [ %v2, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = phi i64 [ %v5, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = phi i64 [ %v7, %entry ]
  %v18 = alloca [2 x i8], align 1
  br label %bb1
bb1:
  %v19 = phi float [ 0.0, %bb0 ], [ %v45, %bb19 ]
  %v20 = phi i64 [ 0, %bb0 ], [ %v190, %bb19 ]
  %v21 = zext i32 %v16 to i64
  %v22 = icmp ult i64 %v20, %v21
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb20, label %bb2
bb2:
  %v24 = mul i64 %v20, 210
  %v25 = add i64 %v13, %v24
  %v26 = add i64 %v25, 208
  %v27 = extractvalue { ptr, i64 } %v12, 1
  %v28 = icmp ult i64 %v26, %v27
  br i1 %v28, label %bb3, label %bb21
bb3:
  %v29 = extractvalue { ptr, i64 } %v12, 0
  %v30 = getelementptr inbounds i8, ptr %v29, i64 %v26
  %v31 = load i8, ptr %v30, align 1
  %v32 = add i64 %v25, 209
  %v33 = icmp ult i64 %v32, %v27
  br i1 %v33, label %bb4, label %bb22
bb4:
  %v34 = extractvalue { ptr, i64 } %v12, 0
  %v35 = getelementptr inbounds i8, ptr %v34, i64 %v32
  %v36 = load i8, ptr %v35, align 1
  %v37 = getelementptr inbounds [2 x i8], ptr %v18, i32 0, i64 0
  store i8 %v31, ptr %v37, align 1
  %v38 = getelementptr inbounds [2 x i8], ptr %v18, i32 0, i64 1
  store i8 %v36, ptr %v38, align 1
  %v39 = load [2 x i8], ptr %v18, align 1
  %v40 = alloca [2 x i8], align 2
  store [2 x i8] %v39, ptr %v40, align 2
  %v41 = load i16, ptr %v40, align 2
  %v42 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v41) #0
  br label %bb5
bb5:
  %v43 = mul i64 %v20, 256
  %v44 = add i64 %v15, %v43
  br label %bb6
bb6:
  %v45 = phi float [ %v19, %bb5 ], [ %v188, %bb18 ]
  %v46 = phi i64 [ 0, %bb5 ], [ %v189, %bb18 ]
  %v47 = icmp ult i64 %v46, 2
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb19, label %bb7
bb7:
  %v49 = mul i64 %v46, 64
  %v50 = add i64 %v25, %v49
  %v51 = add i64 %v25, 128
  %v52 = mul i64 %v46, 32
  %v53 = add i64 %v51, %v52
  %v54 = add i64 %v25, 192
  %v55 = mul i64 %v46, 8
  %v56 = add i64 %v54, %v55
  %v57 = mul i64 %v46, 128
  %v58 = add i64 %v44, %v57
  %v59 = udiv i64 %v17, 16
  %v60 = add i64 %v50, %v17
  %v61 = icmp ult i64 %v60, %v27
  br i1 %v61, label %bb8, label %bb23
bb8:
  %v62 = extractvalue { ptr, i64 } %v12, 0
  %v63 = getelementptr inbounds i8, ptr %v62, i64 %v60
  %v64 = load i8, ptr %v63, align 1
  %v65 = and i8 %v64, 15
  %v66 = zext i8 %v65 to i32
  %v67 = add i64 %v53, %v17
  %v68 = icmp ult i64 %v67, %v27
  br i1 %v68, label %bb9, label %bb24
bb9:
  %v69 = extractvalue { ptr, i64 } %v12, 0
  %v70 = getelementptr inbounds i8, ptr %v69, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v72 = and i8 %v71, 3
  %v73 = zext i8 %v72 to i32
  %v74 = and i32 4, 31
  %v75 = shl i32 %v73, %v74
  %v76 = or i32 %v66, %v75
  %v77 = sub i32 %v76, 32
  %v78 = add i64 %v60, 32
  %v79 = icmp ult i64 %v78, %v27
  br i1 %v79, label %bb10, label %bb25
bb10:
  %v80 = extractvalue { ptr, i64 } %v12, 0
  %v81 = getelementptr inbounds i8, ptr %v80, i64 %v78
  %v82 = load i8, ptr %v81, align 1
  %v83 = and i8 %v82, 15
  %v84 = zext i8 %v83 to i32
  %v85 = trunc i32 2 to i8
  %v86 = and i8 %v85, 7
  %v87 = lshr i8 %v71, %v86
  %v88 = and i8 %v87, 3
  %v89 = zext i8 %v88 to i32
  %v90 = and i32 4, 31
  %v91 = shl i32 %v89, %v90
  %v92 = or i32 %v84, %v91
  %v93 = sub i32 %v92, 32
  %v94 = trunc i32 4 to i8
  %v95 = and i8 %v94, 7
  %v96 = lshr i8 %v64, %v95
  %v97 = zext i8 %v96 to i32
  %v98 = trunc i32 4 to i8
  %v99 = and i8 %v98, 7
  %v100 = lshr i8 %v71, %v99
  %v101 = and i8 %v100, 3
  %v102 = zext i8 %v101 to i32
  %v103 = and i32 4, 31
  %v104 = shl i32 %v102, %v103
  %v105 = or i32 %v97, %v104
  %v106 = sub i32 %v105, 32
  %v107 = trunc i32 4 to i8
  %v108 = and i8 %v107, 7
  %v109 = lshr i8 %v82, %v108
  %v110 = zext i8 %v109 to i32
  %v111 = trunc i32 6 to i8
  %v112 = and i8 %v111, 7
  %v113 = lshr i8 %v71, %v112
  %v114 = and i8 %v113, 3
  %v115 = zext i8 %v114 to i32
  %v116 = and i32 4, 31
  %v117 = shl i32 %v115, %v116
  %v118 = or i32 %v110, %v117
  %v119 = sub i32 %v118, 32
  %v120 = add i64 %v56, %v59
  %v121 = icmp ult i64 %v120, %v27
  br i1 %v121, label %bb11, label %bb26
bb11:
  %v122 = extractvalue { ptr, i64 } %v12, 0
  %v123 = getelementptr inbounds i8, ptr %v122, i64 %v120
  %v124 = load i8, ptr %v123, align 1
  %v125 = bitcast i8 %v124 to i8
  %v126 = sitofp i8 %v125 to float
  %v127 = fmul contract float %v42, %v126
  %v128 = sitofp i32 %v77 to float
  %v129 = fmul contract float %v127, %v128
  %v130 = add i64 %v58, %v17
  %v131 = extractvalue { ptr, i64 } %v14, 1
  %v132 = icmp ult i64 %v130, %v131
  br i1 %v132, label %bb12, label %bb27
bb12:
  %v133 = extractvalue { ptr, i64 } %v14, 0
  %v134 = getelementptr inbounds float, ptr %v133, i64 %v130
  %v135 = load float, ptr %v134, align 4
  %v136 = fmul contract float %v129, %v135
  %v137 = fadd contract float %v45, %v136
  %v138 = add i64 %v120, 2
  %v139 = icmp ult i64 %v138, %v27
  br i1 %v139, label %bb13, label %bb28
bb13:
  %v140 = extractvalue { ptr, i64 } %v12, 0
  %v141 = getelementptr inbounds i8, ptr %v140, i64 %v138
  %v142 = load i8, ptr %v141, align 1
  %v143 = bitcast i8 %v142 to i8
  %v144 = sitofp i8 %v143 to float
  %v145 = fmul contract float %v42, %v144
  %v146 = sitofp i32 %v93 to float
  %v147 = fmul contract float %v145, %v146
  %v148 = add i64 %v130, 32
  %v149 = icmp ult i64 %v148, %v131
  br i1 %v149, label %bb14, label %bb29
bb14:
  %v150 = extractvalue { ptr, i64 } %v14, 0
  %v151 = getelementptr inbounds float, ptr %v150, i64 %v148
  %v152 = load float, ptr %v151, align 4
  %v153 = fmul contract float %v147, %v152
  %v154 = fadd contract float %v137, %v153
  %v155 = add i64 %v120, 4
  %v156 = icmp ult i64 %v155, %v27
  br i1 %v156, label %bb15, label %bb30
bb15:
  %v157 = extractvalue { ptr, i64 } %v12, 0
  %v158 = getelementptr inbounds i8, ptr %v157, i64 %v155
  %v159 = load i8, ptr %v158, align 1
  %v160 = bitcast i8 %v159 to i8
  %v161 = sitofp i8 %v160 to float
  %v162 = fmul contract float %v42, %v161
  %v163 = sitofp i32 %v106 to float
  %v164 = fmul contract float %v162, %v163
  %v165 = add i64 %v130, 64
  %v166 = icmp ult i64 %v165, %v131
  br i1 %v166, label %bb16, label %bb31
bb16:
  %v167 = extractvalue { ptr, i64 } %v14, 0
  %v168 = getelementptr inbounds float, ptr %v167, i64 %v165
  %v169 = load float, ptr %v168, align 4
  %v170 = fmul contract float %v164, %v169
  %v171 = fadd contract float %v154, %v170
  %v172 = add i64 %v120, 6
  %v173 = icmp ult i64 %v172, %v27
  br i1 %v173, label %bb17, label %bb32
bb17:
  %v174 = extractvalue { ptr, i64 } %v12, 0
  %v175 = getelementptr inbounds i8, ptr %v174, i64 %v172
  %v176 = load i8, ptr %v175, align 1
  %v177 = bitcast i8 %v176 to i8
  %v178 = sitofp i8 %v177 to float
  %v179 = fmul contract float %v42, %v178
  %v180 = sitofp i32 %v119 to float
  %v181 = fmul contract float %v179, %v180
  %v182 = add i64 %v130, 96
  %v183 = icmp ult i64 %v182, %v131
  br i1 %v183, label %bb18, label %bb33
bb18:
  %v184 = extractvalue { ptr, i64 } %v14, 0
  %v185 = getelementptr inbounds float, ptr %v184, i64 %v182
  %v186 = load float, ptr %v185, align 4
  %v187 = fmul contract float %v181, %v186
  %v188 = fadd contract float %v171, %v187
  %v189 = add i64 %v46, 1
  br label %bb6
bb19:
  %v190 = add i64 %v20, 1
  br label %bb1
bb20:
  ret float %v19
bb21:
  call void @llvm.trap() #0
  unreachable
bb22:
  call void @llvm.trap() #0
  unreachable
bb23:
  call void @llvm.trap() #0
  unreachable
bb24:
  call void @llvm.trap() #0
  unreachable
bb25:
  call void @llvm.trap() #0
  unreachable
bb26:
  call void @llvm.trap() #0
  unreachable
bb27:
  call void @llvm.trap() #0
  unreachable
bb28:
  call void @llvm.trap() #0
  unreachable
bb29:
  call void @llvm.trap() #0
  unreachable
bb30:
  call void @llvm.trap() #0
  unreachable
bb31:
  call void @llvm.trap() #0
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
}

define i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = alloca i32, align 4
  %v5 = alloca i32, align 4
  store i32 %v2, ptr %v4, align 4
  store i32 %v3, ptr %v5, align 4
  %v6 = bitcast ptr %v5 to ptr
  %v7 = bitcast ptr %v4 to ptr
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(ptr %v6, ptr %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v10 = load i32, ptr %v4, align 4
  br label %bb4
bb3:
  %v11 = load i32, ptr %v5, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
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

declare i32 @llvm.nvvm.idp4a.s.s(i32, i32, i32)

define float @cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk(ptr %v0, i64 %v1, i64 %v2, i64 %v3, i32 %v4, float %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v7, %entry ]
  %v9 = phi i64 [ %v2, %entry ]
  %v10 = phi i64 [ %v3, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = phi float [ %v5, %entry ]
  %v13 = alloca [4 x i8], align 1
  %v14 = alloca [2 x i8], align 1
  %v15 = extractvalue { ptr, i64 } %v8, 0
  %v16 = extractvalue { ptr, i64 } %v8, 1
  %v17 = call i8 @cuda_kernels__oxide_kernels__kernels__q6k_quant(ptr %v15, i64 %v16, i64 %v9, i64 %v10) #0
  br label %bb1
bb1:
  %v18 = bitcast i8 %v17 to i8
  %v19 = add i64 %v10, 1
  %v20 = extractvalue { ptr, i64 } %v8, 0
  %v21 = extractvalue { ptr, i64 } %v8, 1
  %v22 = call i8 @cuda_kernels__oxide_kernels__kernels__q6k_quant(ptr %v20, i64 %v21, i64 %v9, i64 %v19) #0
  br label %bb2
bb2:
  %v23 = bitcast i8 %v22 to i8
  %v24 = add i64 %v10, 2
  %v25 = extractvalue { ptr, i64 } %v8, 0
  %v26 = extractvalue { ptr, i64 } %v8, 1
  %v27 = call i8 @cuda_kernels__oxide_kernels__kernels__q6k_quant(ptr %v25, i64 %v26, i64 %v9, i64 %v24) #0
  br label %bb3
bb3:
  %v28 = bitcast i8 %v27 to i8
  %v29 = add i64 %v10, 3
  %v30 = extractvalue { ptr, i64 } %v8, 0
  %v31 = extractvalue { ptr, i64 } %v8, 1
  %v32 = call i8 @cuda_kernels__oxide_kernels__kernels__q6k_quant(ptr %v30, i64 %v31, i64 %v9, i64 %v29) #0
  br label %bb4
bb4:
  %v33 = bitcast i8 %v32 to i8
  %v34 = getelementptr inbounds [4 x i8], ptr %v13, i32 0, i64 0
  store i8 %v18, ptr %v34, align 1
  %v35 = getelementptr inbounds [4 x i8], ptr %v13, i32 0, i64 1
  store i8 %v23, ptr %v35, align 1
  %v36 = getelementptr inbounds [4 x i8], ptr %v13, i32 0, i64 2
  store i8 %v28, ptr %v36, align 1
  %v37 = getelementptr inbounds [4 x i8], ptr %v13, i32 0, i64 3
  store i8 %v33, ptr %v37, align 1
  %v38 = load [4 x i8], ptr %v13, align 1
  %v39 = alloca [4 x i8], align 4
  store [4 x i8] %v38, ptr %v39, align 4
  %v40 = load i32, ptr %v39, align 4
  %v41 = add i64 %v9, 208
  %v42 = extractvalue { ptr, i64 } %v8, 1
  %v43 = icmp ult i64 %v41, %v42
  %v44 = extractvalue { ptr, i64 } %v8, 0
  %v45 = getelementptr inbounds i8, ptr %v44, i64 %v41
  %v46 = load i8, ptr %v45, align 1
  %v47 = add i64 %v9, 209
  %v48 = icmp ult i64 %v47, %v42
  %v49 = extractvalue { ptr, i64 } %v8, 0
  %v50 = getelementptr inbounds i8, ptr %v49, i64 %v47
  %v51 = load i8, ptr %v50, align 1
  %v52 = getelementptr inbounds [2 x i8], ptr %v14, i32 0, i64 0
  store i8 %v46, ptr %v52, align 1
  %v53 = getelementptr inbounds [2 x i8], ptr %v14, i32 0, i64 1
  store i8 %v51, ptr %v53, align 1
  %v54 = load [2 x i8], ptr %v14, align 1
  %v55 = alloca [2 x i8], align 2
  store [2 x i8] %v54, ptr %v55, align 2
  %v56 = load i16, ptr %v55, align 2
  %v57 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v56) #0
  br label %bb5
bb5:
  %v58 = udiv i64 %v10, 128
  %v59 = urem i64 %v10, 128
  %v60 = urem i64 %v59, 32
  %v61 = udiv i64 %v59, 32
  %v62 = add i64 %v9, 192
  %v63 = mul i64 %v58, 8
  %v64 = add i64 %v62, %v63
  %v65 = udiv i64 %v60, 16
  %v66 = add i64 %v64, %v65
  %v67 = mul i64 %v61, 2
  %v68 = add i64 %v66, %v67
  %v69 = icmp ult i64 %v68, %v42
  %v70 = extractvalue { ptr, i64 } %v8, 0
  %v71 = getelementptr inbounds i8, ptr %v70, i64 %v68
  %v72 = load i8, ptr %v71, align 1
  %v73 = bitcast i8 %v72 to i8
  %v74 = sitofp i8 %v73 to float
  %v75 = fmul contract float %v57, %v74
  %v76 = fmul contract float %v75, %v12
  %v77 = call i32 @llvm.nvvm.idp4a.s.s(i32 %v40, i32 %v11, i32 0) #0
  br label %bb6
bb6:
  %v78 = sitofp i32 %v77 to float
  %v79 = fmul contract float %v76, %v78
  ret float %v79
}

define float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v0, i64 %v1, i64 %v2, ptr %v3, i64 %v4, i64 %v5, i32 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v4, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi i64 [ %v2, %entry ]
  %v13 = phi { ptr, i64 } [ %v10, %entry ]
  %v14 = phi i64 [ %v5, %entry ]
  %v15 = phi i32 [ %v6, %entry ]
  %v16 = alloca [2 x i8], align 1
  %v17 = alloca [2 x i8], align 1
  %v18 = alloca [8 x i8], align 1
  %v19 = alloca [8 x i8], align 1
  br label %bb1
bb1:
  %v20 = phi float [ 0.0, %bb0 ], [ %v152, %bb40 ]
  %v21 = phi i32 [ 0, %bb0 ], [ %v213, %bb40 ]
  %v22 = icmp ult i32 %v21, %v15
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb41, label %bb2
bb2:
  %v24 = zext i32 %v21 to i64
  %v25 = mul i64 %v24, 144
  %v26 = add i64 %v12, %v25
  %v27 = extractvalue { ptr, i64 } %v11, 1
  %v28 = icmp ult i64 %v26, %v27
  br i1 %v28, label %bb3, label %bb42
bb3:
  %v29 = extractvalue { ptr, i64 } %v11, 0
  %v30 = getelementptr inbounds i8, ptr %v29, i64 %v26
  %v31 = load i8, ptr %v30, align 1
  %v32 = add i64 %v26, 1
  %v33 = icmp ult i64 %v32, %v27
  br i1 %v33, label %bb4, label %bb43
bb4:
  %v34 = extractvalue { ptr, i64 } %v11, 0
  %v35 = getelementptr inbounds i8, ptr %v34, i64 %v32
  %v36 = load i8, ptr %v35, align 1
  %v37 = getelementptr inbounds [2 x i8], ptr %v16, i32 0, i64 0
  store i8 %v31, ptr %v37, align 1
  %v38 = getelementptr inbounds [2 x i8], ptr %v16, i32 0, i64 1
  store i8 %v36, ptr %v38, align 1
  %v39 = load [2 x i8], ptr %v16, align 1
  %v40 = alloca [2 x i8], align 2
  store [2 x i8] %v39, ptr %v40, align 2
  %v41 = load i16, ptr %v40, align 2
  %v42 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v41) #0
  br label %bb5
bb5:
  %v43 = add i64 %v26, 2
  %v44 = icmp ult i64 %v43, %v27
  br i1 %v44, label %bb6, label %bb44
bb6:
  %v45 = extractvalue { ptr, i64 } %v11, 0
  %v46 = getelementptr inbounds i8, ptr %v45, i64 %v43
  %v47 = load i8, ptr %v46, align 1
  %v48 = add i64 %v26, 3
  %v49 = icmp ult i64 %v48, %v27
  br i1 %v49, label %bb7, label %bb45
bb7:
  %v50 = extractvalue { ptr, i64 } %v11, 0
  %v51 = getelementptr inbounds i8, ptr %v50, i64 %v48
  %v52 = load i8, ptr %v51, align 1
  %v53 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 0
  store i8 %v47, ptr %v53, align 1
  %v54 = getelementptr inbounds [2 x i8], ptr %v17, i32 0, i64 1
  store i8 %v52, ptr %v54, align 1
  %v55 = load [2 x i8], ptr %v17, align 1
  %v56 = alloca [2 x i8], align 2
  store [2 x i8] %v55, ptr %v56, align 2
  %v57 = load i16, ptr %v56, align 2
  %v58 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v57) #0
  br label %bb8
bb8:
  %v59 = add i64 %v26, 4
  %v60 = icmp ult i64 %v59, %v27
  br i1 %v60, label %bb9, label %bb46
bb9:
  %v61 = extractvalue { ptr, i64 } %v11, 0
  %v62 = getelementptr inbounds i8, ptr %v61, i64 %v59
  %v63 = load i8, ptr %v62, align 1
  %v64 = add i64 %v26, 5
  %v65 = icmp ult i64 %v64, %v27
  br i1 %v65, label %bb10, label %bb47
bb10:
  %v66 = extractvalue { ptr, i64 } %v11, 0
  %v67 = getelementptr inbounds i8, ptr %v66, i64 %v64
  %v68 = load i8, ptr %v67, align 1
  %v69 = add i64 %v26, 6
  %v70 = icmp ult i64 %v69, %v27
  br i1 %v70, label %bb11, label %bb48
bb11:
  %v71 = extractvalue { ptr, i64 } %v11, 0
  %v72 = getelementptr inbounds i8, ptr %v71, i64 %v69
  %v73 = load i8, ptr %v72, align 1
  %v74 = add i64 %v26, 7
  %v75 = icmp ult i64 %v74, %v27
  br i1 %v75, label %bb12, label %bb49
bb12:
  %v76 = extractvalue { ptr, i64 } %v11, 0
  %v77 = getelementptr inbounds i8, ptr %v76, i64 %v74
  %v78 = load i8, ptr %v77, align 1
  %v79 = add i64 %v26, 8
  %v80 = icmp ult i64 %v79, %v27
  br i1 %v80, label %bb13, label %bb50
bb13:
  %v81 = extractvalue { ptr, i64 } %v11, 0
  %v82 = getelementptr inbounds i8, ptr %v81, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = add i64 %v26, 9
  %v85 = icmp ult i64 %v84, %v27
  br i1 %v85, label %bb14, label %bb51
bb14:
  %v86 = extractvalue { ptr, i64 } %v11, 0
  %v87 = getelementptr inbounds i8, ptr %v86, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = add i64 %v26, 10
  %v90 = icmp ult i64 %v89, %v27
  br i1 %v90, label %bb15, label %bb52
bb15:
  %v91 = extractvalue { ptr, i64 } %v11, 0
  %v92 = getelementptr inbounds i8, ptr %v91, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = add i64 %v26, 11
  %v95 = icmp ult i64 %v94, %v27
  br i1 %v95, label %bb16, label %bb53
bb16:
  %v96 = extractvalue { ptr, i64 } %v11, 0
  %v97 = getelementptr inbounds i8, ptr %v96, i64 %v94
  %v98 = load i8, ptr %v97, align 1
  %v99 = add i64 %v26, 12
  %v100 = icmp ult i64 %v99, %v27
  br i1 %v100, label %bb17, label %bb54
bb17:
  %v101 = extractvalue { ptr, i64 } %v11, 0
  %v102 = getelementptr inbounds i8, ptr %v101, i64 %v99
  %v103 = load i8, ptr %v102, align 1
  %v104 = add i64 %v26, 13
  %v105 = icmp ult i64 %v104, %v27
  br i1 %v105, label %bb18, label %bb55
bb18:
  %v106 = extractvalue { ptr, i64 } %v11, 0
  %v107 = getelementptr inbounds i8, ptr %v106, i64 %v104
  %v108 = load i8, ptr %v107, align 1
  %v109 = add i64 %v26, 14
  %v110 = icmp ult i64 %v109, %v27
  br i1 %v110, label %bb19, label %bb56
bb19:
  %v111 = extractvalue { ptr, i64 } %v11, 0
  %v112 = getelementptr inbounds i8, ptr %v111, i64 %v109
  %v113 = load i8, ptr %v112, align 1
  %v114 = add i64 %v26, 15
  %v115 = icmp ult i64 %v114, %v27
  br i1 %v115, label %bb20, label %bb57
bb20:
  %v116 = extractvalue { ptr, i64 } %v11, 0
  %v117 = getelementptr inbounds i8, ptr %v116, i64 %v114
  %v118 = load i8, ptr %v117, align 1
  %v119 = call { [8 x i8], [8 x i8] } @cuda_kernels__oxide_kernels__decode_scales_mins(i8 %v63, i8 %v68, i8 %v73, i8 %v78, i8 %v83, i8 %v88, i8 %v93, i8 %v98, i8 %v103, i8 %v108, i8 %v113, i8 %v118) #0
  br label %bb21
bb21:
  %v120 = extractvalue { [8 x i8], [8 x i8] } %v119, 0
  store [8 x i8] %v120, ptr %v18, align 1
  %v121 = extractvalue { [8 x i8], [8 x i8] } %v119, 1
  store [8 x i8] %v121, ptr %v19, align 1
  %v122 = zext i32 %v21 to i64
  %v123 = mul i64 %v122, 256
  %v124 = add i64 %v14, %v123
  br label %bb22
bb22:
  %v125 = phi float [ %v20, %bb21 ], [ %v149, %bb28 ]
  %v126 = phi i64 [ 0, %bb21 ], [ %v150, %bb28 ]
  %v127 = icmp ult i64 %v126, 8
  %v128 = xor i1 %v127, 1
  br i1 %v128, label %bb29, label %bb23
bb23:
  br label %bb24
bb24:
  %v129 = phi float [ 0.0, %bb23 ], [ %v141, %bb26 ]
  %v130 = phi i64 [ 0, %bb23 ], [ %v142, %bb26 ]
  %v131 = icmp ult i64 %v130, 32
  %v132 = xor i1 %v131, 1
  br i1 %v132, label %bb27, label %bb25
bb25:
  %v133 = mul i64 %v126, 32
  %v134 = add i64 %v124, %v133
  %v135 = add i64 %v134, %v130
  %v136 = extractvalue { ptr, i64 } %v13, 1
  %v137 = icmp ult i64 %v135, %v136
  br i1 %v137, label %bb26, label %bb58
bb26:
  %v138 = extractvalue { ptr, i64 } %v13, 0
  %v139 = getelementptr inbounds float, ptr %v138, i64 %v135
  %v140 = load float, ptr %v139, align 4
  %v141 = fadd contract float %v129, %v140
  %v142 = add i64 %v130, 1
  br label %bb24
bb27:
  %v143 = icmp ult i64 %v126, 8
  br i1 %v143, label %bb28, label %bb59
bb28:
  %v144 = getelementptr inbounds [8 x i8], ptr %v19, i32 0, i64 %v126
  %v145 = load i8, ptr %v144, align 1
  %v146 = uitofp i8 %v145 to float
  %v147 = fmul contract float %v58, %v146
  %v148 = fmul contract float %v147, %v129
  %v149 = fsub contract float %v125, %v148
  %v150 = add i64 %v126, 1
  br label %bb22
bb29:
  %v151 = add i64 %v26, 16
  br label %bb30
bb30:
  %v152 = phi float [ %v125, %bb29 ], [ %v211, %bb39 ]
  %v153 = phi i64 [ 0, %bb29 ], [ %v212, %bb39 ]
  %v154 = icmp ult i64 %v153, 4
  %v155 = xor i1 %v154, 1
  br i1 %v155, label %bb40, label %bb31
bb31:
  %v156 = mul i64 %v153, 32
  %v157 = add i64 %v151, %v156
  br label %bb32
bb32:
  %v158 = phi float [ 0.0, %bb31 ], [ %v179, %bb36 ]
  %v159 = phi float [ 0.0, %bb31 ], [ %v193, %bb36 ]
  %v160 = phi i64 [ 0, %bb31 ], [ %v194, %bb36 ]
  %v161 = icmp ult i64 %v160, 32
  %v162 = xor i1 %v161, 1
  br i1 %v162, label %bb37, label %bb33
bb33:
  %v163 = add i64 %v157, %v160
  %v164 = icmp ult i64 %v163, %v27
  br i1 %v164, label %bb34, label %bb60
bb34:
  %v165 = extractvalue { ptr, i64 } %v11, 0
  %v166 = getelementptr inbounds i8, ptr %v165, i64 %v163
  %v167 = load i8, ptr %v166, align 1
  %v168 = and i8 %v167, 15
  %v169 = uitofp i8 %v168 to float
  %v170 = mul i64 %v153, 64
  %v171 = add i64 %v124, %v170
  %v172 = add i64 %v171, %v160
  %v173 = extractvalue { ptr, i64 } %v13, 1
  %v174 = icmp ult i64 %v172, %v173
  br i1 %v174, label %bb35, label %bb61
bb35:
  %v175 = extractvalue { ptr, i64 } %v13, 0
  %v176 = getelementptr inbounds float, ptr %v175, i64 %v172
  %v177 = load float, ptr %v176, align 4
  %v178 = fmul contract float %v169, %v177
  %v179 = fadd contract float %v158, %v178
  %v180 = trunc i32 4 to i8
  %v181 = and i8 %v180, 7
  %v182 = lshr i8 %v167, %v181
  %v183 = uitofp i8 %v182 to float
  %v184 = mul i64 %v153, 64
  %v185 = add i64 %v124, %v184
  %v186 = add i64 %v185, 32
  %v187 = add i64 %v186, %v160
  %v188 = icmp ult i64 %v187, %v173
  br i1 %v188, label %bb36, label %bb62
bb36:
  %v189 = extractvalue { ptr, i64 } %v13, 0
  %v190 = getelementptr inbounds float, ptr %v189, i64 %v187
  %v191 = load float, ptr %v190, align 4
  %v192 = fmul contract float %v183, %v191
  %v193 = fadd contract float %v159, %v192
  %v194 = add i64 %v160, 1
  br label %bb32
bb37:
  %v195 = mul i64 %v153, 2
  %v196 = icmp ult i64 %v195, 8
  br i1 %v196, label %bb38, label %bb63
bb38:
  %v197 = getelementptr inbounds [8 x i8], ptr %v18, i32 0, i64 %v195
  %v198 = load i8, ptr %v197, align 1
  %v199 = uitofp i8 %v198 to float
  %v200 = fmul contract float %v42, %v199
  %v201 = fmul contract float %v200, %v158
  %v202 = fadd contract float %v152, %v201
  %v203 = mul i64 %v153, 2
  %v204 = add i64 %v203, 1
  %v205 = icmp ult i64 %v204, 8
  br i1 %v205, label %bb39, label %bb64
bb39:
  %v206 = getelementptr inbounds [8 x i8], ptr %v18, i32 0, i64 %v204
  %v207 = load i8, ptr %v206, align 1
  %v208 = uitofp i8 %v207 to float
  %v209 = fmul contract float %v42, %v208
  %v210 = fmul contract float %v209, %v159
  %v211 = fadd contract float %v202, %v210
  %v212 = add i64 %v153, 1
  br label %bb30
bb40:
  %v213 = add i32 %v21, 1
  br label %bb1
bb41:
  ret float %v20
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
bb47:
  call void @llvm.trap() #0
  unreachable
bb48:
  call void @llvm.trap() #0
  unreachable
bb49:
  call void @llvm.trap() #0
  unreachable
bb50:
  call void @llvm.trap() #0
  unreachable
bb51:
  call void @llvm.trap() #0
  unreachable
bb52:
  call void @llvm.trap() #0
  unreachable
bb53:
  call void @llvm.trap() #0
  unreachable
bb54:
  call void @llvm.trap() #0
  unreachable
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
bb57:
  call void @llvm.trap() #0
  unreachable
bb58:
  call void @llvm.trap() #0
  unreachable
bb59:
  call void @llvm.trap() #0
  unreachable
bb60:
  call void @llvm.trap() #0
  unreachable
bb61:
  call void @llvm.trap() #0
  unreachable
bb62:
  call void @llvm.trap() #0
  unreachable
bb63:
  call void @llvm.trap() #0
  unreachable
bb64:
  call void @llvm.trap() #0
  unreachable
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
  %v24 = alloca [4 x i32], align 4
  %v25 = alloca [4 x i8], align 1
  %v26 = alloca [4 x i8], align 1
  %v27 = alloca [4 x i8], align 1
  %v28 = alloca [4 x i8], align 1
  %v29 = alloca [4 x i8], align 1
  %v30 = alloca [4 x i8], align 1
  %v31 = alloca [4 x i8], align 1
  %v32 = alloca [8 x i8], align 1
  %v33 = alloca [8 x i8], align 1
  %v34 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 0
  store i32 0, ptr %v34, align 4
  %v35 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 1
  store i32 0, ptr %v35, align 4
  %v36 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 2
  store i32 0, ptr %v36, align 4
  %v37 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 3
  store i32 0, ptr %v37, align 4
  %v38 = getelementptr inbounds [4 x i8], ptr %v25, i32 0, i64 0
  store i8 %v12, ptr %v38, align 1
  %v39 = getelementptr inbounds [4 x i8], ptr %v25, i32 0, i64 1
  store i8 %v13, ptr %v39, align 1
  %v40 = getelementptr inbounds [4 x i8], ptr %v25, i32 0, i64 2
  store i8 %v14, ptr %v40, align 1
  %v41 = getelementptr inbounds [4 x i8], ptr %v25, i32 0, i64 3
  store i8 %v15, ptr %v41, align 1
  %v42 = load [4 x i8], ptr %v25, align 1
  %v43 = alloca [4 x i8], align 4
  store [4 x i8] %v42, ptr %v43, align 4
  %v44 = load i32, ptr %v43, align 4
  %v45 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 0
  store i32 %v44, ptr %v45, align 4
  %v46 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 0
  store i8 %v16, ptr %v46, align 1
  %v47 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 1
  store i8 %v17, ptr %v47, align 1
  %v48 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 2
  store i8 %v18, ptr %v48, align 1
  %v49 = getelementptr inbounds [4 x i8], ptr %v26, i32 0, i64 3
  store i8 %v19, ptr %v49, align 1
  %v50 = load [4 x i8], ptr %v26, align 1
  %v51 = alloca [4 x i8], align 4
  store [4 x i8] %v50, ptr %v51, align 4
  %v52 = load i32, ptr %v51, align 4
  %v53 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 1
  store i32 %v52, ptr %v53, align 4
  %v54 = getelementptr inbounds [4 x i8], ptr %v27, i32 0, i64 0
  store i8 %v20, ptr %v54, align 1
  %v55 = getelementptr inbounds [4 x i8], ptr %v27, i32 0, i64 1
  store i8 %v21, ptr %v55, align 1
  %v56 = getelementptr inbounds [4 x i8], ptr %v27, i32 0, i64 2
  store i8 %v22, ptr %v56, align 1
  %v57 = getelementptr inbounds [4 x i8], ptr %v27, i32 0, i64 3
  store i8 %v23, ptr %v57, align 1
  %v58 = load [4 x i8], ptr %v27, align 1
  %v59 = alloca [4 x i8], align 4
  store [4 x i8] %v58, ptr %v59, align 4
  %v60 = load i32, ptr %v59, align 4
  %v61 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 2
  store i32 %v60, ptr %v61, align 4
  %v62 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 2
  %v63 = load i32, ptr %v62, align 4
  %v64 = and i32 4, 31
  %v65 = lshr i32 %v63, %v64
  %v66 = and i32 %v65, 252645135
  %v67 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 1
  %v68 = load i32, ptr %v67, align 4
  %v69 = and i32 6, 31
  %v70 = lshr i32 %v68, %v69
  %v71 = and i32 %v70, 50529027
  %v72 = and i32 4, 31
  %v73 = shl i32 %v71, %v72
  %v74 = or i32 %v66, %v73
  %v75 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 3
  store i32 %v74, ptr %v75, align 4
  %v76 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 1
  %v77 = load i32, ptr %v76, align 4
  %v78 = and i32 %v77, 1061109567
  %v79 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 2
  %v80 = load i32, ptr %v79, align 4
  %v81 = and i32 %v80, 252645135
  %v82 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 0
  %v83 = load i32, ptr %v82, align 4
  %v84 = and i32 6, 31
  %v85 = lshr i32 %v83, %v84
  %v86 = and i32 %v85, 50529027
  %v87 = and i32 4, 31
  %v88 = shl i32 %v86, %v87
  %v89 = or i32 %v81, %v88
  %v90 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 1
  store i32 %v89, ptr %v90, align 4
  %v91 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 2
  store i32 %v78, ptr %v91, align 4
  %v92 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 0
  %v93 = load i32, ptr %v92, align 4
  %v94 = and i32 %v93, 1061109567
  %v95 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 0
  store i32 %v94, ptr %v95, align 4
  %v96 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 0
  %v97 = load i32, ptr %v96, align 4
  %v98 = alloca i32, align 4
  store i32 %v97, ptr %v98, align 4
  %v99 = load [4 x i8], ptr %v98, align 4
  store [4 x i8] %v99, ptr %v28, align 1
  %v100 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 1
  %v101 = load i32, ptr %v100, align 4
  %v102 = alloca i32, align 4
  store i32 %v101, ptr %v102, align 4
  %v103 = load [4 x i8], ptr %v102, align 4
  store [4 x i8] %v103, ptr %v29, align 1
  %v104 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 2
  %v105 = load i32, ptr %v104, align 4
  %v106 = alloca i32, align 4
  store i32 %v105, ptr %v106, align 4
  %v107 = load [4 x i8], ptr %v106, align 4
  store [4 x i8] %v107, ptr %v30, align 1
  %v108 = getelementptr inbounds [4 x i32], ptr %v24, i32 0, i64 3
  %v109 = load i32, ptr %v108, align 4
  %v110 = alloca i32, align 4
  store i32 %v109, ptr %v110, align 4
  %v111 = load [4 x i8], ptr %v110, align 4
  store [4 x i8] %v111, ptr %v31, align 1
  %v112 = getelementptr inbounds [4 x i8], ptr %v28, i32 0, i64 0
  %v113 = load i8, ptr %v112, align 1
  %v114 = getelementptr inbounds [4 x i8], ptr %v28, i32 0, i64 1
  %v115 = load i8, ptr %v114, align 1
  %v116 = getelementptr inbounds [4 x i8], ptr %v28, i32 0, i64 2
  %v117 = load i8, ptr %v116, align 1
  %v118 = getelementptr inbounds [4 x i8], ptr %v28, i32 0, i64 3
  %v119 = load i8, ptr %v118, align 1
  %v120 = getelementptr inbounds [4 x i8], ptr %v29, i32 0, i64 0
  %v121 = load i8, ptr %v120, align 1
  %v122 = getelementptr inbounds [4 x i8], ptr %v29, i32 0, i64 1
  %v123 = load i8, ptr %v122, align 1
  %v124 = getelementptr inbounds [4 x i8], ptr %v29, i32 0, i64 2
  %v125 = load i8, ptr %v124, align 1
  %v126 = getelementptr inbounds [4 x i8], ptr %v29, i32 0, i64 3
  %v127 = load i8, ptr %v126, align 1
  %v128 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 0
  store i8 %v113, ptr %v128, align 1
  %v129 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 1
  store i8 %v115, ptr %v129, align 1
  %v130 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 2
  store i8 %v117, ptr %v130, align 1
  %v131 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 3
  store i8 %v119, ptr %v131, align 1
  %v132 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 4
  store i8 %v121, ptr %v132, align 1
  %v133 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 5
  store i8 %v123, ptr %v133, align 1
  %v134 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 6
  store i8 %v125, ptr %v134, align 1
  %v135 = getelementptr inbounds [8 x i8], ptr %v32, i32 0, i64 7
  store i8 %v127, ptr %v135, align 1
  %v136 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 0
  %v137 = load i8, ptr %v136, align 1
  %v138 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 1
  %v139 = load i8, ptr %v138, align 1
  %v140 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 2
  %v141 = load i8, ptr %v140, align 1
  %v142 = getelementptr inbounds [4 x i8], ptr %v30, i32 0, i64 3
  %v143 = load i8, ptr %v142, align 1
  %v144 = getelementptr inbounds [4 x i8], ptr %v31, i32 0, i64 0
  %v145 = load i8, ptr %v144, align 1
  %v146 = getelementptr inbounds [4 x i8], ptr %v31, i32 0, i64 1
  %v147 = load i8, ptr %v146, align 1
  %v148 = getelementptr inbounds [4 x i8], ptr %v31, i32 0, i64 2
  %v149 = load i8, ptr %v148, align 1
  %v150 = getelementptr inbounds [4 x i8], ptr %v31, i32 0, i64 3
  %v151 = load i8, ptr %v150, align 1
  %v152 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 0
  store i8 %v137, ptr %v152, align 1
  %v153 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 1
  store i8 %v139, ptr %v153, align 1
  %v154 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 2
  store i8 %v141, ptr %v154, align 1
  %v155 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 3
  store i8 %v143, ptr %v155, align 1
  %v156 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 4
  store i8 %v145, ptr %v156, align 1
  %v157 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 5
  store i8 %v147, ptr %v157, align 1
  %v158 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 6
  store i8 %v149, ptr %v158, align 1
  %v159 = getelementptr inbounds [8 x i8], ptr %v33, i32 0, i64 7
  store i8 %v151, ptr %v159, align 1
  %v160 = load [8 x i8], ptr %v32, align 1
  %v161 = load [8 x i8], ptr %v33, align 1
  %v162 = insertvalue { [8 x i8], [8 x i8] } undef, [8 x i8] %v160, 0
  %v163 = insertvalue { [8 x i8], [8 x i8] } %v162, [8 x i8] %v161, 1
  ret { [8 x i8], [8 x i8] } %v163
}

define float @core__f32___impl_f32___clamp(float %v0, float %v1, float %v2) #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi float [ %v0, %entry ]
  %v4 = phi float [ %v1, %entry ]
  %v5 = phi float [ %v2, %entry ]
  %v6 = fcmp ole float %v4, %v5
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb2, label %bb1
bb1:
  %v8 = fcmp olt float %v3, %v4
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb4, label %bb3
bb2:
  call void asm sideeffect "trap;", ""()
  unreachable
bb3:
  br label %bb5
bb4:
  br label %bb5
bb5:
  %v11 = phi float [ %v4, %bb3 ], [ %v3, %bb4 ]
  %v12 = fcmp ogt float %v11, %v5
  %v13 = xor i1 %v12, 1
  br i1 %v13, label %bb7, label %bb6
bb6:
  br label %bb8
bb7:
  br label %bb8
bb8:
  %v14 = phi float [ %v5, %bb6 ], [ %v11, %bb7 ]
  ret float %v14
}

define float @cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk(ptr %v0, i64 %v1, i64 %v2, i64 %v3, i32 %v4, i32 %v5, float %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v8, %entry ]
  %v10 = phi i64 [ %v2, %entry ]
  %v11 = phi i64 [ %v3, %entry ]
  %v12 = phi i32 [ %v4, %entry ]
  %v13 = phi i32 [ %v5, %entry ]
  %v14 = phi float [ %v6, %entry ]
  %v15 = alloca [2 x i8], align 1
  %v16 = alloca [2 x i8], align 1
  %v17 = extractvalue { ptr, i64 } %v9, 0
  %v18 = extractvalue { ptr, i64 } %v9, 1
  %v19 = call i32 @cuda_kernels__oxide_kernels__kernels__q4k_signed4(ptr %v17, i64 %v18, i64 %v10, i64 %v11) #0
  br label %bb1
bb1:
  %v20 = call i32 @llvm.nvvm.idp4a.s.s(i32 %v19, i32 %v12, i32 0) #0
  br label %bb2
bb2:
  %v21 = mul i32 8, %v13
  %v22 = add i32 %v20, %v21
  %v23 = extractvalue { ptr, i64 } %v9, 1
  %v24 = icmp ult i64 %v10, %v23
  %v25 = extractvalue { ptr, i64 } %v9, 0
  %v26 = getelementptr inbounds i8, ptr %v25, i64 %v10
  %v27 = load i8, ptr %v26, align 1
  %v28 = add i64 %v10, 1
  %v29 = icmp ult i64 %v28, %v23
  %v30 = extractvalue { ptr, i64 } %v9, 0
  %v31 = getelementptr inbounds i8, ptr %v30, i64 %v28
  %v32 = load i8, ptr %v31, align 1
  %v33 = getelementptr inbounds [2 x i8], ptr %v15, i32 0, i64 0
  store i8 %v27, ptr %v33, align 1
  %v34 = getelementptr inbounds [2 x i8], ptr %v15, i32 0, i64 1
  store i8 %v32, ptr %v34, align 1
  %v35 = load [2 x i8], ptr %v15, align 1
  %v36 = alloca [2 x i8], align 2
  store [2 x i8] %v35, ptr %v36, align 2
  %v37 = load i16, ptr %v36, align 2
  %v38 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v37) #0
  br label %bb3
bb3:
  %v39 = add i64 %v10, 2
  %v40 = icmp ult i64 %v39, %v23
  %v41 = extractvalue { ptr, i64 } %v9, 0
  %v42 = getelementptr inbounds i8, ptr %v41, i64 %v39
  %v43 = load i8, ptr %v42, align 1
  %v44 = add i64 %v10, 3
  %v45 = icmp ult i64 %v44, %v23
  %v46 = extractvalue { ptr, i64 } %v9, 0
  %v47 = getelementptr inbounds i8, ptr %v46, i64 %v44
  %v48 = load i8, ptr %v47, align 1
  %v49 = getelementptr inbounds [2 x i8], ptr %v16, i32 0, i64 0
  store i8 %v43, ptr %v49, align 1
  %v50 = getelementptr inbounds [2 x i8], ptr %v16, i32 0, i64 1
  store i8 %v48, ptr %v50, align 1
  %v51 = load [2 x i8], ptr %v16, align 1
  %v52 = alloca [2 x i8], align 2
  store [2 x i8] %v51, ptr %v52, align 2
  %v53 = load i16, ptr %v52, align 2
  %v54 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v53) #0
  br label %bb4
bb4:
  %v55 = udiv i64 %v11, 32
  %v56 = extractvalue { ptr, i64 } %v9, 0
  %v57 = extractvalue { ptr, i64 } %v9, 1
  %v58 = call i8 @cuda_kernels__oxide_kernels__kernels__q4k_scale(ptr %v56, i64 %v57, i64 %v10, i64 %v55) #0
  br label %bb5
bb5:
  %v59 = uitofp i8 %v58 to float
  %v60 = fmul contract float %v38, %v59
  %v61 = sitofp i32 %v22 to float
  %v62 = fmul contract float %v60, %v61
  %v63 = extractvalue { ptr, i64 } %v9, 0
  %v64 = extractvalue { ptr, i64 } %v9, 1
  %v65 = call i8 @cuda_kernels__oxide_kernels__kernels__q4k_min(ptr %v63, i64 %v64, i64 %v10, i64 %v55) #0
  br label %bb6
bb6:
  %v66 = uitofp i8 %v65 to float
  %v67 = fmul contract float %v54, %v66
  %v68 = sitofp i32 %v13 to float
  %v69 = fmul contract float %v67, %v68
  %v70 = fsub contract float %v62, %v69
  %v71 = fmul contract float %v14, %v70
  ret float %v71
}

define i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCs5VsnSnoaHeT_12cuda_kernels(i64 %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i64 [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = alloca i64, align 8
  %v5 = alloca i64, align 8
  store i64 %v2, ptr %v4, align 8
  store i64 %v3, ptr %v5, align 8
  %v6 = bitcast ptr %v5 to ptr
  %v7 = bitcast ptr %v4 to ptr
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___lt(ptr %v6, ptr %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v10 = load i64, ptr %v4, align 8
  br label %bb4
bb3:
  %v11 = load i64, ptr %v5, align 8
  br label %bb4
bb4:
  %v12 = phi i64 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i64 %v12
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

define i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCs5VsnSnoaHeT_12cuda_kernels(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = alloca i32, align 4
  %v5 = alloca i32, align 4
  store i32 %v2, ptr %v4, align 4
  store i32 %v3, ptr %v5, align 4
  %v6 = bitcast ptr %v5 to ptr
  %v7 = bitcast ptr %v4 to ptr
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(ptr %v6, ptr %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v10 = load i32, ptr %v5, align 4
  br label %bb4
bb3:
  %v11 = load i32, ptr %v4, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

define float @cuda_kernels__oxide_kernels__kernels__dot_q6k(ptr %v0, i64 %v1, i64 %v2, ptr %v3, i64 %v4, i64 %v5, i32 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v4, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi i64 [ %v2, %entry ]
  %v13 = phi { ptr, i64 } [ %v10, %entry ]
  %v14 = phi i64 [ %v5, %entry ]
  %v15 = phi i32 [ %v6, %entry ]
  %v16 = alloca [2 x i8], align 1
  br label %bb1
bb1:
  %v17 = phi float [ 0.0, %bb0 ], [ %v44, %bb27 ]
  %v18 = phi i32 [ 0, %bb0 ], [ %v224, %bb27 ]
  %v19 = icmp ult i32 %v18, %v15
  %v20 = xor i1 %v19, 1
  br i1 %v20, label %bb28, label %bb2
bb2:
  %v21 = zext i32 %v18 to i64
  %v22 = mul i64 %v21, 210
  %v23 = add i64 %v12, %v22
  %v24 = add i64 %v23, 208
  %v25 = extractvalue { ptr, i64 } %v11, 1
  %v26 = icmp ult i64 %v24, %v25
  br i1 %v26, label %bb3, label %bb29
bb3:
  %v27 = extractvalue { ptr, i64 } %v11, 0
  %v28 = getelementptr inbounds i8, ptr %v27, i64 %v24
  %v29 = load i8, ptr %v28, align 1
  %v30 = add i64 %v23, 209
  %v31 = icmp ult i64 %v30, %v25
  br i1 %v31, label %bb4, label %bb30
bb4:
  %v32 = extractvalue { ptr, i64 } %v11, 0
  %v33 = getelementptr inbounds i8, ptr %v32, i64 %v30
  %v34 = load i8, ptr %v33, align 1
  %v35 = getelementptr inbounds [2 x i8], ptr %v16, i32 0, i64 0
  store i8 %v29, ptr %v35, align 1
  %v36 = getelementptr inbounds [2 x i8], ptr %v16, i32 0, i64 1
  store i8 %v34, ptr %v36, align 1
  %v37 = load [2 x i8], ptr %v16, align 1
  %v38 = alloca [2 x i8], align 2
  store [2 x i8] %v37, ptr %v38, align 2
  %v39 = load i16, ptr %v38, align 2
  %v40 = call float @cuda_kernels__oxide_kernels__f16_to_f32(i16 %v39) #0
  br label %bb5
bb5:
  %v41 = zext i32 %v18 to i64
  %v42 = mul i64 %v41, 256
  %v43 = add i64 %v14, %v42
  br label %bb6
bb6:
  %v44 = phi float [ %v17, %bb5 ], [ %v58, %bb26 ]
  %v45 = phi i64 [ 0, %bb5 ], [ %v223, %bb26 ]
  %v46 = icmp ult i64 %v45, 2
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb27, label %bb7
bb7:
  %v48 = mul i64 %v45, 64
  %v49 = add i64 %v23, %v48
  %v50 = add i64 %v23, 128
  %v51 = mul i64 %v45, 32
  %v52 = add i64 %v50, %v51
  %v53 = add i64 %v23, 192
  %v54 = mul i64 %v45, 8
  %v55 = add i64 %v53, %v54
  %v56 = mul i64 %v45, 128
  %v57 = add i64 %v43, %v56
  br label %bb8
bb8:
  %v58 = phi float [ %v44, %bb7 ], [ %v221, %bb25 ]
  %v59 = phi i64 [ 0, %bb7 ], [ %v222, %bb25 ]
  %v60 = icmp ult i64 %v59, 32
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb26, label %bb9
bb9:
  %v62 = udiv i64 %v59, 16
  %v63 = add i64 %v49, %v59
  %v64 = icmp ult i64 %v63, %v25
  br i1 %v64, label %bb10, label %bb31
bb10:
  %v65 = extractvalue { ptr, i64 } %v11, 0
  %v66 = getelementptr inbounds i8, ptr %v65, i64 %v63
  %v67 = load i8, ptr %v66, align 1
  %v68 = and i8 %v67, 15
  %v69 = zext i8 %v68 to i32
  %v70 = add i64 %v52, %v59
  %v71 = icmp ult i64 %v70, %v25
  br i1 %v71, label %bb11, label %bb32
bb11:
  %v72 = extractvalue { ptr, i64 } %v11, 0
  %v73 = getelementptr inbounds i8, ptr %v72, i64 %v70
  %v74 = load i8, ptr %v73, align 1
  %v75 = and i8 %v74, 3
  %v76 = zext i8 %v75 to i32
  %v77 = and i32 4, 31
  %v78 = shl i32 %v76, %v77
  %v79 = or i32 %v69, %v78
  %v80 = sub i32 %v79, 32
  %v81 = add i64 %v49, %v59
  %v82 = add i64 %v81, 32
  %v83 = icmp ult i64 %v82, %v25
  br i1 %v83, label %bb12, label %bb33
bb12:
  %v84 = extractvalue { ptr, i64 } %v11, 0
  %v85 = getelementptr inbounds i8, ptr %v84, i64 %v82
  %v86 = load i8, ptr %v85, align 1
  %v87 = and i8 %v86, 15
  %v88 = zext i8 %v87 to i32
  %v89 = add i64 %v52, %v59
  %v90 = icmp ult i64 %v89, %v25
  br i1 %v90, label %bb13, label %bb34
bb13:
  %v91 = extractvalue { ptr, i64 } %v11, 0
  %v92 = getelementptr inbounds i8, ptr %v91, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = trunc i32 2 to i8
  %v95 = and i8 %v94, 7
  %v96 = lshr i8 %v93, %v95
  %v97 = and i8 %v96, 3
  %v98 = zext i8 %v97 to i32
  %v99 = and i32 4, 31
  %v100 = shl i32 %v98, %v99
  %v101 = or i32 %v88, %v100
  %v102 = sub i32 %v101, 32
  %v103 = add i64 %v49, %v59
  %v104 = icmp ult i64 %v103, %v25
  br i1 %v104, label %bb14, label %bb35
bb14:
  %v105 = extractvalue { ptr, i64 } %v11, 0
  %v106 = getelementptr inbounds i8, ptr %v105, i64 %v103
  %v107 = load i8, ptr %v106, align 1
  %v108 = trunc i32 4 to i8
  %v109 = and i8 %v108, 7
  %v110 = lshr i8 %v107, %v109
  %v111 = zext i8 %v110 to i32
  %v112 = add i64 %v52, %v59
  %v113 = icmp ult i64 %v112, %v25
  br i1 %v113, label %bb15, label %bb36
bb15:
  %v114 = extractvalue { ptr, i64 } %v11, 0
  %v115 = getelementptr inbounds i8, ptr %v114, i64 %v112
  %v116 = load i8, ptr %v115, align 1
  %v117 = trunc i32 4 to i8
  %v118 = and i8 %v117, 7
  %v119 = lshr i8 %v116, %v118
  %v120 = and i8 %v119, 3
  %v121 = zext i8 %v120 to i32
  %v122 = and i32 4, 31
  %v123 = shl i32 %v121, %v122
  %v124 = or i32 %v111, %v123
  %v125 = sub i32 %v124, 32
  %v126 = add i64 %v49, %v59
  %v127 = add i64 %v126, 32
  %v128 = icmp ult i64 %v127, %v25
  br i1 %v128, label %bb16, label %bb37
bb16:
  %v129 = extractvalue { ptr, i64 } %v11, 0
  %v130 = getelementptr inbounds i8, ptr %v129, i64 %v127
  %v131 = load i8, ptr %v130, align 1
  %v132 = trunc i32 4 to i8
  %v133 = and i8 %v132, 7
  %v134 = lshr i8 %v131, %v133
  %v135 = zext i8 %v134 to i32
  %v136 = add i64 %v52, %v59
  %v137 = icmp ult i64 %v136, %v25
  br i1 %v137, label %bb17, label %bb38
bb17:
  %v138 = extractvalue { ptr, i64 } %v11, 0
  %v139 = getelementptr inbounds i8, ptr %v138, i64 %v136
  %v140 = load i8, ptr %v139, align 1
  %v141 = trunc i32 6 to i8
  %v142 = and i8 %v141, 7
  %v143 = lshr i8 %v140, %v142
  %v144 = and i8 %v143, 3
  %v145 = zext i8 %v144 to i32
  %v146 = and i32 4, 31
  %v147 = shl i32 %v145, %v146
  %v148 = or i32 %v135, %v147
  %v149 = sub i32 %v148, 32
  %v150 = add i64 %v55, %v62
  %v151 = icmp ult i64 %v150, %v25
  br i1 %v151, label %bb18, label %bb39
bb18:
  %v152 = extractvalue { ptr, i64 } %v11, 0
  %v153 = getelementptr inbounds i8, ptr %v152, i64 %v150
  %v154 = load i8, ptr %v153, align 1
  %v155 = bitcast i8 %v154 to i8
  %v156 = sitofp i8 %v155 to float
  %v157 = fmul contract float %v40, %v156
  %v158 = sitofp i32 %v80 to float
  %v159 = fmul contract float %v157, %v158
  %v160 = add i64 %v57, %v59
  %v161 = extractvalue { ptr, i64 } %v13, 1
  %v162 = icmp ult i64 %v160, %v161
  br i1 %v162, label %bb19, label %bb40
bb19:
  %v163 = extractvalue { ptr, i64 } %v13, 0
  %v164 = getelementptr inbounds float, ptr %v163, i64 %v160
  %v165 = load float, ptr %v164, align 4
  %v166 = fmul contract float %v159, %v165
  %v167 = fadd contract float %v58, %v166
  %v168 = add i64 %v150, 2
  %v169 = icmp ult i64 %v168, %v25
  br i1 %v169, label %bb20, label %bb41
bb20:
  %v170 = extractvalue { ptr, i64 } %v11, 0
  %v171 = getelementptr inbounds i8, ptr %v170, i64 %v168
  %v172 = load i8, ptr %v171, align 1
  %v173 = bitcast i8 %v172 to i8
  %v174 = sitofp i8 %v173 to float
  %v175 = fmul contract float %v40, %v174
  %v176 = sitofp i32 %v102 to float
  %v177 = fmul contract float %v175, %v176
  %v178 = add i64 %v57, %v59
  %v179 = add i64 %v178, 32
  %v180 = icmp ult i64 %v179, %v161
  br i1 %v180, label %bb21, label %bb42
bb21:
  %v181 = extractvalue { ptr, i64 } %v13, 0
  %v182 = getelementptr inbounds float, ptr %v181, i64 %v179
  %v183 = load float, ptr %v182, align 4
  %v184 = fmul contract float %v177, %v183
  %v185 = fadd contract float %v167, %v184
  %v186 = add i64 %v150, 4
  %v187 = icmp ult i64 %v186, %v25
  br i1 %v187, label %bb22, label %bb43
bb22:
  %v188 = extractvalue { ptr, i64 } %v11, 0
  %v189 = getelementptr inbounds i8, ptr %v188, i64 %v186
  %v190 = load i8, ptr %v189, align 1
  %v191 = bitcast i8 %v190 to i8
  %v192 = sitofp i8 %v191 to float
  %v193 = fmul contract float %v40, %v192
  %v194 = sitofp i32 %v125 to float
  %v195 = fmul contract float %v193, %v194
  %v196 = add i64 %v57, %v59
  %v197 = add i64 %v196, 64
  %v198 = icmp ult i64 %v197, %v161
  br i1 %v198, label %bb23, label %bb44
bb23:
  %v199 = extractvalue { ptr, i64 } %v13, 0
  %v200 = getelementptr inbounds float, ptr %v199, i64 %v197
  %v201 = load float, ptr %v200, align 4
  %v202 = fmul contract float %v195, %v201
  %v203 = fadd contract float %v185, %v202
  %v204 = add i64 %v150, 6
  %v205 = icmp ult i64 %v204, %v25
  br i1 %v205, label %bb24, label %bb45
bb24:
  %v206 = extractvalue { ptr, i64 } %v11, 0
  %v207 = getelementptr inbounds i8, ptr %v206, i64 %v204
  %v208 = load i8, ptr %v207, align 1
  %v209 = bitcast i8 %v208 to i8
  %v210 = sitofp i8 %v209 to float
  %v211 = fmul contract float %v40, %v210
  %v212 = sitofp i32 %v149 to float
  %v213 = fmul contract float %v211, %v212
  %v214 = add i64 %v57, %v59
  %v215 = add i64 %v214, 96
  %v216 = icmp ult i64 %v215, %v161
  br i1 %v216, label %bb25, label %bb46
bb25:
  %v217 = extractvalue { ptr, i64 } %v13, 0
  %v218 = getelementptr inbounds float, ptr %v217, i64 %v215
  %v219 = load float, ptr %v218, align 4
  %v220 = fmul contract float %v213, %v219
  %v221 = fadd contract float %v203, %v220
  %v222 = add i64 %v59, 1
  br label %bb8
bb26:
  %v223 = add i64 %v45, 1
  br label %bb6
bb27:
  %v224 = add i32 %v18, 1
  br label %bb1
bb28:
  ret float %v17
bb29:
  call void @llvm.trap() #0
  unreachable
bb30:
  call void @llvm.trap() #0
  unreachable
bb31:
  call void @llvm.trap() #0
  unreachable
bb32:
  call void @llvm.trap() #0
  unreachable
bb33:
  call void @llvm.trap() #0
  unreachable
bb34:
  call void @llvm.trap() #0
  unreachable
bb35:
  call void @llvm.trap() #0
  unreachable
bb36:
  call void @llvm.trap() #0
  unreachable
bb37:
  call void @llvm.trap() #0
  unreachable
bb38:
  call void @llvm.trap() #0
  unreachable
bb39:
  call void @llvm.trap() #0
  unreachable
bb40:
  call void @llvm.trap() #0
  unreachable
bb41:
  call void @llvm.trap() #0
  unreachable
bb42:
  call void @llvm.trap() #0
  unreachable
bb43:
  call void @llvm.trap() #0
  unreachable
bb44:
  call void @llvm.trap() #0
  unreachable
bb45:
  call void @llvm.trap() #0
  unreachable
bb46:
  call void @llvm.trap() #0
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCs52pM6xjztaY_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECs5VsnSnoaHeT_12cuda_kernels(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1
  %v3 = xor i1 %v2, 1
  br i1 %v3, label %bb2, label %bb1
bb1:
  br label %bb8
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb3
bb3:
  %v5 = icmp eq i32 %v4, 1
  br i1 %v5, label %bb4, label %bb5
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  br label %bb6
bb5:
  br label %bb7
bb6:
  %v7 = icmp eq i32 %v6, 1
  br label %bb7
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1
  br i1 %v10, label %bb9, label %bb10
bb9:
  %v11 = icmp eq i8 0, 2
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb12
bb12:
  %v14 = icmp eq i32 %v13, 1
  br i1 %v14, label %bb13, label %bb14
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  br label %bb15
bb14:
  br label %bb16
bb15:
  %v16 = icmp eq i32 %v15, 1
  br label %bb16
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1
  br i1 %v19, label %bb19, label %bb18
bb18:
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i32, ptr %v2, align 4
  %v5 = load i32, ptr %v3, align 4
  %v6 = icmp ult i32 %v4, %v5
  ret i1 %v6
}

define i8 @cuda_kernels__oxide_kernels__kernels__q6k_quant(ptr %v0, i64 %v1, i64 %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = udiv i64 %v8, 128
  %v10 = urem i64 %v8, 128
  %v11 = urem i64 %v10, 32
  %v12 = udiv i64 %v10, 32
  %v13 = mul i64 %v9, 64
  %v14 = add i64 %v7, %v13
  %v15 = add i64 %v7, 128
  %v16 = mul i64 %v9, 32
  %v17 = add i64 %v15, %v16
  %v18 = mul i64 %v12, 2
  %v19 = trunc i64 %v18 to i32
  %v20 = icmp eq i64 %v12, 1
  br i1 %v20, label %bb2, label %bb1
bb1:
  %v21 = icmp eq i64 %v12, 3
  br i1 %v21, label %bb2, label %bb3
bb2:
  br label %bb4
bb3:
  br label %bb4
bb4:
  %v22 = phi i64 [ 32, %bb2 ], [ 0, %bb3 ]
  %v23 = add i64 %v11, %v22
  %v24 = add i64 %v14, %v23
  %v25 = extractvalue { ptr, i64 } %v6, 1
  %v26 = icmp ult i64 %v24, %v25
  %v27 = extractvalue { ptr, i64 } %v6, 0
  %v28 = getelementptr inbounds i8, ptr %v27, i64 %v24
  %v29 = load i8, ptr %v28, align 1
  %v30 = icmp ult i64 %v12, 2
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb6, label %bb5
bb5:
  %v32 = and i8 %v29, 15
  br label %bb7
bb6:
  %v33 = trunc i32 4 to i8
  %v34 = and i8 %v33, 7
  %v35 = lshr i8 %v29, %v34
  br label %bb7
bb7:
  %v36 = phi i8 [ %v32, %bb5 ], [ %v35, %bb6 ]
  %v37 = zext i8 %v36 to i32
  %v38 = add i64 %v17, %v11
  %v39 = icmp ult i64 %v38, %v25
  %v40 = extractvalue { ptr, i64 } %v6, 0
  %v41 = getelementptr inbounds i8, ptr %v40, i64 %v38
  %v42 = load i8, ptr %v41, align 1
  %v43 = trunc i32 %v19 to i8
  %v44 = and i8 %v43, 7
  %v45 = lshr i8 %v42, %v44
  %v46 = and i8 %v45, 3
  %v47 = zext i8 %v46 to i32
  %v48 = and i32 4, 31
  %v49 = shl i32 %v47, %v48
  %v50 = sub i32 %v49, 32
  %v51 = or i32 %v37, %v50
  %v52 = trunc i32 %v51 to i8
  ret i8 %v52
}

define i32 @cuda_kernels__oxide_kernels__kernels__q4k_signed4(ptr %v0, i64 %v1, i64 %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = alloca { [0 x i32], { { i32 } } }, align 4
  %v10 = alloca { [0 x i32], { { i32 } } }, align 4
  %v11 = urem i64 %v8, 32
  %v12 = udiv i64 %v8, 64
  %v13 = add i64 %v7, 16
  %v14 = mul i64 %v12, 32
  %v15 = add i64 %v13, %v14
  %v16 = add i64 %v15, %v11
  %v17 = extractvalue { ptr, i64 } %v6, 0
  %v18 = getelementptr inbounds i8, ptr %v17, i64 %v16
  store { [0 x i32], { { i32 } } } undef, ptr %v9, align 4
  %v19 = bitcast ptr %v9 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %v19, ptr %v18, i64 4, i1 0) #0
  %v21 = load { [0 x i32], { { i32 } } }, ptr %v9, align 4
  store { [0 x i32], { { i32 } } } %v21, ptr %v10, align 4
  %v22 = getelementptr inbounds i8, ptr %v10, i32 0
  %v23 = bitcast ptr %v22 to ptr
  %v24 = load i32, ptr %v23, align 4
  %v25 = urem i64 %v8, 64
  %v26 = icmp ult i64 %v25, 32
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb2, label %bb1
bb1:
  %v28 = and i32 %v24, 252645135
  br label %bb3
bb2:
  %v29 = and i32 4, 31
  %v30 = lshr i32 %v24, %v29
  %v31 = and i32 %v30, 252645135
  br label %bb3
bb3:
  %v32 = phi i32 [ %v28, %bb1 ], [ %v31, %bb2 ]
  %v33 = xor i32 %v32, 134744072
  %v34 = and i32 %v33, 134744072
  %v35 = and i32 1, 31
  %v36 = shl i32 %v34, %v35
  %v37 = or i32 %v33, %v36
  %v38 = and i32 2, 31
  %v39 = shl i32 %v34, %v38
  %v40 = or i32 %v37, %v39
  %v41 = and i32 3, 31
  %v42 = shl i32 %v34, %v41
  %v43 = or i32 %v40, %v42
  %v44 = and i32 4, 31
  %v45 = shl i32 %v34, %v44
  %v46 = or i32 %v43, %v45
  ret i32 %v46
}

define i8 @cuda_kernels__oxide_kernels__kernels__q4k_scale(ptr %v0, i64 %v1, i64 %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = icmp ult i64 %v8, 4
  %v10 = xor i1 %v9, 1
  br i1 %v10, label %bb2, label %bb1
bb1:
  %v11 = add i64 %v7, 4
  %v12 = add i64 %v11, %v8
  %v13 = extractvalue { ptr, i64 } %v6, 1
  %v14 = icmp ult i64 %v12, %v13
  %v15 = extractvalue { ptr, i64 } %v6, 0
  %v16 = getelementptr inbounds i8, ptr %v15, i64 %v12
  %v17 = load i8, ptr %v16, align 1
  %v18 = and i8 %v17, 63
  br label %bb3
bb2:
  %v19 = sub i64 %v8, 4
  %v20 = add i64 %v7, 12
  %v21 = add i64 %v20, %v19
  %v22 = extractvalue { ptr, i64 } %v6, 1
  %v23 = icmp ult i64 %v21, %v22
  %v24 = extractvalue { ptr, i64 } %v6, 0
  %v25 = getelementptr inbounds i8, ptr %v24, i64 %v21
  %v26 = load i8, ptr %v25, align 1
  %v27 = and i8 %v26, 15
  %v28 = add i64 %v7, 4
  %v29 = add i64 %v28, %v19
  %v30 = icmp ult i64 %v29, %v22
  %v31 = extractvalue { ptr, i64 } %v6, 0
  %v32 = getelementptr inbounds i8, ptr %v31, i64 %v29
  %v33 = load i8, ptr %v32, align 1
  %v34 = trunc i32 6 to i8
  %v35 = and i8 %v34, 7
  %v36 = lshr i8 %v33, %v35
  %v37 = trunc i32 4 to i8
  %v38 = and i8 %v37, 7
  %v39 = shl i8 %v36, %v38
  %v40 = or i8 %v27, %v39
  br label %bb3
bb3:
  %v41 = phi i8 [ %v18, %bb1 ], [ %v40, %bb2 ]
  ret i8 %v41
}

define i8 @cuda_kernels__oxide_kernels__kernels__q4k_min(ptr %v0, i64 %v1, i64 %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = icmp ult i64 %v8, 4
  %v10 = xor i1 %v9, 1
  br i1 %v10, label %bb2, label %bb1
bb1:
  %v11 = add i64 %v7, 8
  %v12 = add i64 %v11, %v8
  %v13 = extractvalue { ptr, i64 } %v6, 1
  %v14 = icmp ult i64 %v12, %v13
  %v15 = extractvalue { ptr, i64 } %v6, 0
  %v16 = getelementptr inbounds i8, ptr %v15, i64 %v12
  %v17 = load i8, ptr %v16, align 1
  %v18 = and i8 %v17, 63
  br label %bb3
bb2:
  %v19 = sub i64 %v8, 4
  %v20 = add i64 %v7, 12
  %v21 = add i64 %v20, %v19
  %v22 = extractvalue { ptr, i64 } %v6, 1
  %v23 = icmp ult i64 %v21, %v22
  %v24 = extractvalue { ptr, i64 } %v6, 0
  %v25 = getelementptr inbounds i8, ptr %v24, i64 %v21
  %v26 = load i8, ptr %v25, align 1
  %v27 = trunc i32 4 to i8
  %v28 = and i8 %v27, 7
  %v29 = lshr i8 %v26, %v28
  %v30 = add i64 %v7, 8
  %v31 = add i64 %v30, %v19
  %v32 = icmp ult i64 %v31, %v22
  %v33 = extractvalue { ptr, i64 } %v6, 0
  %v34 = getelementptr inbounds i8, ptr %v33, i64 %v31
  %v35 = load i8, ptr %v34, align 1
  %v36 = trunc i32 6 to i8
  %v37 = and i8 %v36, 7
  %v38 = lshr i8 %v35, %v37
  %v39 = trunc i32 4 to i8
  %v40 = and i8 %v39, 7
  %v41 = shl i8 %v38, %v40
  %v42 = or i8 %v29, %v41
  br label %bb3
bb3:
  %v43 = phi i8 [ %v18, %bb1 ], [ %v42, %bb2 ]
  ret i8 %v43
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___lt(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i64, ptr %v2, align 8
  %v5 = load i64, ptr %v3, align 8
  %v6 = icmp ult i64 %v4, %v5
  ret i1 %v6
}


@llvm.used = appending global [61 x ptr] [ptr @add_f32, ptr @add_in_place_f32, ptr @add_inplace_f32, ptr @argmax_token, ptr @attention_canvas_heads, ptr @attention_canvas_paged_heads, ptr @attention_fa2_decode_paged, ptr @attention_fa2_prefill_paged, ptr @attention_fa3_decode_paged, ptr @attention_heads, ptr @attention_paged_batch_heads, ptr @attention_paged_heads, ptr @attention_paged_warp, ptr @embedding_f32, ptr @embedding_f32_rows, ptr @embedding_q4k_row, ptr @embedding_q4k_rows, ptr @embedding_q6k_row, ptr @embedding_q6k_rows, ptr @embedding_q8_0_row, ptr @embedding_q8_0_rows, ptr @fill_u32, ptr @kv_write_batch, ptr @kv_write_row, ptr @moe_count_assignments, ptr @moe_prefix_offsets, ptr @moe_q4k_project, ptr @moe_q4k_project_warp, ptr @moe_q5_0_project, ptr @moe_q5_0_project_warp, ptr @moe_q6k_project, ptr @moe_q8_0_project, ptr @moe_q8_0_project_warp, ptr @moe_route_topk, ptr @moe_scatter_assignments, ptr @moe_weighted_reduce, ptr @mul_f32, ptr @q4k_gate_up_swiglu_multiwarp, ptr @q4k_gemm_warp, ptr @q4k_gemv_row, ptr @q4k_gemv_row_tiled, ptr @q4k_q8_gemv_multiwarp, ptr @q4k_q8_gemv_warp4, ptr @q5_0_gemm_element, ptr @q5_0_gemm_warp, ptr @q6k_gemm_warp, ptr @q6k_gemv_row, ptr @q6k_gemv_warp4, ptr @q6k_q8_gemv_multiwarp, ptr @q6k_q8_gemv_warp4, ptr @q8_0_gemm_element, ptr @q8_0_gemm_warp, ptr @quantize_q8_32, ptr @rmsnorm_group, ptr @rope, ptr @rope_batch, ptr @scale_f32, ptr @shortconv_mix, ptr @shortconv_mix_rows, ptr @silu_gate, ptr @weighted_embedding_q6k_topk], section "llvm.metadata"

attributes #0 = { convergent }
