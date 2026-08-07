; ModuleID = '/mnt/c/Users/oriel/Documents/Projects/FeLLM/plugins/cuda_kernels/cuda_kernels.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_16 = local_unnamed_addr addrspace(3) global [4 x float] zeroinitializer, align 4
@__shared_mem_15 = local_unnamed_addr addrspace(3) global [4 x float] zeroinitializer, align 4
@__shared_mem_14 = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_13 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_12 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_11 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_10 = local_unnamed_addr addrspace(3) global [8 x float] zeroinitializer, align 4
@__shared_mem_9 = local_unnamed_addr addrspace(3) global [256 x i32] zeroinitializer, align 4
@__shared_mem_8 = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_7 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_6 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_5 = local_unnamed_addr addrspace(3) global [8 x float] zeroinitializer, align 4
@__shared_mem_4 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_3 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_2 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_1 = local_unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 4
@__shared_mem_0 = local_unnamed_addr addrspace(3) global [128 x float] zeroinitializer, align 4
@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@__cudart_i2opi_f = internal unnamed_addr addrspace(1) constant [6 x i32] [i32 1011060801, i32 -614296167, i32 -181084736, i32 -64530479, i32 1313084713, i32 -1560706194], align 4
@llvm.used = appending global [51 x ptr] [ptr @add_f32, ptr @add_in_place_f32, ptr @add_inplace_f32, ptr @argmax_token, ptr @attention_canvas_heads, ptr @attention_canvas_paged_heads, ptr @attention_heads, ptr @attention_paged_heads, ptr @attention_paged_warp, ptr @embedding_f32, ptr @embedding_q4k_row, ptr @embedding_q6k_row, ptr @embedding_q6k_rows, ptr @embedding_q8_0_row, ptr @fill_u32, ptr @kv_write_row, ptr @moe_count_assignments, ptr @moe_prefix_offsets, ptr @moe_q4k_project, ptr @moe_q4k_project_warp, ptr @moe_q5_0_project, ptr @moe_q5_0_project_warp, ptr @moe_q6k_project, ptr @moe_q8_0_project, ptr @moe_q8_0_project_warp, ptr @moe_route_topk, ptr @moe_scatter_assignments, ptr @moe_weighted_reduce, ptr @mul_f32, ptr @q4k_gate_up_swiglu_multiwarp, ptr @q4k_gemm_warp, ptr @q4k_gemv_row, ptr @q4k_gemv_row_tiled, ptr @q4k_q8_gemv_multiwarp, ptr @q4k_q8_gemv_warp4, ptr @q5_0_gemm_element, ptr @q5_0_gemm_warp, ptr @q6k_gemm_warp, ptr @q6k_gemv_row, ptr @q6k_gemv_warp4, ptr @q6k_q8_gemv_multiwarp, ptr @q6k_q8_gemv_warp4, ptr @q8_0_gemm_element, ptr @q8_0_gemm_warp, ptr @quantize_q8_32, ptr @rmsnorm_group, ptr @rope, ptr @scale_f32, ptr @shortconv_mix, ptr @silu_gate, ptr @weighted_embedding_q6k_topk], section "llvm.metadata"

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @add_f32(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr writeonly captures(address_is_null) %v4, i64 %v5) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v32 = icmp ult i64 %v22.i, %v5
  %or.cond.not = select i1 %.v18.i, i1 %v32, i1 false
  %v35 = getelementptr inbounds float, ptr %v4, i64 %v22.i
  %v461 = icmp ne ptr %v4, null
  %v46 = select i1 %or.cond.not, i1 %v461, i1 false
  br i1 %v46, label %bb2, label %bb6

bb2:                                              ; preds = %entry
  %v21 = icmp ult i64 %v22.i, %v1
  br i1 %v21, label %bb3, label %bb14

bb3:                                              ; preds = %bb2
  %v26 = icmp ult i64 %v22.i, %v3
  br i1 %v26, label %bb4, label %bb15

bb4:                                              ; preds = %bb3
  %v23 = getelementptr inbounds float, ptr %v0, i64 %v22.i
  %v24 = load float, ptr %v23, align 4
  %v28 = getelementptr inbounds float, ptr %v2, i64 %v22.i
  %v29 = load float, ptr %v28, align 4
  %v30 = fadd contract float %v24, %v29
  store float %v30, ptr %v35, align 4
  br label %bb6

bb6:                                              ; preds = %entry, %bb4
  ret void

bb14:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable

bb15:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @add_in_place_f32(ptr readonly captures(none) %v0, i64 %v1, ptr captures(address_is_null) %v2, i64 %v3) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v23 = icmp ult i64 %v22.i, %v3
  %or.cond.not = select i1 %.v18.i, i1 %v23, i1 false
  %v26 = getelementptr inbounds float, ptr %v2, i64 %v22.i
  %v371 = icmp ne ptr %v2, null
  %v37 = select i1 %or.cond.not, i1 %v371, i1 false
  br i1 %v37, label %bb2, label %bb5

bb2:                                              ; preds = %entry
  %v16 = icmp ult i64 %v22.i, %v1
  br i1 %v16, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  %v18 = getelementptr inbounds float, ptr %v0, i64 %v22.i
  %v19 = load float, ptr %v18, align 4
  %v20 = load float, ptr %v26, align 4
  %v21 = fadd contract float %v19, %v20
  store float %v21, ptr %v26, align 4
  br label %bb5

bb5:                                              ; preds = %entry, %bb3
  ret void

bb13:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @add_inplace_f32(ptr readonly captures(none) %v0, i64 %v1, ptr captures(address_is_null) %v2, i64 %v3) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v23 = icmp ult i64 %v22.i, %v3
  %or.cond.not = select i1 %.v18.i, i1 %v23, i1 false
  %v26 = getelementptr inbounds float, ptr %v2, i64 %v22.i
  %v371 = icmp ne ptr %v2, null
  %v37 = select i1 %or.cond.not, i1 %v371, i1 false
  br i1 %v37, label %bb2, label %bb5

bb2:                                              ; preds = %entry
  %v16 = icmp ult i64 %v22.i, %v1
  br i1 %v16, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  %v18 = getelementptr inbounds float, ptr %v0, i64 %v22.i
  %v19 = load float, ptr %v18, align 4
  %v20 = load float, ptr %v26, align 4
  %v21 = fadd contract float %v19, %v20
  store float %v21, ptr %v26, align 4
  br label %bb5

bb5:                                              ; preds = %entry, %bb3
  ret void

bb13:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent norecurse nounwind
define ptx_kernel void @argmax_token(ptr readonly captures(none) %v0, i64 %v1, i32 %v2, ptr writeonly captures(none) %v3, i64 %v4) #1 {
entry:
  %v12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v13 = zext nneg i32 %v12 to i64
  %v17 = zext i32 %v2 to i64
  %v18.not3 = icmp ult i32 %v12, %v2
  br i1 %v18.not3, label %bb3, label %bb12

bb3:                                              ; preds = %entry, %bb11
  %v166 = phi i64 [ %v35, %bb11 ], [ %v13, %entry ]
  %v155 = phi i32 [ %v34, %bb11 ], [ 0, %entry ]
  %v144 = phi float [ %v33, %bb11 ], [ 0xFFF0000000000000, %entry ]
  %v23 = getelementptr inbounds nuw float, ptr %v0, i64 %v166
  %v24 = load float, ptr %v23, align 4
  %v25 = fcmp ule float %v24, %v144
  br i1 %v25, label %bb5, label %bb3.bb8_crit_edge

bb3.bb8_crit_edge:                                ; preds = %bb3
  %.pre = trunc nuw i64 %v166 to i32
  br label %bb11

bb5:                                              ; preds = %bb3
  %v27 = fcmp une float %v24, %v144
  %v29 = trunc nuw i64 %v166 to i32
  %v30 = icmp ule i32 %v155, %v29
  %or.cond1 = select i1 %v27, i1 true, i1 %v30
  %spec.select = select i1 %or.cond1, float %v144, float %v24
  %spec.select9 = select i1 %or.cond1, i32 %v155, i32 %v29
  br label %bb11

bb11:                                             ; preds = %bb5, %bb3.bb8_crit_edge
  %v33 = phi float [ %spec.select, %bb5 ], [ %v24, %bb3.bb8_crit_edge ]
  %v34 = phi i32 [ %spec.select9, %bb5 ], [ %.pre, %bb3.bb8_crit_edge ]
  %v35 = add nuw nsw i64 %v166, 256
  %v18.not = icmp samesign ult i64 %v35, %v17
  br i1 %v18.not, label %bb3, label %bb12

bb12:                                             ; preds = %bb11, %entry
  %v14.lcssa = phi float [ 0xFFF0000000000000, %entry ], [ %v33, %bb11 ]
  %v15.lcssa = phi i32 [ 0, %entry ], [ %v34, %bb11 ]
  %v36 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v13
  store float %v14.lcssa, ptr addrspace(3) %v36, align 4
  %v37 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v13
  store i32 %v15.lcssa, ptr addrspace(3) %v37, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not = icmp samesign ult i32 %v12, 128
  br i1 %v41.not, label %bb18, label %bb30

bb18:                                             ; preds = %bb12
  %v44 = or disjoint i64 %v13, 128
  %v45 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44
  %v46 = load float, ptr addrspace(3) %v45, align 4
  %v49 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44
  %v50 = load i32, ptr addrspace(3) %v49, align 4
  %v53 = load float, ptr addrspace(3) %v36, align 4
  %v57 = fcmp ule float %v46, %v53
  br i1 %v57, label %bb23, label %bb25

bb23:                                             ; preds = %bb18
  %v56 = load i32, ptr addrspace(3) %v37, align 4
  %v59 = fcmp une float %v46, %v53
  %v61 = icmp uge i32 %v50, %v56
  %or.cond = select i1 %v59, i1 true, i1 %v61
  br i1 %or.cond, label %bb30, label %bb25

bb25:                                             ; preds = %bb23, %bb18
  store float %v46, ptr addrspace(3) %v36, align 4
  store i32 %v50, ptr addrspace(3) %v37, align 4
  br label %bb30

bb30:                                             ; preds = %bb12, %bb23, %bb25
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.1 = icmp samesign ult i32 %v12, 64
  br i1 %v41.not.1, label %bb18.1, label %bb30.1

bb18.1:                                           ; preds = %bb30
  %v44.1 = or disjoint i64 %v13, 64
  %v45.1 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44.1
  %v46.1 = load float, ptr addrspace(3) %v45.1, align 4
  %v49.1 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44.1
  %v50.1 = load i32, ptr addrspace(3) %v49.1, align 4
  %v53.1 = load float, ptr addrspace(3) %v36, align 4
  %v57.1 = fcmp ule float %v46.1, %v53.1
  br i1 %v57.1, label %bb23.1, label %bb25.1

bb23.1:                                           ; preds = %bb18.1
  %v56.1 = load i32, ptr addrspace(3) %v37, align 4
  %v59.1 = fcmp une float %v46.1, %v53.1
  %v61.1 = icmp uge i32 %v50.1, %v56.1
  %or.cond.1 = select i1 %v59.1, i1 true, i1 %v61.1
  br i1 %or.cond.1, label %bb30.1, label %bb25.1

bb25.1:                                           ; preds = %bb23.1, %bb18.1
  store float %v46.1, ptr addrspace(3) %v36, align 4
  store i32 %v50.1, ptr addrspace(3) %v37, align 4
  br label %bb30.1

bb30.1:                                           ; preds = %bb25.1, %bb23.1, %bb30
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.2 = icmp samesign ult i32 %v12, 32
  br i1 %v41.not.2, label %bb18.2, label %bb30.2

bb18.2:                                           ; preds = %bb30.1
  %v44.2 = or disjoint i64 %v13, 32
  %v45.2 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44.2
  %v46.2 = load float, ptr addrspace(3) %v45.2, align 4
  %v49.2 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44.2
  %v50.2 = load i32, ptr addrspace(3) %v49.2, align 4
  %v53.2 = load float, ptr addrspace(3) %v36, align 4
  %v57.2 = fcmp ule float %v46.2, %v53.2
  br i1 %v57.2, label %bb23.2, label %bb25.2

bb23.2:                                           ; preds = %bb18.2
  %v56.2 = load i32, ptr addrspace(3) %v37, align 4
  %v59.2 = fcmp une float %v46.2, %v53.2
  %v61.2 = icmp uge i32 %v50.2, %v56.2
  %or.cond.2 = select i1 %v59.2, i1 true, i1 %v61.2
  br i1 %or.cond.2, label %bb30.2, label %bb25.2

bb25.2:                                           ; preds = %bb23.2, %bb18.2
  store float %v46.2, ptr addrspace(3) %v36, align 4
  store i32 %v50.2, ptr addrspace(3) %v37, align 4
  br label %bb30.2

bb30.2:                                           ; preds = %bb25.2, %bb23.2, %bb30.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.3 = icmp samesign ult i32 %v12, 16
  br i1 %v41.not.3, label %bb18.3, label %bb30.3

bb18.3:                                           ; preds = %bb30.2
  %v44.3 = or disjoint i64 %v13, 16
  %v45.3 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44.3
  %v46.3 = load float, ptr addrspace(3) %v45.3, align 4
  %v49.3 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44.3
  %v50.3 = load i32, ptr addrspace(3) %v49.3, align 4
  %v53.3 = load float, ptr addrspace(3) %v36, align 4
  %v57.3 = fcmp ule float %v46.3, %v53.3
  br i1 %v57.3, label %bb23.3, label %bb25.3

bb23.3:                                           ; preds = %bb18.3
  %v56.3 = load i32, ptr addrspace(3) %v37, align 4
  %v59.3 = fcmp une float %v46.3, %v53.3
  %v61.3 = icmp uge i32 %v50.3, %v56.3
  %or.cond.3 = select i1 %v59.3, i1 true, i1 %v61.3
  br i1 %or.cond.3, label %bb30.3, label %bb25.3

bb25.3:                                           ; preds = %bb23.3, %bb18.3
  store float %v46.3, ptr addrspace(3) %v36, align 4
  store i32 %v50.3, ptr addrspace(3) %v37, align 4
  br label %bb30.3

bb30.3:                                           ; preds = %bb25.3, %bb23.3, %bb30.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.4 = icmp samesign ult i32 %v12, 8
  br i1 %v41.not.4, label %bb18.4, label %bb30.4

bb18.4:                                           ; preds = %bb30.3
  %v44.4 = or disjoint i64 %v13, 8
  %v45.4 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44.4
  %v46.4 = load float, ptr addrspace(3) %v45.4, align 4
  %v49.4 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44.4
  %v50.4 = load i32, ptr addrspace(3) %v49.4, align 4
  %v53.4 = load float, ptr addrspace(3) %v36, align 4
  %v57.4 = fcmp ule float %v46.4, %v53.4
  br i1 %v57.4, label %bb23.4, label %bb25.4

bb23.4:                                           ; preds = %bb18.4
  %v56.4 = load i32, ptr addrspace(3) %v37, align 4
  %v59.4 = fcmp une float %v46.4, %v53.4
  %v61.4 = icmp uge i32 %v50.4, %v56.4
  %or.cond.4 = select i1 %v59.4, i1 true, i1 %v61.4
  br i1 %or.cond.4, label %bb30.4, label %bb25.4

bb25.4:                                           ; preds = %bb23.4, %bb18.4
  store float %v46.4, ptr addrspace(3) %v36, align 4
  store i32 %v50.4, ptr addrspace(3) %v37, align 4
  br label %bb30.4

bb30.4:                                           ; preds = %bb25.4, %bb23.4, %bb30.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.5 = icmp samesign ult i32 %v12, 4
  br i1 %v41.not.5, label %bb18.5, label %bb30.5

bb18.5:                                           ; preds = %bb30.4
  %v44.5 = or disjoint i64 %v13, 4
  %v45.5 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44.5
  %v46.5 = load float, ptr addrspace(3) %v45.5, align 4
  %v49.5 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44.5
  %v50.5 = load i32, ptr addrspace(3) %v49.5, align 4
  %v53.5 = load float, ptr addrspace(3) %v36, align 4
  %v57.5 = fcmp ule float %v46.5, %v53.5
  br i1 %v57.5, label %bb23.5, label %bb25.5

bb23.5:                                           ; preds = %bb18.5
  %v56.5 = load i32, ptr addrspace(3) %v37, align 4
  %v59.5 = fcmp une float %v46.5, %v53.5
  %v61.5 = icmp uge i32 %v50.5, %v56.5
  %or.cond.5 = select i1 %v59.5, i1 true, i1 %v61.5
  br i1 %or.cond.5, label %bb30.5, label %bb25.5

bb25.5:                                           ; preds = %bb23.5, %bb18.5
  store float %v46.5, ptr addrspace(3) %v36, align 4
  store i32 %v50.5, ptr addrspace(3) %v37, align 4
  br label %bb30.5

bb30.5:                                           ; preds = %bb25.5, %bb23.5, %bb30.4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.6 = icmp samesign ult i32 %v12, 2
  br i1 %v41.not.6, label %bb18.6, label %bb30.6

bb18.6:                                           ; preds = %bb30.5
  %v44.6 = or disjoint i64 %v13, 2
  %v45.6 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_8, i64 %v44.6
  %v46.6 = load float, ptr addrspace(3) %v45.6, align 4
  %v49.6 = getelementptr inbounds nuw i32, ptr addrspace(3) @__shared_mem_9, i64 %v44.6
  %v50.6 = load i32, ptr addrspace(3) %v49.6, align 4
  %v53.6 = load float, ptr addrspace(3) %v36, align 4
  %v57.6 = fcmp ule float %v46.6, %v53.6
  br i1 %v57.6, label %bb23.6, label %bb25.6

bb23.6:                                           ; preds = %bb18.6
  %v56.6 = load i32, ptr addrspace(3) %v37, align 4
  %v59.6 = fcmp une float %v46.6, %v53.6
  %v61.6 = icmp uge i32 %v50.6, %v56.6
  %or.cond.6 = select i1 %v59.6, i1 true, i1 %v61.6
  br i1 %or.cond.6, label %bb30.6, label %bb25.6

bb25.6:                                           ; preds = %bb23.6, %bb18.6
  store float %v46.6, ptr addrspace(3) %v36, align 4
  store i32 %v50.6, ptr addrspace(3) %v37, align 4
  br label %bb30.6

bb30.6:                                           ; preds = %bb25.6, %bb23.6, %bb30.5
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v41.not.7 = icmp eq i32 %v12, 0
  br i1 %v41.not.7, label %bb18.7, label %bb30.7

bb18.7:                                           ; preds = %bb30.6
  %v46.7 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_8, i64 4), align 4
  %v50.7 = load i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_9, i64 4), align 4
  %v53.7 = load float, ptr addrspace(3) %v36, align 4
  %v57.7 = fcmp ule float %v46.7, %v53.7
  br i1 %v57.7, label %bb23.7, label %bb25.7

bb23.7:                                           ; preds = %bb18.7
  %v56.7 = load i32, ptr addrspace(3) %v37, align 4
  %v59.7 = fcmp une float %v46.7, %v53.7
  %v61.7 = icmp uge i32 %v50.7, %v56.7
  %or.cond.7 = select i1 %v59.7, i1 true, i1 %v61.7
  br i1 %or.cond.7, label %bb30.7, label %bb25.7

bb25.7:                                           ; preds = %bb23.7, %bb18.7
  store float %v46.7, ptr addrspace(3) %v36, align 4
  store i32 %v50.7, ptr addrspace(3) %v37, align 4
  br label %bb30.7

bb30.7:                                           ; preds = %bb25.7, %bb23.7, %bb30.6
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v67 = icmp eq i32 %v12, 0
  br i1 %v67, label %bb33, label %bb35

bb33:                                             ; preds = %bb30.7
  %v70 = load i32, ptr addrspace(3) @__shared_mem_9, align 4
  %v72 = uitofp i32 %v70 to float
  store float %v72, ptr %v3, align 4
  br label %bb35

bb35:                                             ; preds = %bb33, %bb30.7
  ret void
}

; Function Attrs: convergent nounwind memory(read, argmem: readwrite, inaccessiblemem: write, target_mem0: none, target_mem1: none)
define ptx_kernel void @attention_canvas_heads(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, ptr readonly captures(none) %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, float %v15, ptr captures(none) %v16, i64 %v17) #2 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i5 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i6 = icmp eq i32 %v4.i4, 1
  %v7.i7 = icmp eq i32 %v6.i5, 1
  %v8.not.not.i = and i1 %v5.i6, %v7.i7
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i8 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i8
  %v18.i.fr = freeze i64 %v18.i
  %v22.i = select i1 %.v18.i, i64 %v18.i.fr, i64 -1
  %v45 = zext i32 %v10 to i64
  %v46 = zext i32 %v12 to i64
  %v47 = mul nuw i64 %v46, %v45
  %v48.not = icmp ult i64 %v22.i, %v47
  br i1 %v48.not, label %bb3, label %bb39

bb3:                                              ; preds = %entry
  %v50.not = icmp eq i32 %v12, 0
  br i1 %v50.not, label %bb42, label %bb4

bb4:                                              ; preds = %bb3
  %v53 = urem i64 %v22.i, %v46
  %v54 = zext i32 %v14 to i64
  %v12.i = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v13, i32 1)
  %v56 = zext i32 %v12.i to i64
  %v5916 = udiv i32 %v12, %v12.i
  %0 = tail call i32 @llvm.umax.i32(i32 %v5916, i32 1)
  %v63.lhs.trunc = trunc nuw i64 %v53 to i32
  %v6317 = udiv i32 %v63.lhs.trunc, %0
  %v63.zext = zext i32 %v6317 to i64
  %v67 = mul i64 %v22.i, %v54
  %v69.not23.not = icmp eq i32 %v14, 0
  br i1 %v69.not23.not, label %bb11, label %bb10.lr.ph

bb10.lr.ph:                                       ; preds = %bb4
  %1 = getelementptr float, ptr %v16, i64 %v67
  %xtraiter = and i64 %v54, 7
  %2 = icmp ult i32 %v14, 8
  br i1 %2, label %bb10.epil.preheader, label %bb10.lr.ph.new

bb10.lr.ph.new:                                   ; preds = %bb10.lr.ph
  %unroll_iter = and i64 %v54, 4294967288
  br label %bb10

bb10:                                             ; preds = %bb10, %bb10.lr.ph.new
  %v6824 = phi i64 [ 0, %bb10.lr.ph.new ], [ %v74.7, %bb10 ]
  %niter = phi i64 [ 0, %bb10.lr.ph.new ], [ %niter.next.7, %bb10 ]
  %v73 = getelementptr float, ptr %1, i64 %v6824
  store float 0.000000e+00, ptr %v73, align 4
  %3 = getelementptr float, ptr %1, i64 %v6824
  %v73.1 = getelementptr i8, ptr %3, i64 4
  store float 0.000000e+00, ptr %v73.1, align 4
  %4 = getelementptr float, ptr %1, i64 %v6824
  %v73.2 = getelementptr i8, ptr %4, i64 8
  store float 0.000000e+00, ptr %v73.2, align 4
  %5 = getelementptr float, ptr %1, i64 %v6824
  %v73.3 = getelementptr i8, ptr %5, i64 12
  store float 0.000000e+00, ptr %v73.3, align 4
  %6 = getelementptr float, ptr %1, i64 %v6824
  %v73.4 = getelementptr i8, ptr %6, i64 16
  store float 0.000000e+00, ptr %v73.4, align 4
  %7 = getelementptr float, ptr %1, i64 %v6824
  %v73.5 = getelementptr i8, ptr %7, i64 20
  store float 0.000000e+00, ptr %v73.5, align 4
  %8 = getelementptr float, ptr %1, i64 %v6824
  %v73.6 = getelementptr i8, ptr %8, i64 24
  store float 0.000000e+00, ptr %v73.6, align 4
  %9 = getelementptr float, ptr %1, i64 %v6824
  %v73.7 = getelementptr i8, ptr %9, i64 28
  store float 0.000000e+00, ptr %v73.7, align 4
  %v74.7 = add nuw nsw i64 %v6824, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %bb11.loopexit.unr-lcssa, label %bb10

bb11.loopexit.unr-lcssa:                          ; preds = %bb10
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb11, label %bb10.epil.preheader

bb10.epil.preheader:                              ; preds = %bb11.loopexit.unr-lcssa, %bb10.lr.ph
  %v6824.epil.init = phi i64 [ 0, %bb10.lr.ph ], [ %v74.7, %bb11.loopexit.unr-lcssa ]
  %lcmp.mod46 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod46)
  br label %bb10.epil

bb10.epil:                                        ; preds = %bb10.epil, %bb10.epil.preheader
  %v6824.epil = phi i64 [ %v6824.epil.init, %bb10.epil.preheader ], [ %v74.epil, %bb10.epil ]
  %epil.iter = phi i64 [ 0, %bb10.epil.preheader ], [ %epil.iter.next, %bb10.epil ]
  %v73.epil = getelementptr float, ptr %1, i64 %v6824.epil
  store float 0.000000e+00, ptr %v73.epil, align 4
  %v74.epil = add nuw nsw i64 %v6824.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %bb11, label %bb10.epil, !llvm.loop !2

bb11:                                             ; preds = %bb11.loopexit.unr-lcssa, %bb10.epil, %bb4
  %v75 = zext i32 %v11 to i64
  %v76 = add nuw nsw i64 %v75, %v45
  %v81.not30.not = icmp eq i64 %v76, 0
  br i1 %v81.not30.not, label %bb32, label %bb13.lr.ph

bb13.lr.ph:                                       ; preds = %bb11
  %10 = getelementptr float, ptr %v16, i64 %v67
  br label %bb13

bb13:                                             ; preds = %bb13.lr.ph, %bb31
  %v8034 = phi i64 [ 0, %bb13.lr.ph ], [ %v159, %bb31 ]
  %v13133 = phi i1 [ true, %bb13.lr.ph ], [ false, %bb31 ]
  %v7832 = phi float [ 0.000000e+00, %bb13.lr.ph ], [ %v172, %bb31 ]
  %v7731 = phi float [ 0.000000e+00, %bb13.lr.ph ], [ %v138, %bb31 ]
  %v83.not = icmp samesign ult i64 %v8034, %v75
  br i1 %v83.not, label %bb16, label %bb15

bb15:                                             ; preds = %bb13
  %v93 = sub nuw nsw i64 %v8034, %v75
  br label %bb16

bb16:                                             ; preds = %bb13, %bb15
  %v80.pn = phi i64 [ %v93, %bb15 ], [ %v8034, %bb13 ]
  %v103 = phi ptr [ %v6, %bb15 ], [ %v2, %bb13 ]
  %v104 = phi i64 [ %v7, %bb15 ], [ %v3, %bb13 ]
  %v105 = phi ptr [ %v8, %bb15 ], [ %v4, %bb13 ]
  %v106 = phi i64 [ %v9, %bb15 ], [ %v5, %bb13 ]
  %v85.pn = mul i64 %v80.pn, %v56
  %v862.pn = add i64 %v85.pn, %v63.zext
  %v102 = mul i64 %v862.pn, %v54
  br i1 %v69.not23.not, label %bb21, label %bb18

bb18:                                             ; preds = %bb16, %bb20
  %v11227 = phi float [ %v128, %bb20 ], [ 0.000000e+00, %bb16 ]
  %v11126 = phi i64 [ %v129, %bb20 ], [ 0, %bb16 ]
  %v115 = add nuw i64 %v11126, %v67
  %v117 = icmp ult i64 %v115, %v1
  br i1 %v117, label %bb19, label %bb45

bb19:                                             ; preds = %bb18
  %v121 = add nuw i64 %v11126, %v102
  %v123 = icmp ult i64 %v121, %v104
  br i1 %v123, label %bb20, label %bb46

bb20:                                             ; preds = %bb19
  %v119 = getelementptr inbounds float, ptr %v0, i64 %v115
  %v120 = load float, ptr %v119, align 4
  %v125 = getelementptr inbounds float, ptr %v103, i64 %v121
  %v126 = load float, ptr %v125, align 4
  %v127 = fmul contract float %v120, %v126
  %v128 = fadd contract float %v11227, %v127
  %v129 = add nuw nsw i64 %v11126, 1
  %exitcond39.not = icmp eq i64 %v129, %v54
  br i1 %exitcond39.not, label %bb21, label %bb18

bb21:                                             ; preds = %bb20, %bb16
  %v112.lcssa = phi float [ 0.000000e+00, %bb16 ], [ %v128, %bb20 ]
  %v130 = fmul contract float %v15, %v112.lcssa
  br i1 %v13133, label %bb27, label %bb22

bb22:                                             ; preds = %bb21
  %v132 = fcmp ule float %v130, %v7731
  br i1 %v132, label %bb27, label %bb24

bb24:                                             ; preds = %bb22
  %v134 = fsub contract float %v7731, %v130
  %11 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %11, 0
  %12 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v134, float 0x3F777313A0000000, float 5.000000e-01) #20
  %13 = tail call float @llvm.fma.f32(float %v134, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %13, float %12
  %14 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %15 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %15, float %14
  %16 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %17 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %17, float %16
  %18 = fadd float %.04.i, 0xC168000FE0000000
  %19 = fneg float %18
  %20 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v134, float 0x3FF7154760000000, float %19) #20
  %21 = tail call float @llvm.fma.f32(float %v134, float 0x3FF7154760000000, float %19)
  %.0.i = select i1 %.not.i, float %21, float %20
  %22 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v134, float 0x3E54AE0C00000000, float %.0.i) #20
  %23 = tail call float @llvm.fma.f32(float %v134, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %23, float %22
  %24 = bitcast float %.04.i to i32
  %25 = shl i32 %24, 23
  %26 = bitcast i32 %25 to float
  %27 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %28 = fmul float %27, %26
  br label %bb27

bb27:                                             ; preds = %bb24, %bb22, %bb21
  %v138 = phi float [ %v130, %bb21 ], [ %v130, %bb24 ], [ %v7731, %bb22 ]
  %v139 = phi float [ 0.000000e+00, %bb21 ], [ %28, %bb24 ], [ 1.000000e+00, %bb22 ]
  %v140 = fsub contract float %v130, %v138
  %29 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i9 = icmp eq i32 %29, 0
  %30 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v140, float 0x3F777313A0000000, float 5.000000e-01) #20
  %31 = tail call float @llvm.fma.f32(float %v140, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i10 = select i1 %.not.i9, float %31, float %30
  %32 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i10) #20
  %33 = tail call float @llvm.nvvm.saturate.f(float %.02.i10) #20
  %.03.i11 = select i1 %.not.i9, float %33, float %32
  %34 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i11, float 2.520000e+02, float 0x4168000020000000) #20
  %35 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i11, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i12 = select i1 %.not.i9, float %35, float %34
  %36 = fadd float %.04.i12, 0xC168000FE0000000
  %37 = fneg float %36
  %38 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v140, float 0x3FF7154760000000, float %37) #20
  %39 = tail call float @llvm.fma.f32(float %v140, float 0x3FF7154760000000, float %37)
  %.0.i13 = select i1 %.not.i9, float %39, float %38
  %40 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v140, float 0x3E54AE0C00000000, float %.0.i13) #20
  %41 = tail call float @llvm.fma.f32(float %v140, float 0x3E54AE0C00000000, float %.0.i13)
  %.01.i14 = select i1 %.not.i9, float %41, float %40
  %42 = bitcast float %.04.i12 to i32
  %43 = shl i32 %42, 23
  %44 = bitcast i32 %43 to float
  %45 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i14)
  %46 = fmul float %45, %44
  %v171 = fmul contract float %v7832, %v139
  %v172 = fadd contract float %v171, %46
  br i1 %v69.not23.not, label %bb31, label %bb29

bb29:                                             ; preds = %bb27, %bb30
  %v14229 = phi i64 [ %v158, %bb30 ], [ 0, %bb27 ]
  %v150 = add nuw i64 %v14229, %v102
  %v152 = icmp ult i64 %v150, %v106
  br i1 %v152, label %bb30, label %bb47

bb30:                                             ; preds = %bb29
  %v147 = getelementptr float, ptr %10, i64 %v14229
  %v148 = load float, ptr %v147, align 4
  %v149 = fmul contract float %v139, %v148
  %v154 = getelementptr inbounds float, ptr %v105, i64 %v150
  %v155 = load float, ptr %v154, align 4
  %v156 = fmul contract float %46, %v155
  %v157 = fadd contract float %v149, %v156
  store float %v157, ptr %v147, align 4
  %v158 = add nuw nsw i64 %v14229, 1
  %exitcond40.not = icmp eq i64 %v158, %v54
  br i1 %exitcond40.not, label %bb31, label %bb29

bb31:                                             ; preds = %bb30, %bb27
  %v159 = add nuw nsw i64 %v8034, 1
  %exitcond41.not = icmp eq i64 %v159, %v76
  br i1 %exitcond41.not, label %bb32, label %bb13

bb32:                                             ; preds = %bb31, %bb11
  %v78.lcssa = phi float [ 0.000000e+00, %bb11 ], [ %v172, %bb31 ]
  %v160 = fcmp ogt float %v78.lcssa, 0.000000e+00
  %v163.not36 = icmp ne i32 %v14, 0
  %or.cond = and i1 %v160, %v163.not36
  br i1 %or.cond, label %bb36.lr.ph, label %bb39

bb36.lr.ph:                                       ; preds = %bb32
  %47 = getelementptr float, ptr %v16, i64 %v67
  %xtraiter47 = and i64 %v54, 1
  %48 = icmp eq i32 %v14, 1
  br i1 %48, label %bb36.epil.preheader, label %bb36.lr.ph.new

bb36.lr.ph.new:                                   ; preds = %bb36.lr.ph
  %unroll_iter51 = and i64 %v54, 4294967294
  br label %bb36

bb36:                                             ; preds = %bb36, %bb36.lr.ph.new
  %v16237 = phi i64 [ 0, %bb36.lr.ph.new ], [ %v170.1, %bb36 ]
  %niter52 = phi i64 [ 0, %bb36.lr.ph.new ], [ %niter52.next.1, %bb36 ]
  %v167 = getelementptr float, ptr %47, i64 %v16237
  %v168 = load float, ptr %v167, align 4
  %v169 = fdiv contract float %v168, %v78.lcssa
  store float %v169, ptr %v167, align 4
  %49 = getelementptr float, ptr %47, i64 %v16237
  %v167.1 = getelementptr i8, ptr %49, i64 4
  %v168.1 = load float, ptr %v167.1, align 4
  %v169.1 = fdiv contract float %v168.1, %v78.lcssa
  store float %v169.1, ptr %v167.1, align 4
  %v170.1 = add nuw nsw i64 %v16237, 2
  %niter52.next.1 = add i64 %niter52, 2
  %niter52.ncmp.1 = icmp eq i64 %niter52.next.1, %unroll_iter51
  br i1 %niter52.ncmp.1, label %bb39.loopexit.unr-lcssa, label %bb36

bb39.loopexit.unr-lcssa:                          ; preds = %bb36
  %lcmp.mod49.not = icmp eq i64 %xtraiter47, 0
  br i1 %lcmp.mod49.not, label %bb39, label %bb36.epil.preheader

bb36.epil.preheader:                              ; preds = %bb39.loopexit.unr-lcssa, %bb36.lr.ph
  %v16237.epil.init = phi i64 [ 0, %bb36.lr.ph ], [ %v170.1, %bb39.loopexit.unr-lcssa ]
  %lcmp.mod50 = icmp ne i64 %xtraiter47, 0
  tail call void @llvm.assume(i1 %lcmp.mod50)
  %v167.epil = getelementptr float, ptr %47, i64 %v16237.epil.init
  %v168.epil = load float, ptr %v167.epil, align 4
  %v169.epil = fdiv contract float %v168.epil, %v78.lcssa
  store float %v169.epil, ptr %v167.epil, align 4
  br label %bb39

bb39:                                             ; preds = %bb36.epil.preheader, %bb39.loopexit.unr-lcssa, %bb32, %entry
  ret void

bb42:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb29
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @attention_canvas_paged_heads(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, ptr readonly captures(none) %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, float %v15, i32 %v16, i32 %v17, i32 %v18, i32 %v19, i32 %v20, ptr captures(none) %v21, i64 %v22) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i5 = icmp eq i32 %v4.i3, 1
  %v7.i6 = icmp eq i32 %v6.i4, 1
  %v8.not.not.i = and i1 %v5.i5, %v7.i6
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i7 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i7
  %v18.i.fr = freeze i64 %v18.i
  %v22.i = select i1 %.v18.i, i64 %v18.i.fr, i64 -1
  %v55 = zext i32 %v10 to i64
  %v56 = zext i32 %v12 to i64
  %v57 = mul nuw i64 %v56, %v55
  %v58.not = icmp ult i64 %v22.i, %v57
  br i1 %v58.not, label %bb3, label %bb60

bb3:                                              ; preds = %entry
  %v60.not = icmp eq i32 %v12, 0
  br i1 %v60.not, label %bb65, label %bb4

bb4:                                              ; preds = %bb3
  %v63 = urem i64 %v22.i, %v56
  %v64 = zext i32 %v14 to i64
  %v12.i = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v13, i32 1)
  %v66 = zext i32 %v12.i to i64
  %v6969 = udiv i32 %v12, %v12.i
  %0 = tail call i32 @llvm.umax.i32(i32 %v6969, i32 1)
  %v73.lhs.trunc = trunc nuw i64 %v63 to i32
  %v7370 = udiv i32 %v73.lhs.trunc, %0
  %v73.zext = zext i32 %v7370 to i64
  %v77 = mul i64 %v22.i, %v64
  %v79.not89.not = icmp eq i32 %v14, 0
  br i1 %v79.not89.not, label %bb12.preheader, label %bb10.lr.ph

bb10.lr.ph:                                       ; preds = %bb4
  %1 = getelementptr float, ptr %v21, i64 %v77
  %xtraiter = and i64 %v64, 7
  %2 = icmp ult i32 %v14, 8
  br i1 %2, label %bb10.epil.preheader, label %bb10.lr.ph.new

bb10.lr.ph.new:                                   ; preds = %bb10.lr.ph
  %unroll_iter = and i64 %v64, 4294967288
  br label %bb10

bb12.preheader.loopexit.unr-lcssa:                ; preds = %bb10
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb12.preheader, label %bb10.epil.preheader

bb10.epil.preheader:                              ; preds = %bb12.preheader.loopexit.unr-lcssa, %bb10.lr.ph
  %v7890.epil.init = phi i64 [ 0, %bb10.lr.ph ], [ %v84.7, %bb12.preheader.loopexit.unr-lcssa ]
  %lcmp.mod120 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod120)
  br label %bb10.epil

bb10.epil:                                        ; preds = %bb10.epil, %bb10.epil.preheader
  %v7890.epil = phi i64 [ %v7890.epil.init, %bb10.epil.preheader ], [ %v84.epil, %bb10.epil ]
  %epil.iter = phi i64 [ 0, %bb10.epil.preheader ], [ %epil.iter.next, %bb10.epil ]
  %v83.epil = getelementptr float, ptr %1, i64 %v7890.epil
  store float 0.000000e+00, ptr %v83.epil, align 4
  %v84.epil = add nuw nsw i64 %v7890.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %bb12.preheader, label %bb10.epil, !llvm.loop !4

bb12.preheader:                                   ; preds = %bb12.preheader.loopexit.unr-lcssa, %bb10.epil, %bb4
  %v89 = zext i32 %v11 to i64
  %v90 = add nuw nsw i64 %v89, %v55
  %v91.not102.not = icmp eq i64 %v90, 0
  br i1 %v91.not102.not, label %bb53, label %bb13.lr.ph

bb13.lr.ph:                                       ; preds = %bb12.preheader
  %factor.op.mul = mul nuw i64 %v64, %v73.zext
  %3 = getelementptr float, ptr %v21, i64 %v77
  %v95 = zext i32 %v18 to i64
  %v96.not = icmp eq i32 %v18, 0
  %v100 = zext i32 %v16 to i64
  %v101 = zext i32 %v17 to i64
  %v102 = mul nuw i64 %v101, %v100
  %v110 = zext i32 %v20 to i64
  %v111 = shl nuw nsw i64 %v110, 1
  %v112 = zext i32 %v19 to i64
  %v116.reass = shl i64 %factor.op.mul, 1
  %v267 = mul i64 %v111, %v95
  %invariant.op = add i64 %v267, %v116.reass
  br label %bb13

bb10:                                             ; preds = %bb10, %bb10.lr.ph.new
  %v7890 = phi i64 [ 0, %bb10.lr.ph.new ], [ %v84.7, %bb10 ]
  %niter = phi i64 [ 0, %bb10.lr.ph.new ], [ %niter.next.7, %bb10 ]
  %v83 = getelementptr float, ptr %1, i64 %v7890
  store float 0.000000e+00, ptr %v83, align 4
  %4 = getelementptr float, ptr %1, i64 %v7890
  %v83.1 = getelementptr i8, ptr %4, i64 4
  store float 0.000000e+00, ptr %v83.1, align 4
  %5 = getelementptr float, ptr %1, i64 %v7890
  %v83.2 = getelementptr i8, ptr %5, i64 8
  store float 0.000000e+00, ptr %v83.2, align 4
  %6 = getelementptr float, ptr %1, i64 %v7890
  %v83.3 = getelementptr i8, ptr %6, i64 12
  store float 0.000000e+00, ptr %v83.3, align 4
  %7 = getelementptr float, ptr %1, i64 %v7890
  %v83.4 = getelementptr i8, ptr %7, i64 16
  store float 0.000000e+00, ptr %v83.4, align 4
  %8 = getelementptr float, ptr %1, i64 %v7890
  %v83.5 = getelementptr i8, ptr %8, i64 20
  store float 0.000000e+00, ptr %v83.5, align 4
  %9 = getelementptr float, ptr %1, i64 %v7890
  %v83.6 = getelementptr i8, ptr %9, i64 24
  store float 0.000000e+00, ptr %v83.6, align 4
  %10 = getelementptr float, ptr %1, i64 %v7890
  %v83.7 = getelementptr i8, ptr %10, i64 28
  store float 0.000000e+00, ptr %v83.7, align 4
  %v84.7 = add nuw nsw i64 %v7890, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %bb12.preheader.loopexit.unr-lcssa, label %bb10

bb13:                                             ; preds = %bb13.lr.ph, %bb52
  %v88106 = phi i64 [ 0, %bb13.lr.ph ], [ %v253, %bb52 ]
  %v87105 = phi i1 [ true, %bb13.lr.ph ], [ false, %bb52 ]
  %v86104 = phi float [ 0.000000e+00, %bb13.lr.ph ], [ %v251, %bb52 ]
  %v85103 = phi float [ 0.000000e+00, %bb13.lr.ph ], [ %v250, %bb52 ]
  %v93.not = icmp samesign ult i64 %v88106, %v89
  br i1 %v93.not, label %bb14, label %bb36

bb14:                                             ; preds = %bb13
  br i1 %v96.not, label %bb68, label %bb15

bb15:                                             ; preds = %bb14
  %v98.lhs.trunc = trunc nuw i64 %v88106 to i32
  %v98.lhs.trunc.frozen = freeze i32 %v98.lhs.trunc
  %v18.frozen = freeze i32 %v18
  %v9871 = udiv i32 %v98.lhs.trunc.frozen, %v18.frozen
  %v98.zext = zext i32 %v9871 to i64
  %v103 = add nuw i64 %v102, %v98.zext
  %v105 = icmp ult i64 %v103, %v5
  br i1 %v105, label %bb16, label %bb69

bb16:                                             ; preds = %bb15
  %11 = mul i32 %v9871, %v18.frozen
  %v9972.decomposed = sub i32 %v98.lhs.trunc.frozen, %11
  %v99.zext = zext i32 %v9972.decomposed to i64
  %v107 = getelementptr inbounds i32, ptr %v4, i64 %v103
  %v108 = load i32, ptr %v107, align 4
  %v109 = zext i32 %v108 to i64
  %v113 = mul nuw i64 %v109, %v112
  %v114 = mul i64 %v111, %v99.zext
  %v115 = add i64 %v113, %v114
  %v118 = add i64 %v115, %v116.reass
  br i1 %v79.not89.not, label %bb23, label %bb18

bb18:                                             ; preds = %bb16, %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v12098 = phi float [ %v151, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ 0.000000e+00, %bb16 ]
  %v11997 = phi i64 [ %v152, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ 0, %bb16 ]
  %v123 = shl nuw i64 %v11997, 1
  %v124 = add i64 %v118, %v123
  %v126 = icmp ult i64 %v124, %v3
  br i1 %v126, label %bb19, label %bb70

bb19:                                             ; preds = %bb18
  %v133 = add nuw i64 %v124, 1
  %v134 = icmp ult i64 %v133, %v3
  br i1 %v134, label %bb20, label %bb71

bb20:                                             ; preds = %bb19
  %v143 = add nuw i64 %v11997, %v77
  %v145 = icmp ult i64 %v143, %v1
  br i1 %v145, label %bb21, label %bb72

bb21:                                             ; preds = %bb20
  %v128 = getelementptr inbounds i8, ptr %v2, i64 %v124
  %v129 = load i8, ptr %v128, align 1
  %v130 = zext i8 %v129 to i16
  %v136 = getelementptr inbounds i8, ptr %v2, i64 %v133
  %v137 = load i8, ptr %v136, align 1
  %v138 = zext i8 %v137 to i16
  %v141 = shl nuw i16 %v138, 8
  %v147 = getelementptr inbounds float, ptr %v0, i64 %v143
  %v148 = load float, ptr %v147, align 4
  %v4.i27 = lshr i16 %v138, 7
  %v6.i28 = zext nneg i16 %v4.i27 to i32
  %v9.i = lshr i16 %v138, 2
  %v10.i = and i16 %v9.i, 31
  %v141.masked = and i16 %v141, 768
  %v12.i29 = or disjoint i16 %v141.masked, %v130
  %v13.i30 = zext nneg i16 %v12.i29 to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb21
  %v15.i31 = icmp eq i16 %v12.i29, 0
  br i1 %v15.i31, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i32 = shl nuw i32 %v6.i28, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i30, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i30, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i28, 31
  %12 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %12
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb21
  %v38.i = shl nuw i32 %v6.i28, 31
  %v41.i = shl nuw nsw i32 %v13.i30, 13
  %v39.i = or disjoint i32 %v41.i, %v38.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb21
  %v44.i = shl nuw i32 %v6.i28, 31
  %13 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %13 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i30, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i32, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v150 = fmul contract float %v148, %v55.i
  %v151 = fadd contract float %v12098, %v150
  %v152 = add nuw nsw i64 %v11997, 1
  %exitcond114.not = icmp eq i64 %v152, %v64
  br i1 %exitcond114.not, label %bb23, label %bb18

bb23:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb16
  %v120.lcssa = phi float [ 0.000000e+00, %bb16 ], [ %v151, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %v153 = fmul contract float %v15, %v120.lcssa
  br i1 %v87105, label %bb29, label %bb24

bb24:                                             ; preds = %bb23
  %v155 = fcmp ule float %v153, %v85103
  br i1 %v155, label %bb29, label %bb26

bb26:                                             ; preds = %bb24
  %v157 = fsub contract float %v85103, %v153
  %14 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %14, 0
  %15 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v157, float 0x3F777313A0000000, float 5.000000e-01) #20
  %16 = tail call float @llvm.fma.f32(float %v157, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %16, float %15
  %17 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %18 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %18, float %17
  %19 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %20 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %20, float %19
  %21 = fadd float %.04.i, 0xC168000FE0000000
  %22 = fneg float %21
  %23 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v157, float 0x3FF7154760000000, float %22) #20
  %24 = tail call float @llvm.fma.f32(float %v157, float 0x3FF7154760000000, float %22)
  %.0.i = select i1 %.not.i, float %24, float %23
  %25 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v157, float 0x3E54AE0C00000000, float %.0.i) #20
  %26 = tail call float @llvm.fma.f32(float %v157, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %26, float %25
  %27 = bitcast float %.04.i to i32
  %28 = shl i32 %27, 23
  %29 = bitcast i32 %28 to float
  %30 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %31 = fmul float %30, %29
  br label %bb29

bb29:                                             ; preds = %bb26, %bb24, %bb23
  %v161 = phi float [ %v153, %bb23 ], [ %v153, %bb26 ], [ %v85103, %bb24 ]
  %v162 = phi float [ 0.000000e+00, %bb23 ], [ %31, %bb26 ], [ 1.000000e+00, %bb24 ]
  %v163 = fsub contract float %v153, %v161
  %32 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i8 = icmp eq i32 %32, 0
  %33 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v163, float 0x3F777313A0000000, float 5.000000e-01) #20
  %34 = tail call float @llvm.fma.f32(float %v163, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i9 = select i1 %.not.i8, float %34, float %33
  %35 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i9) #20
  %36 = tail call float @llvm.nvvm.saturate.f(float %.02.i9) #20
  %.03.i10 = select i1 %.not.i8, float %36, float %35
  %37 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i10, float 2.520000e+02, float 0x4168000020000000) #20
  %38 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i10, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i11 = select i1 %.not.i8, float %38, float %37
  %39 = fadd float %.04.i11, 0xC168000FE0000000
  %40 = fneg float %39
  %41 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v163, float 0x3FF7154760000000, float %40) #20
  %42 = tail call float @llvm.fma.f32(float %v163, float 0x3FF7154760000000, float %40)
  %.0.i12 = select i1 %.not.i8, float %42, float %41
  %43 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v163, float 0x3E54AE0C00000000, float %.0.i12) #20
  %44 = tail call float @llvm.fma.f32(float %v163, float 0x3E54AE0C00000000, float %.0.i12)
  %.01.i13 = select i1 %.not.i8, float %44, float %43
  %45 = bitcast float %.04.i11 to i32
  %46 = shl i32 %45, 23
  %47 = bitcast i32 %46 to float
  %48 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i13)
  %49 = fmul float %48, %47
  %v265 = fmul contract float %v86104, %v162
  %v266 = fadd contract float %v265, %49
  br i1 %v79.not89.not, label %bb52, label %bb31.lr.ph

bb31.lr.ph:                                       ; preds = %bb29
  %v270.reass = add i64 %v115, %invariant.op
  br label %bb31

bb31:                                             ; preds = %bb31.lr.ph, %cuda_kernels__oxide_kernels__f16_to_f32.exit68
  %v165101 = phi i64 [ 0, %bb31.lr.ph ], [ %v196, %cuda_kernels__oxide_kernels__f16_to_f32.exit68 ]
  %v168 = shl nuw i64 %v165101, 1
  %v169 = add i64 %v270.reass, %v168
  %v171 = icmp ult i64 %v169, %v3
  br i1 %v171, label %bb32, label %bb73

bb32:                                             ; preds = %bb31
  %v178 = add nuw i64 %v169, 1
  %v179 = icmp ult i64 %v178, %v3
  br i1 %v179, label %bb33, label %bb74

bb33:                                             ; preds = %bb32
  %v173 = getelementptr inbounds i8, ptr %v2, i64 %v169
  %v174 = load i8, ptr %v173, align 1
  %v175 = zext i8 %v174 to i16
  %v181 = getelementptr inbounds i8, ptr %v2, i64 %v178
  %v182 = load i8, ptr %v181, align 1
  %v183 = zext i8 %v182 to i16
  %v186 = shl nuw i16 %v183, 8
  %v190 = getelementptr float, ptr %3, i64 %v165101
  %v191 = load float, ptr %v190, align 4
  %v192 = fmul contract float %v162, %v191
  %v4.i33 = lshr i16 %v183, 7
  %v6.i34 = zext nneg i16 %v4.i33 to i32
  %v9.i35 = lshr i16 %v183, 2
  %v10.i36 = and i16 %v9.i35, 31
  %v186.masked = and i16 %v186, 768
  %v12.i37 = or disjoint i16 %v186.masked, %v175
  %v13.i38 = zext nneg i16 %v12.i37 to i32
  switch i16 %v10.i36, label %bb10.i61 [
    i16 0, label %bb1.i46
    i16 31, label %bb9.i39
  ]

bb1.i46:                                          ; preds = %bb33
  %v15.i47 = icmp eq i16 %v12.i37, 0
  br i1 %v15.i47, label %bb2.i59, label %bb6.i48

bb2.i59:                                          ; preds = %bb1.i46
  %v17.i60 = shl nuw i32 %v6.i34, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit68

bb6.i48:                                          ; preds = %bb1.i46
  %v13.masked.numleadingzeros.i49 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i38, i1 true)
  %v13.masked.leadingonepos.i50 = xor i32 %v13.masked.numleadingzeros.i49, 31
  %bb5.tripcount.i51 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i50
  %v23.i52 = shl nuw nsw i32 %v13.i38, %bb5.tripcount.i51
  %v27.i53 = shl nuw i32 %v6.i34, 31
  %50 = shl nuw nsw i32 %v13.masked.numleadingzeros.i49, 23
  %reass.sub110 = sub i32 %v27.i53, %50
  %v31.i55 = add i32 %reass.sub110, 1124073472
  %v25.i56 = shl i32 %v23.i52, 13
  %v33.i57 = and i32 %v25.i56, 8380416
  %v34.i58 = or disjoint i32 %v33.i57, %v31.i55
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit68

bb9.i39:                                          ; preds = %bb33
  %v38.i40 = shl nuw i32 %v6.i34, 31
  %v41.i41 = shl nuw nsw i32 %v13.i38, 13
  %v39.i42 = or disjoint i32 %v41.i41, %v38.i40
  %v42.i43 = or disjoint i32 %v39.i42, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit68

bb10.i61:                                         ; preds = %bb33
  %v44.i62 = shl nuw i32 %v6.i34, 31
  %51 = add nuw nsw i16 %v10.i36, 112
  %v46.i63 = zext nneg i16 %51 to i32
  %v48.i64 = shl nuw nsw i32 %v46.i63, 23
  %v49.i65 = or disjoint i32 %v48.i64, %v44.i62
  %v51.i66 = shl nuw nsw i32 %v13.i38, 13
  %v52.i67 = or disjoint i32 %v49.i65, %v51.i66
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit68

cuda_kernels__oxide_kernels__f16_to_f32.exit68:   ; preds = %bb2.i59, %bb6.i48, %bb9.i39, %bb10.i61
  %v54.i44 = phi i32 [ %v34.i58, %bb6.i48 ], [ %v17.i60, %bb2.i59 ], [ %v42.i43, %bb9.i39 ], [ %v52.i67, %bb10.i61 ]
  %v55.i45 = bitcast i32 %v54.i44 to float
  %v194 = fmul contract float %49, %v55.i45
  %v195 = fadd contract float %v192, %v194
  store float %v195, ptr %v190, align 4
  %v196 = add nuw nsw i64 %v165101, 1
  %exitcond115.not = icmp eq i64 %v196, %v64
  br i1 %exitcond115.not, label %bb52, label %bb31

bb36:                                             ; preds = %bb13
  %v197 = sub nuw nsw i64 %v88106, %v89
  %v198 = mul i64 %v197, %v66
  %v1992 = add i64 %v198, %v73.zext
  %v201 = mul i64 %v1992, %v64
  br i1 %v79.not89.not, label %bb41, label %bb38

bb38:                                             ; preds = %bb36, %bb40
  %v20393 = phi float [ %v219, %bb40 ], [ 0.000000e+00, %bb36 ]
  %v20292 = phi i64 [ %v220, %bb40 ], [ 0, %bb36 ]
  %v206 = add nuw i64 %v20292, %v77
  %v208 = icmp ult i64 %v206, %v1
  br i1 %v208, label %bb39, label %bb75

bb39:                                             ; preds = %bb38
  %v212 = add nuw i64 %v20292, %v201
  %v214 = icmp ult i64 %v212, %v7
  br i1 %v214, label %bb40, label %bb76

bb40:                                             ; preds = %bb39
  %v210 = getelementptr inbounds float, ptr %v0, i64 %v206
  %v211 = load float, ptr %v210, align 4
  %v216 = getelementptr inbounds float, ptr %v6, i64 %v212
  %v217 = load float, ptr %v216, align 4
  %v218 = fmul contract float %v211, %v217
  %v219 = fadd contract float %v20393, %v218
  %v220 = add nuw nsw i64 %v20292, 1
  %exitcond112.not = icmp eq i64 %v220, %v64
  br i1 %exitcond112.not, label %bb41, label %bb38

bb41:                                             ; preds = %bb40, %bb36
  %v203.lcssa = phi float [ 0.000000e+00, %bb36 ], [ %v219, %bb40 ]
  %v221 = fmul contract float %v15, %v203.lcssa
  br i1 %v87105, label %bb47, label %bb42

bb42:                                             ; preds = %bb41
  %v223 = fcmp ule float %v221, %v85103
  br i1 %v223, label %bb47, label %bb44

bb44:                                             ; preds = %bb42
  %v225 = fsub contract float %v85103, %v221
  %52 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i14 = icmp eq i32 %52, 0
  %53 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v225, float 0x3F777313A0000000, float 5.000000e-01) #20
  %54 = tail call float @llvm.fma.f32(float %v225, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i15 = select i1 %.not.i14, float %54, float %53
  %55 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i15) #20
  %56 = tail call float @llvm.nvvm.saturate.f(float %.02.i15) #20
  %.03.i16 = select i1 %.not.i14, float %56, float %55
  %57 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i16, float 2.520000e+02, float 0x4168000020000000) #20
  %58 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i16, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i17 = select i1 %.not.i14, float %58, float %57
  %59 = fadd float %.04.i17, 0xC168000FE0000000
  %60 = fneg float %59
  %61 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v225, float 0x3FF7154760000000, float %60) #20
  %62 = tail call float @llvm.fma.f32(float %v225, float 0x3FF7154760000000, float %60)
  %.0.i18 = select i1 %.not.i14, float %62, float %61
  %63 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v225, float 0x3E54AE0C00000000, float %.0.i18) #20
  %64 = tail call float @llvm.fma.f32(float %v225, float 0x3E54AE0C00000000, float %.0.i18)
  %.01.i19 = select i1 %.not.i14, float %64, float %63
  %65 = bitcast float %.04.i17 to i32
  %66 = shl i32 %65, 23
  %67 = bitcast i32 %66 to float
  %68 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i19)
  %69 = fmul float %68, %67
  br label %bb47

bb47:                                             ; preds = %bb44, %bb42, %bb41
  %v229 = phi float [ %v221, %bb41 ], [ %v221, %bb44 ], [ %v85103, %bb42 ]
  %v230 = phi float [ 0.000000e+00, %bb41 ], [ %69, %bb44 ], [ 1.000000e+00, %bb42 ]
  %v231 = fsub contract float %v221, %v229
  %70 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i20 = icmp eq i32 %70, 0
  %71 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v231, float 0x3F777313A0000000, float 5.000000e-01) #20
  %72 = tail call float @llvm.fma.f32(float %v231, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i21 = select i1 %.not.i20, float %72, float %71
  %73 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i21) #20
  %74 = tail call float @llvm.nvvm.saturate.f(float %.02.i21) #20
  %.03.i22 = select i1 %.not.i20, float %74, float %73
  %75 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i22, float 2.520000e+02, float 0x4168000020000000) #20
  %76 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i22, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i23 = select i1 %.not.i20, float %76, float %75
  %77 = fadd float %.04.i23, 0xC168000FE0000000
  %78 = fneg float %77
  %79 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v231, float 0x3FF7154760000000, float %78) #20
  %80 = tail call float @llvm.fma.f32(float %v231, float 0x3FF7154760000000, float %78)
  %.0.i24 = select i1 %.not.i20, float %80, float %79
  %81 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v231, float 0x3E54AE0C00000000, float %.0.i24) #20
  %82 = tail call float @llvm.fma.f32(float %v231, float 0x3E54AE0C00000000, float %.0.i24)
  %.01.i25 = select i1 %.not.i20, float %82, float %81
  %83 = bitcast float %.04.i23 to i32
  %84 = shl i32 %83, 23
  %85 = bitcast i32 %84 to float
  %86 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i25)
  %87 = fmul float %86, %85
  %v271 = fmul contract float %v86104, %v230
  %v272 = fadd contract float %v271, %87
  br i1 %v79.not89.not, label %bb52, label %bb49

bb49:                                             ; preds = %bb47, %bb50
  %v23395 = phi i64 [ %v249, %bb50 ], [ 0, %bb47 ]
  %v241 = add nuw i64 %v23395, %v201
  %v243 = icmp ult i64 %v241, %v9
  br i1 %v243, label %bb50, label %bb77

bb50:                                             ; preds = %bb49
  %v238 = getelementptr float, ptr %3, i64 %v23395
  %v239 = load float, ptr %v238, align 4
  %v240 = fmul contract float %v230, %v239
  %v245 = getelementptr inbounds float, ptr %v8, i64 %v241
  %v246 = load float, ptr %v245, align 4
  %v247 = fmul contract float %87, %v246
  %v248 = fadd contract float %v240, %v247
  store float %v248, ptr %v238, align 4
  %v249 = add nuw nsw i64 %v23395, 1
  %exitcond113.not = icmp eq i64 %v249, %v64
  br i1 %exitcond113.not, label %bb52, label %bb49

bb52:                                             ; preds = %bb50, %cuda_kernels__oxide_kernels__f16_to_f32.exit68, %bb47, %bb29
  %v250 = phi float [ %v161, %bb29 ], [ %v229, %bb47 ], [ %v161, %cuda_kernels__oxide_kernels__f16_to_f32.exit68 ], [ %v229, %bb50 ]
  %v251 = phi float [ %v266, %bb29 ], [ %v272, %bb47 ], [ %v266, %cuda_kernels__oxide_kernels__f16_to_f32.exit68 ], [ %v272, %bb50 ]
  %v253 = add nuw nsw i64 %v88106, 1
  %exitcond116.not = icmp eq i64 %v253, %v90
  br i1 %exitcond116.not, label %bb53, label %bb13

bb53:                                             ; preds = %bb52, %bb12.preheader
  %v86.lcssa = phi float [ 0.000000e+00, %bb12.preheader ], [ %v251, %bb52 ]
  %v254 = fcmp ogt float %v86.lcssa, 0.000000e+00
  %v257.not108 = icmp ne i32 %v14, 0
  %or.cond = and i1 %v254, %v257.not108
  br i1 %or.cond, label %bb57.lr.ph, label %bb60

bb57.lr.ph:                                       ; preds = %bb53
  %88 = getelementptr float, ptr %v21, i64 %v77
  %xtraiter121 = and i64 %v64, 1
  %89 = icmp eq i32 %v14, 1
  br i1 %89, label %bb57.epil.preheader, label %bb57.lr.ph.new

bb57.lr.ph.new:                                   ; preds = %bb57.lr.ph
  %unroll_iter125 = and i64 %v64, 4294967294
  br label %bb57

bb57:                                             ; preds = %bb57, %bb57.lr.ph.new
  %v256109 = phi i64 [ 0, %bb57.lr.ph.new ], [ %v264.1, %bb57 ]
  %niter126 = phi i64 [ 0, %bb57.lr.ph.new ], [ %niter126.next.1, %bb57 ]
  %v261 = getelementptr float, ptr %88, i64 %v256109
  %v262 = load float, ptr %v261, align 4
  %v263 = fdiv contract float %v262, %v86.lcssa
  store float %v263, ptr %v261, align 4
  %90 = getelementptr float, ptr %88, i64 %v256109
  %v261.1 = getelementptr i8, ptr %90, i64 4
  %v262.1 = load float, ptr %v261.1, align 4
  %v263.1 = fdiv contract float %v262.1, %v86.lcssa
  store float %v263.1, ptr %v261.1, align 4
  %v264.1 = add nuw nsw i64 %v256109, 2
  %niter126.next.1 = add i64 %niter126, 2
  %niter126.ncmp.1 = icmp eq i64 %niter126.next.1, %unroll_iter125
  br i1 %niter126.ncmp.1, label %bb60.loopexit.unr-lcssa, label %bb57

bb60.loopexit.unr-lcssa:                          ; preds = %bb57
  %lcmp.mod123.not = icmp eq i64 %xtraiter121, 0
  br i1 %lcmp.mod123.not, label %bb60, label %bb57.epil.preheader

bb57.epil.preheader:                              ; preds = %bb60.loopexit.unr-lcssa, %bb57.lr.ph
  %v256109.epil.init = phi i64 [ 0, %bb57.lr.ph ], [ %v264.1, %bb60.loopexit.unr-lcssa ]
  %lcmp.mod124 = icmp ne i64 %xtraiter121, 0
  tail call void @llvm.assume(i1 %lcmp.mod124)
  %v261.epil = getelementptr float, ptr %88, i64 %v256109.epil.init
  %v262.epil = load float, ptr %v261.epil, align 4
  %v263.epil = fdiv contract float %v262.epil, %v86.lcssa
  store float %v263.epil, ptr %v261.epil, align 4
  br label %bb60

bb60:                                             ; preds = %bb57.epil.preheader, %bb60.loopexit.unr-lcssa, %bb53, %entry
  ret void

bb65:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb68:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb69:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb70:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb71:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb72:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb73:                                             ; preds = %bb31
  tail call void @llvm.trap() #19
  unreachable

bb74:                                             ; preds = %bb32
  tail call void @llvm.trap() #19
  unreachable

bb75:                                             ; preds = %bb38
  tail call void @llvm.trap() #19
  unreachable

bb76:                                             ; preds = %bb39
  tail call void @llvm.trap() #19
  unreachable

bb77:                                             ; preds = %bb49
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @attention_heads(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, ptr captures(none) %v11, i64 %v12) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v33 = trunc i64 %v22.i to i32
  %v34.not = icmp ugt i32 %v6, %v33
  br i1 %v34.not, label %bb3, label %bb35

bb3:                                              ; preds = %entry
  %v36 = zext i32 %v8 to i64
  %v37 = zext i32 %v9 to i64
  %v12.i = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v7, i32 1)
  %v41 = udiv i32 %v6, %v12.i
  %v12.i13 = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v41, i32 1)
  %v45 = udiv i32 %v33, %v12.i13
  %v46 = zext i32 %v45 to i64
  %v47 = and i64 %v22.i, 4294967295
  %v48 = mul nuw i64 %v47, %v36
  %v50.not19.not = icmp eq i32 %v8, 0
  br i1 %v50.not19.not, label %bb11.preheader, label %bb9.lr.ph

bb9.lr.ph:                                        ; preds = %bb3
  %0 = getelementptr float, ptr %v11, i64 %v48
  %xtraiter = and i64 %v36, 7
  %1 = icmp ult i32 %v8, 8
  br i1 %1, label %bb9.epil.preheader, label %bb9.lr.ph.new

bb9.lr.ph.new:                                    ; preds = %bb9.lr.ph
  %unroll_iter = and i64 %v36, 4294967288
  br label %bb9

bb11.preheader.loopexit.unr-lcssa:                ; preds = %bb9
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb11.preheader, label %bb9.epil.preheader

bb9.epil.preheader:                               ; preds = %bb11.preheader.loopexit.unr-lcssa, %bb9.lr.ph
  %v4920.epil.init = phi i64 [ 0, %bb9.lr.ph ], [ %v55.7, %bb11.preheader.loopexit.unr-lcssa ]
  %lcmp.mod53 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod53)
  br label %bb9.epil

bb9.epil:                                         ; preds = %bb9.epil, %bb9.epil.preheader
  %v4920.epil = phi i64 [ %v4920.epil.init, %bb9.epil.preheader ], [ %v55.epil, %bb9.epil ]
  %epil.iter = phi i64 [ 0, %bb9.epil.preheader ], [ %epil.iter.next, %bb9.epil ]
  %v54.epil = getelementptr float, ptr %0, i64 %v4920.epil
  store float 0.000000e+00, ptr %v54.epil, align 4
  %v55.epil = add nuw nsw i64 %v4920.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %bb11.preheader, label %bb9.epil, !llvm.loop !5

bb11.preheader:                                   ; preds = %bb11.preheader.loopexit.unr-lcssa, %bb9.epil, %bb3
  %v60.not26.not = icmp eq i32 %v9, 0
  br i1 %v60.not26.not, label %bb28, label %bb12.lr.ph

bb12.lr.ph:                                       ; preds = %bb11.preheader
  %v62 = zext i32 %v12.i to i64
  %2 = getelementptr float, ptr %v11, i64 %v48
  %3 = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v48)
  %4 = mul nuw i64 %v46, %v36
  %5 = mul nuw i64 %v62, %v36
  %6 = sub i64 0, %4
  %7 = add nsw i64 %v36, -1
  %invariant.gep = getelementptr float, ptr %v0, i64 %v48
  %xtraiter54 = and i64 %v36, 3
  %8 = icmp ult i32 %v8, 4
  %unroll_iter59 = and i64 %v36, 4294967292
  %lcmp.mod56.not = icmp eq i64 %xtraiter54, 0
  %lcmp.mod58 = icmp ne i64 %xtraiter54, 0
  %xtraiter61 = and i64 %v36, 1
  %9 = icmp eq i32 %v8, 1
  %unroll_iter65 = and i64 %v36, 4294967294
  %lcmp.mod63.not = icmp eq i64 %xtraiter61, 0
  %lcmp.mod64 = icmp ne i64 %xtraiter61, 0
  br label %bb12

bb9:                                              ; preds = %bb9, %bb9.lr.ph.new
  %v4920 = phi i64 [ 0, %bb9.lr.ph.new ], [ %v55.7, %bb9 ]
  %niter = phi i64 [ 0, %bb9.lr.ph.new ], [ %niter.next.7, %bb9 ]
  %v54 = getelementptr float, ptr %0, i64 %v4920
  store float 0.000000e+00, ptr %v54, align 4
  %10 = getelementptr float, ptr %0, i64 %v4920
  %v54.1 = getelementptr i8, ptr %10, i64 4
  store float 0.000000e+00, ptr %v54.1, align 4
  %11 = getelementptr float, ptr %0, i64 %v4920
  %v54.2 = getelementptr i8, ptr %11, i64 8
  store float 0.000000e+00, ptr %v54.2, align 4
  %12 = getelementptr float, ptr %0, i64 %v4920
  %v54.3 = getelementptr i8, ptr %12, i64 12
  store float 0.000000e+00, ptr %v54.3, align 4
  %13 = getelementptr float, ptr %0, i64 %v4920
  %v54.4 = getelementptr i8, ptr %13, i64 16
  store float 0.000000e+00, ptr %v54.4, align 4
  %14 = getelementptr float, ptr %0, i64 %v4920
  %v54.5 = getelementptr i8, ptr %14, i64 20
  store float 0.000000e+00, ptr %v54.5, align 4
  %15 = getelementptr float, ptr %0, i64 %v4920
  %v54.6 = getelementptr i8, ptr %15, i64 24
  store float 0.000000e+00, ptr %v54.6, align 4
  %16 = getelementptr float, ptr %0, i64 %v4920
  %v54.7 = getelementptr i8, ptr %16, i64 28
  store float 0.000000e+00, ptr %v54.7, align 4
  %v55.7 = add nuw nsw i64 %v4920, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %bb11.preheader.loopexit.unr-lcssa, label %bb9

bb12:                                             ; preds = %bb12.lr.ph, %bb27
  %indvars.iv36 = phi i64 [ %6, %bb12.lr.ph ], [ %indvars.iv.next37, %bb27 ]
  %indvars.iv = phi i64 [ %4, %bb12.lr.ph ], [ %indvars.iv.next, %bb27 ]
  %v5930 = phi i64 [ 0, %bb12.lr.ph ], [ %v115, %bb27 ]
  %v8729 = phi i1 [ true, %bb12.lr.ph ], [ false, %bb27 ]
  %v5728 = phi float [ 0.000000e+00, %bb12.lr.ph ], [ %v129, %bb27 ]
  %v5627 = phi float [ 0.000000e+00, %bb12.lr.ph ], [ %v94, %bb27 ]
  %v63 = mul nuw i64 %v5930, %v62
  %v641 = add nuw i64 %v63, %v46
  %v66 = mul i64 %v641, %v36
  br i1 %v50.not19.not, label %bb17, label %bb14.preheader

bb14.preheader:                                   ; preds = %bb12
  %umax35 = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv)
  %17 = add i64 %umax35, %indvars.iv36
  %umin = tail call i64 @llvm.umin.i64(i64 %17, i64 %7)
  %18 = freeze i64 %umin
  %.not.not = icmp ugt i64 %3, %18
  br i1 %.not.not, label %bb14.preheader.split, label %bb40

bb14.preheader.split:                             ; preds = %bb14.preheader
  %.not = icmp eq i64 %17, %18
  br i1 %.not, label %bb41, label %bb14.preheader.split.split

bb14.preheader.split.split:                       ; preds = %bb14.preheader.split
  %invariant.gep47 = getelementptr float, ptr %v2, i64 %v66
  br i1 %8, label %bb14.epil.preheader, label %bb14

bb14:                                             ; preds = %bb14.preheader.split.split, %bb14
  %v6823 = phi i64 [ %v85.3, %bb14 ], [ 0, %bb14.preheader.split.split ]
  %v6722 = phi float [ %v84.3, %bb14 ], [ 0.000000e+00, %bb14.preheader.split.split ]
  %niter60 = phi i64 [ %niter60.next.3, %bb14 ], [ 0, %bb14.preheader.split.split ]
  %gep = getelementptr float, ptr %invariant.gep, i64 %v6823
  %v76 = load float, ptr %gep, align 4
  %gep48 = getelementptr float, ptr %invariant.gep47, i64 %v6823
  %v82 = load float, ptr %gep48, align 4
  %v83 = fmul contract float %v76, %v82
  %v84 = fadd contract float %v6722, %v83
  %v85 = or disjoint i64 %v6823, 1
  %gep.1 = getelementptr float, ptr %invariant.gep, i64 %v85
  %v76.1 = load float, ptr %gep.1, align 4
  %gep48.1 = getelementptr float, ptr %invariant.gep47, i64 %v85
  %v82.1 = load float, ptr %gep48.1, align 4
  %v83.1 = fmul contract float %v76.1, %v82.1
  %v84.1 = fadd contract float %v84, %v83.1
  %v85.1 = or disjoint i64 %v6823, 2
  %gep.2 = getelementptr float, ptr %invariant.gep, i64 %v85.1
  %v76.2 = load float, ptr %gep.2, align 4
  %gep48.2 = getelementptr float, ptr %invariant.gep47, i64 %v85.1
  %v82.2 = load float, ptr %gep48.2, align 4
  %v83.2 = fmul contract float %v76.2, %v82.2
  %v84.2 = fadd contract float %v84.1, %v83.2
  %v85.2 = or disjoint i64 %v6823, 3
  %gep.3 = getelementptr float, ptr %invariant.gep, i64 %v85.2
  %v76.3 = load float, ptr %gep.3, align 4
  %gep48.3 = getelementptr float, ptr %invariant.gep47, i64 %v85.2
  %v82.3 = load float, ptr %gep48.3, align 4
  %v83.3 = fmul contract float %v76.3, %v82.3
  %v84.3 = fadd contract float %v84.2, %v83.3
  %v85.3 = add nuw nsw i64 %v6823, 4
  %niter60.next.3 = add i64 %niter60, 4
  %niter60.ncmp.3 = icmp eq i64 %niter60.next.3, %unroll_iter59
  br i1 %niter60.ncmp.3, label %bb17.loopexit.unr-lcssa, label %bb14

bb17.loopexit.unr-lcssa:                          ; preds = %bb14
  br i1 %lcmp.mod56.not, label %bb17, label %bb14.epil.preheader

bb14.epil.preheader:                              ; preds = %bb17.loopexit.unr-lcssa, %bb14.preheader.split.split
  %v6823.epil.init = phi i64 [ 0, %bb14.preheader.split.split ], [ %v85.3, %bb17.loopexit.unr-lcssa ]
  %v6722.epil.init = phi float [ 0.000000e+00, %bb14.preheader.split.split ], [ %v84.3, %bb17.loopexit.unr-lcssa ]
  tail call void @llvm.assume(i1 %lcmp.mod58)
  br label %bb14.epil

bb14.epil:                                        ; preds = %bb14.epil, %bb14.epil.preheader
  %v6823.epil = phi i64 [ %v85.epil, %bb14.epil ], [ %v6823.epil.init, %bb14.epil.preheader ]
  %v6722.epil = phi float [ %v84.epil, %bb14.epil ], [ %v6722.epil.init, %bb14.epil.preheader ]
  %epil.iter55 = phi i64 [ %epil.iter55.next, %bb14.epil ], [ 0, %bb14.epil.preheader ]
  %gep.epil = getelementptr float, ptr %invariant.gep, i64 %v6823.epil
  %v76.epil = load float, ptr %gep.epil, align 4
  %gep48.epil = getelementptr float, ptr %invariant.gep47, i64 %v6823.epil
  %v82.epil = load float, ptr %gep48.epil, align 4
  %v83.epil = fmul contract float %v76.epil, %v82.epil
  %v84.epil = fadd contract float %v6722.epil, %v83.epil
  %v85.epil = add nuw nsw i64 %v6823.epil, 1
  %epil.iter55.next = add i64 %epil.iter55, 1
  %epil.iter55.cmp.not = icmp eq i64 %epil.iter55.next, %xtraiter54
  br i1 %epil.iter55.cmp.not, label %bb17, label %bb14.epil, !llvm.loop !6

bb17:                                             ; preds = %bb17.loopexit.unr-lcssa, %bb14.epil, %bb12
  %v67.lcssa = phi float [ 0.000000e+00, %bb12 ], [ %v84.3, %bb17.loopexit.unr-lcssa ], [ %v84.epil, %bb14.epil ]
  %v86 = fmul contract float %v10, %v67.lcssa
  br i1 %v8729, label %bb23, label %bb18

bb18:                                             ; preds = %bb17
  %v88 = fcmp ule float %v86, %v5627
  br i1 %v88, label %bb23, label %bb20

bb20:                                             ; preds = %bb18
  %v90 = fsub contract float %v5627, %v86
  %19 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %19, 0
  %20 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v90, float 0x3F777313A0000000, float 5.000000e-01) #20
  %21 = tail call float @llvm.fma.f32(float %v90, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %21, float %20
  %22 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %23 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %23, float %22
  %24 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %25 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %25, float %24
  %26 = fadd float %.04.i, 0xC168000FE0000000
  %27 = fneg float %26
  %28 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v90, float 0x3FF7154760000000, float %27) #20
  %29 = tail call float @llvm.fma.f32(float %v90, float 0x3FF7154760000000, float %27)
  %.0.i = select i1 %.not.i, float %29, float %28
  %30 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v90, float 0x3E54AE0C00000000, float %.0.i) #20
  %31 = tail call float @llvm.fma.f32(float %v90, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %31, float %30
  %32 = bitcast float %.04.i to i32
  %33 = shl i32 %32, 23
  %34 = bitcast i32 %33 to float
  %35 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %36 = fmul float %35, %34
  br label %bb23

bb23:                                             ; preds = %bb20, %bb18, %bb17
  %v94 = phi float [ %v86, %bb17 ], [ %v86, %bb20 ], [ %v5627, %bb18 ]
  %v95 = phi float [ 0.000000e+00, %bb17 ], [ %36, %bb20 ], [ 1.000000e+00, %bb18 ]
  %v96 = fsub contract float %v86, %v94
  %37 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i7 = icmp eq i32 %37, 0
  %38 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v96, float 0x3F777313A0000000, float 5.000000e-01) #20
  %39 = tail call float @llvm.fma.f32(float %v96, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i8 = select i1 %.not.i7, float %39, float %38
  %40 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i8) #20
  %41 = tail call float @llvm.nvvm.saturate.f(float %.02.i8) #20
  %.03.i9 = select i1 %.not.i7, float %41, float %40
  %42 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i9, float 2.520000e+02, float 0x4168000020000000) #20
  %43 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i9, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i10 = select i1 %.not.i7, float %43, float %42
  %44 = fadd float %.04.i10, 0xC168000FE0000000
  %45 = fneg float %44
  %46 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v96, float 0x3FF7154760000000, float %45) #20
  %47 = tail call float @llvm.fma.f32(float %v96, float 0x3FF7154760000000, float %45)
  %.0.i11 = select i1 %.not.i7, float %47, float %46
  %48 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v96, float 0x3E54AE0C00000000, float %.0.i11) #20
  %49 = tail call float @llvm.fma.f32(float %v96, float 0x3E54AE0C00000000, float %.0.i11)
  %.01.i12 = select i1 %.not.i7, float %49, float %48
  %50 = bitcast float %.04.i10 to i32
  %51 = shl i32 %50, 23
  %52 = bitcast i32 %51 to float
  %53 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i12)
  %54 = fmul float %53, %52
  %v128 = fmul contract float %v5728, %v95
  %v129 = fadd contract float %v128, %54
  br i1 %v50.not19.not, label %bb27, label %bb25.preheader

bb25.preheader:                                   ; preds = %bb23
  %umax40 = tail call i64 @llvm.umax.i64(i64 %v5, i64 %indvars.iv)
  %55 = add i64 %umax40, %indvars.iv36
  %.not45.not = icmp ugt i64 %55, %7
  br i1 %.not45.not, label %bb25.preheader.split, label %bb42

bb25.preheader.split:                             ; preds = %bb25.preheader
  %invariant.gep49 = getelementptr float, ptr %v4, i64 %v66
  br i1 %9, label %bb25.epil.preheader, label %bb25

bb25:                                             ; preds = %bb25.preheader.split, %bb25
  %v9825 = phi i64 [ %v114.1, %bb25 ], [ 0, %bb25.preheader.split ]
  %niter66 = phi i64 [ %niter66.next.1, %bb25 ], [ 0, %bb25.preheader.split ]
  %v103 = getelementptr float, ptr %2, i64 %v9825
  %v104 = load float, ptr %v103, align 4
  %v105 = fmul contract float %v95, %v104
  %gep50 = getelementptr float, ptr %invariant.gep49, i64 %v9825
  %v111 = load float, ptr %gep50, align 4
  %v112 = fmul contract float %54, %v111
  %v113 = fadd contract float %v105, %v112
  store float %v113, ptr %v103, align 4
  %v114 = or disjoint i64 %v9825, 1
  %v103.1 = getelementptr float, ptr %2, i64 %v114
  %v104.1 = load float, ptr %v103.1, align 4
  %v105.1 = fmul contract float %v95, %v104.1
  %gep50.1 = getelementptr float, ptr %invariant.gep49, i64 %v114
  %v111.1 = load float, ptr %gep50.1, align 4
  %v112.1 = fmul contract float %54, %v111.1
  %v113.1 = fadd contract float %v105.1, %v112.1
  store float %v113.1, ptr %v103.1, align 4
  %v114.1 = add nuw nsw i64 %v9825, 2
  %niter66.next.1 = add i64 %niter66, 2
  %niter66.ncmp.1 = icmp eq i64 %niter66.next.1, %unroll_iter65
  br i1 %niter66.ncmp.1, label %bb27.loopexit.unr-lcssa, label %bb25

bb27.loopexit.unr-lcssa:                          ; preds = %bb25
  br i1 %lcmp.mod63.not, label %bb27, label %bb25.epil.preheader

bb25.epil.preheader:                              ; preds = %bb27.loopexit.unr-lcssa, %bb25.preheader.split
  %v9825.epil.init = phi i64 [ 0, %bb25.preheader.split ], [ %v114.1, %bb27.loopexit.unr-lcssa ]
  tail call void @llvm.assume(i1 %lcmp.mod64)
  %v103.epil = getelementptr float, ptr %2, i64 %v9825.epil.init
  %v104.epil = load float, ptr %v103.epil, align 4
  %v105.epil = fmul contract float %v95, %v104.epil
  %gep50.epil = getelementptr float, ptr %invariant.gep49, i64 %v9825.epil.init
  %v111.epil = load float, ptr %gep50.epil, align 4
  %v112.epil = fmul contract float %54, %v111.epil
  %v113.epil = fadd contract float %v105.epil, %v112.epil
  store float %v113.epil, ptr %v103.epil, align 4
  br label %bb27

bb27:                                             ; preds = %bb25.epil.preheader, %bb27.loopexit.unr-lcssa, %bb23
  %v115 = add nuw nsw i64 %v5930, 1
  %indvars.iv.next = add i64 %indvars.iv, %5
  %indvars.iv.next37 = sub i64 %indvars.iv36, %5
  %exitcond43.not = icmp eq i64 %v115, %v37
  br i1 %exitcond43.not, label %bb28, label %bb12

bb28:                                             ; preds = %bb27, %bb11.preheader
  %v57.lcssa = phi float [ 0.000000e+00, %bb11.preheader ], [ %v129, %bb27 ]
  %v116 = fcmp ule float %v57.lcssa, 0.000000e+00
  br i1 %v116, label %bb35, label %bb29

bb29:                                             ; preds = %bb28
  %v118 = fdiv contract float 1.000000e+00, %v57.lcssa
  br i1 %v50.not19.not, label %bb35, label %bb32.lr.ph

bb32.lr.ph:                                       ; preds = %bb29
  %56 = getelementptr float, ptr %v11, i64 %v48
  %xtraiter67 = and i64 %v36, 3
  %57 = icmp ult i32 %v8, 4
  br i1 %57, label %bb32.epil.preheader, label %bb32.lr.ph.new

bb32.lr.ph.new:                                   ; preds = %bb32.lr.ph
  %unroll_iter71 = and i64 %v36, 4294967292
  br label %bb32

bb32:                                             ; preds = %bb32, %bb32.lr.ph.new
  %v11933 = phi i64 [ 0, %bb32.lr.ph.new ], [ %v127.3, %bb32 ]
  %niter72 = phi i64 [ 0, %bb32.lr.ph.new ], [ %niter72.next.3, %bb32 ]
  %v124 = getelementptr float, ptr %56, i64 %v11933
  %v125 = load float, ptr %v124, align 4
  %v126 = fmul contract float %v118, %v125
  store float %v126, ptr %v124, align 4
  %58 = getelementptr float, ptr %56, i64 %v11933
  %v124.1 = getelementptr i8, ptr %58, i64 4
  %v125.1 = load float, ptr %v124.1, align 4
  %v126.1 = fmul contract float %v118, %v125.1
  store float %v126.1, ptr %v124.1, align 4
  %59 = getelementptr float, ptr %56, i64 %v11933
  %v124.2 = getelementptr i8, ptr %59, i64 8
  %v125.2 = load float, ptr %v124.2, align 4
  %v126.2 = fmul contract float %v118, %v125.2
  store float %v126.2, ptr %v124.2, align 4
  %60 = getelementptr float, ptr %56, i64 %v11933
  %v124.3 = getelementptr i8, ptr %60, i64 12
  %v125.3 = load float, ptr %v124.3, align 4
  %v126.3 = fmul contract float %v118, %v125.3
  store float %v126.3, ptr %v124.3, align 4
  %v127.3 = add nuw nsw i64 %v11933, 4
  %niter72.next.3 = add i64 %niter72, 4
  %niter72.ncmp.3 = icmp eq i64 %niter72.next.3, %unroll_iter71
  br i1 %niter72.ncmp.3, label %bb35.loopexit.unr-lcssa, label %bb32

bb35.loopexit.unr-lcssa:                          ; preds = %bb32
  %lcmp.mod69.not = icmp eq i64 %xtraiter67, 0
  br i1 %lcmp.mod69.not, label %bb35, label %bb32.epil.preheader

bb32.epil.preheader:                              ; preds = %bb35.loopexit.unr-lcssa, %bb32.lr.ph
  %v11933.epil.init = phi i64 [ 0, %bb32.lr.ph ], [ %v127.3, %bb35.loopexit.unr-lcssa ]
  %lcmp.mod70 = icmp ne i64 %xtraiter67, 0
  tail call void @llvm.assume(i1 %lcmp.mod70)
  br label %bb32.epil

bb32.epil:                                        ; preds = %bb32.epil, %bb32.epil.preheader
  %v11933.epil = phi i64 [ %v11933.epil.init, %bb32.epil.preheader ], [ %v127.epil, %bb32.epil ]
  %epil.iter68 = phi i64 [ 0, %bb32.epil.preheader ], [ %epil.iter68.next, %bb32.epil ]
  %v124.epil = getelementptr float, ptr %56, i64 %v11933.epil
  %v125.epil = load float, ptr %v124.epil, align 4
  %v126.epil = fmul contract float %v118, %v125.epil
  store float %v126.epil, ptr %v124.epil, align 4
  %v127.epil = add nuw nsw i64 %v11933.epil, 1
  %epil.iter68.next = add i64 %epil.iter68, 1
  %epil.iter68.cmp.not = icmp eq i64 %epil.iter68.next, %xtraiter67
  br i1 %epil.iter68.cmp.not, label %bb35, label %bb32.epil, !llvm.loop !7

bb35:                                             ; preds = %bb35.loopexit.unr-lcssa, %bb32.epil, %bb29, %bb28, %entry
  ret void

bb40:                                             ; preds = %bb14.preheader
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb14.preheader.split
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb25.preheader
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @attention_paged_heads(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, ptr captures(none) %v16, i64 %v17) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i3 = icmp eq i32 %v4.i1, 1
  %v7.i4 = icmp eq i32 %v6.i2, 1
  %v8.not.not.i = and i1 %v5.i3, %v7.i4
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i5 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i5
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v43 = trunc i64 %v22.i to i32
  %v44.not = icmp ugt i32 %v6, %v43
  br i1 %v44.not, label %bb3, label %bb41

bb3:                                              ; preds = %entry
  %v46 = zext i32 %v8 to i64
  %v47 = zext i32 %v9 to i64
  %v12.i = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v7, i32 1)
  %v51 = udiv i32 %v6, %v12.i
  %v12.i12 = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v51, i32 1)
  %v55 = udiv i32 %v43, %v12.i12
  %v56 = zext i32 %v55 to i64
  %v57 = zext i32 %v15 to i64
  %v58 = zext i32 %v13 to i64
  %v59 = zext i32 %v14 to i64
  %v60 = shl nuw nsw i64 %v57, 1
  %v61 = mul i64 %v60, %v58
  %v62 = and i64 %v22.i, 4294967295
  %v63 = mul nuw i64 %v62, %v46
  %v65.not65.not = icmp eq i32 %v8, 0
  br i1 %v65.not65.not, label %bb11.preheader, label %bb9.lr.ph

bb9.lr.ph:                                        ; preds = %bb3
  %0 = getelementptr float, ptr %v16, i64 %v63
  %xtraiter = and i64 %v46, 7
  %1 = icmp ult i32 %v8, 8
  br i1 %1, label %bb9.epil.preheader, label %bb9.lr.ph.new

bb9.lr.ph.new:                                    ; preds = %bb9.lr.ph
  %unroll_iter = and i64 %v46, 4294967288
  br label %bb9

bb11.preheader.loopexit.unr-lcssa:                ; preds = %bb9
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb11.preheader, label %bb9.epil.preheader

bb9.epil.preheader:                               ; preds = %bb11.preheader.loopexit.unr-lcssa, %bb9.lr.ph
  %v6466.epil.init = phi i64 [ 0, %bb9.lr.ph ], [ %v70.7, %bb11.preheader.loopexit.unr-lcssa ]
  %lcmp.mod93 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod93)
  br label %bb9.epil

bb9.epil:                                         ; preds = %bb9.epil, %bb9.epil.preheader
  %v6466.epil = phi i64 [ %v6466.epil.init, %bb9.epil.preheader ], [ %v70.epil, %bb9.epil ]
  %epil.iter = phi i64 [ 0, %bb9.epil.preheader ], [ %epil.iter.next, %bb9.epil ]
  %v69.epil = getelementptr float, ptr %0, i64 %v6466.epil
  store float 0.000000e+00, ptr %v69.epil, align 4
  %v70.epil = add nuw nsw i64 %v6466.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %bb11.preheader, label %bb9.epil, !llvm.loop !8

bb11.preheader:                                   ; preds = %bb11.preheader.loopexit.unr-lcssa, %bb9.epil, %bb3
  %v75.not72.not = icmp eq i32 %v9, 0
  br i1 %v75.not72.not, label %bb34, label %bb12.lr.ph

bb12.lr.ph:                                       ; preds = %bb11.preheader
  %factor.op.mul = mul nuw i64 %v46, %v56
  %v77.not = icmp eq i32 %v13, 0
  %v81 = zext i32 %v11 to i64
  %v82 = zext i32 %v12 to i64
  %v83 = mul nuw i64 %v82, %v81
  %v94.reass = shl i64 %factor.op.mul, 1
  %invariant.op = add i64 %v61, %v94.reass
  %2 = getelementptr float, ptr %v16, i64 %v63
  br i1 %v77.not, label %bb46, label %bb12

bb9:                                              ; preds = %bb9, %bb9.lr.ph.new
  %v6466 = phi i64 [ 0, %bb9.lr.ph.new ], [ %v70.7, %bb9 ]
  %niter = phi i64 [ 0, %bb9.lr.ph.new ], [ %niter.next.7, %bb9 ]
  %v69 = getelementptr float, ptr %0, i64 %v6466
  store float 0.000000e+00, ptr %v69, align 4
  %3 = getelementptr float, ptr %0, i64 %v6466
  %v69.1 = getelementptr i8, ptr %3, i64 4
  store float 0.000000e+00, ptr %v69.1, align 4
  %4 = getelementptr float, ptr %0, i64 %v6466
  %v69.2 = getelementptr i8, ptr %4, i64 8
  store float 0.000000e+00, ptr %v69.2, align 4
  %5 = getelementptr float, ptr %0, i64 %v6466
  %v69.3 = getelementptr i8, ptr %5, i64 12
  store float 0.000000e+00, ptr %v69.3, align 4
  %6 = getelementptr float, ptr %0, i64 %v6466
  %v69.4 = getelementptr i8, ptr %6, i64 16
  store float 0.000000e+00, ptr %v69.4, align 4
  %7 = getelementptr float, ptr %0, i64 %v6466
  %v69.5 = getelementptr i8, ptr %7, i64 20
  store float 0.000000e+00, ptr %v69.5, align 4
  %8 = getelementptr float, ptr %0, i64 %v6466
  %v69.6 = getelementptr i8, ptr %8, i64 24
  store float 0.000000e+00, ptr %v69.6, align 4
  %9 = getelementptr float, ptr %0, i64 %v6466
  %v69.7 = getelementptr i8, ptr %9, i64 28
  store float 0.000000e+00, ptr %v69.7, align 4
  %v70.7 = add nuw nsw i64 %v6466, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %bb11.preheader.loopexit.unr-lcssa, label %bb9

bb12:                                             ; preds = %bb12.lr.ph, %bb33
  %v7476 = phi i64 [ %v175, %bb33 ], [ 0, %bb12.lr.ph ]
  %v13275 = phi i1 [ false, %bb33 ], [ true, %bb12.lr.ph ]
  %v7274 = phi float [ %v189, %bb33 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v7173 = phi float [ %v139, %bb33 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v7476.frozen = freeze i64 %v7476
  %v58.frozen = freeze i64 %v58
  %v79 = udiv i64 %v7476.frozen, %v58.frozen
  %v84 = add i64 %v79, %v83
  %v86 = icmp ult i64 %v84, %v5
  br i1 %v86, label %bb14, label %bb47

bb14:                                             ; preds = %bb12
  %10 = mul i64 %v79, %v58.frozen
  %v80.decomposed = sub i64 %v7476.frozen, %10
  %v88 = getelementptr inbounds i32, ptr %v4, i64 %v84
  %v89 = load i32, ptr %v88, align 4
  %v90 = zext i32 %v89 to i64
  %v91 = mul nuw i64 %v90, %v59
  %v92 = mul i64 %v80.decomposed, %v60
  %v93 = add i64 %v91, %v92
  %v96 = add i64 %v93, %v94.reass
  br i1 %v65.not65.not, label %bb21, label %bb16

bb16:                                             ; preds = %bb14, %bb20
  %v9869 = phi i64 [ %v130, %bb20 ], [ 0, %bb14 ]
  %v9768 = phi float [ %v129, %bb20 ], [ 0.000000e+00, %bb14 ]
  %v101 = shl nuw i64 %v9869, 1
  %v102 = add i64 %v96, %v101
  %v104 = icmp ult i64 %v102, %v3
  br i1 %v104, label %bb17, label %bb48

bb17:                                             ; preds = %bb16
  %v111 = add nuw i64 %v102, 1
  %v112 = icmp ult i64 %v111, %v3
  br i1 %v112, label %bb18, label %bb49

bb18:                                             ; preds = %bb17
  %v106 = getelementptr inbounds i8, ptr %v2, i64 %v102
  %v107 = load i8, ptr %v106, align 1
  %v108 = zext i8 %v107 to i16
  %v114 = getelementptr inbounds i8, ptr %v2, i64 %v111
  %v115 = load i8, ptr %v114, align 1
  %v116 = zext i8 %v115 to i16
  %v119 = shl nuw i16 %v116, 8
  %v4.i13 = lshr i16 %v116, 7
  %v6.i14 = zext nneg i16 %v4.i13 to i32
  %v9.i = lshr i16 %v116, 2
  %v10.i = and i16 %v9.i, 31
  %v119.masked = and i16 %v119, 768
  %v12.i15 = or disjoint i16 %v119.masked, %v108
  %v13.i16 = zext nneg i16 %v12.i15 to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb18
  %v15.i17 = icmp eq i16 %v12.i15, 0
  br i1 %v15.i17, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i18 = shl nuw i32 %v6.i14, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i16, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i16, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i14, 31
  %11 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %11
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb18
  %v38.i = shl nuw i32 %v6.i14, 31
  %v41.i = shl nuw nsw i32 %v13.i16, 13
  %v39.i = or disjoint i32 %v41.i, %v38.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb18
  %v44.i = shl nuw i32 %v6.i14, 31
  %12 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %12 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i16, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i18, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v122 = add nuw i64 %v9869, %v63
  %v124 = icmp ult i64 %v122, %v1
  br i1 %v124, label %bb20, label %bb50

bb20:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v55.i = bitcast i32 %v54.i to float
  %v126 = getelementptr inbounds float, ptr %v0, i64 %v122
  %v127 = load float, ptr %v126, align 4
  %v128 = fmul contract float %v127, %v55.i
  %v129 = fadd contract float %v9768, %v128
  %v130 = add nuw nsw i64 %v9869, 1
  %exitcond83.not = icmp eq i64 %v130, %v46
  br i1 %exitcond83.not, label %bb21, label %bb16

bb21:                                             ; preds = %bb20, %bb14
  %v97.lcssa = phi float [ 0.000000e+00, %bb14 ], [ %v129, %bb20 ]
  %v131 = fmul contract float %v10, %v97.lcssa
  br i1 %v13275, label %bb27, label %bb22

bb22:                                             ; preds = %bb21
  %v133 = fcmp ule float %v131, %v7173
  br i1 %v133, label %bb27, label %bb24

bb24:                                             ; preds = %bb22
  %v135 = fsub contract float %v7173, %v131
  %13 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %13, 0
  %14 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v135, float 0x3F777313A0000000, float 5.000000e-01) #20
  %15 = tail call float @llvm.fma.f32(float %v135, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %15, float %14
  %16 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %17 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %17, float %16
  %18 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %19 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %19, float %18
  %20 = fadd float %.04.i, 0xC168000FE0000000
  %21 = fneg float %20
  %22 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v135, float 0x3FF7154760000000, float %21) #20
  %23 = tail call float @llvm.fma.f32(float %v135, float 0x3FF7154760000000, float %21)
  %.0.i = select i1 %.not.i, float %23, float %22
  %24 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v135, float 0x3E54AE0C00000000, float %.0.i) #20
  %25 = tail call float @llvm.fma.f32(float %v135, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %25, float %24
  %26 = bitcast float %.04.i to i32
  %27 = shl i32 %26, 23
  %28 = bitcast i32 %27 to float
  %29 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %30 = fmul float %29, %28
  br label %bb27

bb27:                                             ; preds = %bb24, %bb22, %bb21
  %v139 = phi float [ %v131, %bb21 ], [ %v131, %bb24 ], [ %v7173, %bb22 ]
  %v140 = phi float [ 0.000000e+00, %bb21 ], [ %30, %bb24 ], [ 1.000000e+00, %bb22 ]
  %v141 = fsub contract float %v131, %v139
  %31 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i6 = icmp eq i32 %31, 0
  %32 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v141, float 0x3F777313A0000000, float 5.000000e-01) #20
  %33 = tail call float @llvm.fma.f32(float %v141, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i7 = select i1 %.not.i6, float %33, float %32
  %34 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i7) #20
  %35 = tail call float @llvm.nvvm.saturate.f(float %.02.i7) #20
  %.03.i8 = select i1 %.not.i6, float %35, float %34
  %36 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i8, float 2.520000e+02, float 0x4168000020000000) #20
  %37 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i8, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i9 = select i1 %.not.i6, float %37, float %36
  %38 = fadd float %.04.i9, 0xC168000FE0000000
  %39 = fneg float %38
  %40 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v141, float 0x3FF7154760000000, float %39) #20
  %41 = tail call float @llvm.fma.f32(float %v141, float 0x3FF7154760000000, float %39)
  %.0.i10 = select i1 %.not.i6, float %41, float %40
  %42 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v141, float 0x3E54AE0C00000000, float %.0.i10) #20
  %43 = tail call float @llvm.fma.f32(float %v141, float 0x3E54AE0C00000000, float %.0.i10)
  %.01.i11 = select i1 %.not.i6, float %43, float %42
  %44 = bitcast float %.04.i9 to i32
  %45 = shl i32 %44, 23
  %46 = bitcast i32 %45 to float
  %47 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i11)
  %48 = fmul float %47, %46
  %v188 = fmul contract float %v7274, %v140
  %v189 = fadd contract float %v188, %48
  br i1 %v65.not65.not, label %bb33, label %bb29.lr.ph

bb29.lr.ph:                                       ; preds = %bb27
  %v192.reass = add i64 %v93, %invariant.op
  br label %bb29

bb29:                                             ; preds = %bb29.lr.ph, %cuda_kernels__oxide_kernels__f16_to_f32.exit54
  %v14371 = phi i64 [ 0, %bb29.lr.ph ], [ %v174, %cuda_kernels__oxide_kernels__f16_to_f32.exit54 ]
  %v146 = shl nuw i64 %v14371, 1
  %v147 = add i64 %v192.reass, %v146
  %v149 = icmp ult i64 %v147, %v3
  br i1 %v149, label %bb30, label %bb51

bb30:                                             ; preds = %bb29
  %v156 = add nuw i64 %v147, 1
  %v157 = icmp ult i64 %v156, %v3
  br i1 %v157, label %bb31, label %bb52

bb31:                                             ; preds = %bb30
  %v151 = getelementptr inbounds i8, ptr %v2, i64 %v147
  %v152 = load i8, ptr %v151, align 1
  %v153 = zext i8 %v152 to i16
  %v159 = getelementptr inbounds i8, ptr %v2, i64 %v156
  %v160 = load i8, ptr %v159, align 1
  %v161 = zext i8 %v160 to i16
  %v164 = shl nuw i16 %v161, 8
  %v4.i19 = lshr i16 %v161, 7
  %v6.i20 = zext nneg i16 %v4.i19 to i32
  %v9.i21 = lshr i16 %v161, 2
  %v10.i22 = and i16 %v9.i21, 31
  %v164.masked = and i16 %v164, 768
  %v12.i23 = or disjoint i16 %v164.masked, %v153
  %v13.i24 = zext nneg i16 %v12.i23 to i32
  switch i16 %v10.i22, label %bb10.i47 [
    i16 0, label %bb1.i32
    i16 31, label %bb9.i25
  ]

bb1.i32:                                          ; preds = %bb31
  %v15.i33 = icmp eq i16 %v12.i23, 0
  br i1 %v15.i33, label %bb2.i45, label %bb6.i34

bb2.i45:                                          ; preds = %bb1.i32
  %v17.i46 = shl nuw i32 %v6.i20, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

bb6.i34:                                          ; preds = %bb1.i32
  %v13.masked.numleadingzeros.i35 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i24, i1 true)
  %v13.masked.leadingonepos.i36 = xor i32 %v13.masked.numleadingzeros.i35, 31
  %bb5.tripcount.i37 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i36
  %v23.i38 = shl nuw nsw i32 %v13.i24, %bb5.tripcount.i37
  %v27.i39 = shl nuw i32 %v6.i20, 31
  %49 = shl nuw nsw i32 %v13.masked.numleadingzeros.i35, 23
  %reass.sub80 = sub i32 %v27.i39, %49
  %v31.i41 = add i32 %reass.sub80, 1124073472
  %v25.i42 = shl i32 %v23.i38, 13
  %v33.i43 = and i32 %v25.i42, 8380416
  %v34.i44 = or disjoint i32 %v33.i43, %v31.i41
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

bb9.i25:                                          ; preds = %bb31
  %v38.i26 = shl nuw i32 %v6.i20, 31
  %v41.i27 = shl nuw nsw i32 %v13.i24, 13
  %v39.i28 = or disjoint i32 %v41.i27, %v38.i26
  %v42.i29 = or disjoint i32 %v39.i28, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

bb10.i47:                                         ; preds = %bb31
  %v44.i48 = shl nuw i32 %v6.i20, 31
  %50 = add nuw nsw i16 %v10.i22, 112
  %v46.i49 = zext nneg i16 %50 to i32
  %v48.i50 = shl nuw nsw i32 %v46.i49, 23
  %v49.i51 = or disjoint i32 %v48.i50, %v44.i48
  %v51.i52 = shl nuw nsw i32 %v13.i24, 13
  %v52.i53 = or disjoint i32 %v49.i51, %v51.i52
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

cuda_kernels__oxide_kernels__f16_to_f32.exit54:   ; preds = %bb2.i45, %bb6.i34, %bb9.i25, %bb10.i47
  %v54.i30 = phi i32 [ %v34.i44, %bb6.i34 ], [ %v17.i46, %bb2.i45 ], [ %v42.i29, %bb9.i25 ], [ %v52.i53, %bb10.i47 ]
  %v55.i31 = bitcast i32 %v54.i30 to float
  %v169 = getelementptr float, ptr %2, i64 %v14371
  %v170 = load float, ptr %v169, align 4
  %v171 = fmul contract float %v140, %v170
  %v172 = fmul contract float %48, %v55.i31
  %v173 = fadd contract float %v171, %v172
  store float %v173, ptr %v169, align 4
  %v174 = add nuw nsw i64 %v14371, 1
  %exitcond84.not = icmp eq i64 %v174, %v46
  br i1 %exitcond84.not, label %bb33, label %bb29

bb33:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit54, %bb27
  %v175 = add nuw nsw i64 %v7476, 1
  %exitcond85.not = icmp eq i64 %v175, %v47
  br i1 %exitcond85.not, label %bb34, label %bb12

bb34:                                             ; preds = %bb33, %bb11.preheader
  %v72.lcssa = phi float [ 0.000000e+00, %bb11.preheader ], [ %v189, %bb33 ]
  %v176 = fcmp ule float %v72.lcssa, 0.000000e+00
  br i1 %v176, label %bb41, label %bb35

bb35:                                             ; preds = %bb34
  %v178 = fdiv contract float 1.000000e+00, %v72.lcssa
  br i1 %v65.not65.not, label %bb41, label %bb38.lr.ph

bb38.lr.ph:                                       ; preds = %bb35
  %51 = getelementptr float, ptr %v16, i64 %v63
  %xtraiter94 = and i64 %v46, 3
  %52 = icmp ult i32 %v8, 4
  br i1 %52, label %bb38.epil.preheader, label %bb38.lr.ph.new

bb38.lr.ph.new:                                   ; preds = %bb38.lr.ph
  %unroll_iter98 = and i64 %v46, 4294967292
  br label %bb38

bb38:                                             ; preds = %bb38, %bb38.lr.ph.new
  %v17979 = phi i64 [ 0, %bb38.lr.ph.new ], [ %v187.3, %bb38 ]
  %niter99 = phi i64 [ 0, %bb38.lr.ph.new ], [ %niter99.next.3, %bb38 ]
  %v184 = getelementptr float, ptr %51, i64 %v17979
  %v185 = load float, ptr %v184, align 4
  %v186 = fmul contract float %v178, %v185
  store float %v186, ptr %v184, align 4
  %53 = getelementptr float, ptr %51, i64 %v17979
  %v184.1 = getelementptr i8, ptr %53, i64 4
  %v185.1 = load float, ptr %v184.1, align 4
  %v186.1 = fmul contract float %v178, %v185.1
  store float %v186.1, ptr %v184.1, align 4
  %54 = getelementptr float, ptr %51, i64 %v17979
  %v184.2 = getelementptr i8, ptr %54, i64 8
  %v185.2 = load float, ptr %v184.2, align 4
  %v186.2 = fmul contract float %v178, %v185.2
  store float %v186.2, ptr %v184.2, align 4
  %55 = getelementptr float, ptr %51, i64 %v17979
  %v184.3 = getelementptr i8, ptr %55, i64 12
  %v185.3 = load float, ptr %v184.3, align 4
  %v186.3 = fmul contract float %v178, %v185.3
  store float %v186.3, ptr %v184.3, align 4
  %v187.3 = add nuw nsw i64 %v17979, 4
  %niter99.next.3 = add i64 %niter99, 4
  %niter99.ncmp.3 = icmp eq i64 %niter99.next.3, %unroll_iter98
  br i1 %niter99.ncmp.3, label %bb41.loopexit.unr-lcssa, label %bb38

bb41.loopexit.unr-lcssa:                          ; preds = %bb38
  %lcmp.mod96.not = icmp eq i64 %xtraiter94, 0
  br i1 %lcmp.mod96.not, label %bb41, label %bb38.epil.preheader

bb38.epil.preheader:                              ; preds = %bb41.loopexit.unr-lcssa, %bb38.lr.ph
  %v17979.epil.init = phi i64 [ 0, %bb38.lr.ph ], [ %v187.3, %bb41.loopexit.unr-lcssa ]
  %lcmp.mod97 = icmp ne i64 %xtraiter94, 0
  tail call void @llvm.assume(i1 %lcmp.mod97)
  br label %bb38.epil

bb38.epil:                                        ; preds = %bb38.epil, %bb38.epil.preheader
  %v17979.epil = phi i64 [ %v17979.epil.init, %bb38.epil.preheader ], [ %v187.epil, %bb38.epil ]
  %epil.iter95 = phi i64 [ 0, %bb38.epil.preheader ], [ %epil.iter95.next, %bb38.epil ]
  %v184.epil = getelementptr float, ptr %51, i64 %v17979.epil
  %v185.epil = load float, ptr %v184.epil, align 4
  %v186.epil = fmul contract float %v178, %v185.epil
  store float %v186.epil, ptr %v184.epil, align 4
  %v187.epil = add nuw nsw i64 %v17979.epil, 1
  %epil.iter95.next = add i64 %epil.iter95, 1
  %epil.iter95.cmp.not = icmp eq i64 %epil.iter95.next, %xtraiter94
  br i1 %epil.iter95.cmp.not, label %bb41, label %bb38.epil, !llvm.loop !9

bb41:                                             ; preds = %bb41.loopexit.unr-lcssa, %bb38.epil, %bb35, %bb34, %entry
  ret void

bb46:                                             ; preds = %bb12.lr.ph
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb48:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb49:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb50:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  tail call void @llvm.trap() #19
  unreachable

bb51:                                             ; preds = %bb29
  tail call void @llvm.trap() #19
  unreachable

bb52:                                             ; preds = %bb30
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @attention_paged_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, float %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, i32 %v15, ptr writeonly captures(none) %v16, i64 %v17) #3 {
entry:
  %v40 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v41 = zext nneg i32 %v40 to i64
  %v42 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v43.not = icmp ult i32 %v42, %v6
  %v45 = icmp samesign ult i32 %v40, 32
  %or.cond = and i1 %v45, %v43.not
  %v47 = icmp ult i32 %v8, 129
  %or.cond1 = select i1 %or.cond, i1 %v47, i1 false
  br i1 %or.cond1, label %bb6, label %bb51

bb6:                                              ; preds = %entry
  %v49 = zext nneg i32 %v8 to i64
  %v59 = zext i32 %v15 to i64
  %v61 = zext i32 %v14 to i64
  %v62 = shl nuw nsw i64 %v59, 1
  %v64 = zext nneg i32 %v42 to i64
  %v65 = mul nuw nsw i64 %v49, %v64
  %v73 = zext i32 %v9 to i64
  %v74.not167.not = icmp eq i32 %v9, 0
  br i1 %v74.not167.not, label %bb39, label %bb12.lr.ph

bb12.lr.ph:                                       ; preds = %bb6
  %v12.i = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v7, i32 1)
  %v53 = udiv i32 %v6, %v12.i
  %v12.i9 = tail call range(i32 1, 0) i32 @llvm.umax.i32(i32 %v53, i32 1)
  %v57 = udiv i32 %v42, %v12.i9
  %v58 = zext nneg i32 %v57 to i64
  %factor.op.mul = mul nuw nsw i64 %v49, %v58
  %v60 = zext i32 %v13 to i64
  %v63 = mul i64 %v62, %v60
  %v76.not = icmp eq i32 %v13, 0
  %v80 = zext i32 %v11 to i64
  %v81 = zext i32 %v12 to i64
  %v82 = mul nuw i64 %v81, %v80
  %0 = getelementptr i32, ptr %v4, i64 %v82
  %v93.reass = shl nuw nsw i64 %factor.op.mul, 1
  %v98.not162 = icmp samesign ult i32 %v40, %v8
  %1 = getelementptr float, ptr %v0, i64 %v65
  %v284 = add i64 %v93.reass, %v63
  %v145 = shl nuw nsw i64 %v41, 1
  %invariant.gep = getelementptr i8, ptr %v2, i64 %v145
  %v168 = or disjoint i64 %v41, 32
  %v169.not = icmp samesign ult i64 %v168, %v49
  %v171 = shl nuw nsw i64 %v168, 1
  %invariant.gep180 = getelementptr i8, ptr %v2, i64 %v171
  %v194 = or disjoint i64 %v41, 64
  %v195.not = icmp samesign ult i64 %v194, %v49
  %v197 = shl nuw nsw i64 %v194, 1
  %invariant.gep182 = getelementptr i8, ptr %v2, i64 %v197
  %v220 = or disjoint i64 %v41, 96
  %v221.not = icmp samesign ult i64 %v220, %v49
  %v223 = shl nuw nsw i64 %v220, 1
  %invariant.gep184 = getelementptr i8, ptr %v2, i64 %v223
  br i1 %v76.not, label %bb58, label %bb12

bb12:                                             ; preds = %bb12.lr.ph, %bb38
  %v72174 = phi i64 [ %v246, %bb38 ], [ 0, %bb12.lr.ph ]
  %v71173 = phi float [ %v283, %bb38 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v70172 = phi float [ %v141, %bb38 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v69171 = phi float [ %v245, %bb38 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v68170 = phi float [ %v219, %bb38 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v67169 = phi float [ %v193, %bb38 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v66168 = phi float [ %v167, %bb38 ], [ 0.000000e+00, %bb12.lr.ph ]
  %v78.lhs.trunc = trunc i64 %v72174 to i32
  %v78.lhs.trunc.frozen = freeze i32 %v78.lhs.trunc
  %v13.frozen = freeze i32 %v13
  %v78155 = udiv i32 %v78.lhs.trunc.frozen, %v13.frozen
  %v78.zext = zext i32 %v78155 to i64
  %2 = mul i32 %v78155, %v13.frozen
  %v79156.decomposed = sub i32 %v78.lhs.trunc.frozen, %2
  %v79.zext = zext i32 %v79156.decomposed to i64
  %v87 = getelementptr i32, ptr %0, i64 %v78.zext
  %v88 = load i32, ptr %v87, align 4
  %v89 = zext i32 %v88 to i64
  %v90 = mul nuw i64 %v89, %v61
  %v91 = mul i64 %v62, %v79.zext
  br i1 %v98.not162, label %bb15.lr.ph, label %bb18.preheader

bb15.lr.ph:                                       ; preds = %bb12
  %3 = getelementptr i8, ptr %v2, i64 %v90
  %4 = getelementptr i8, ptr %3, i64 %v91
  %5 = getelementptr i8, ptr %4, i64 %v93.reass
  br label %bb15

bb18.preheader:                                   ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb12
  %v96.lcssa = phi float [ 0.000000e+00, %bb12 ], [ %v128, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %v133 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v96.lcssa, i32 16, i32 31) #19
  %v278 = fadd contract float %v96.lcssa, %v133
  %v133.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v278, i32 8, i32 31) #19
  %v278.1 = fadd contract float %v278, %v133.1
  %v133.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v278.1, i32 4, i32 31) #19
  %v278.2 = fadd contract float %v278.1, %v133.2
  %v133.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v278.2, i32 2, i32 31) #19
  %v278.3 = fadd contract float %v278.2, %v133.3
  %v133.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v278.3, i32 1, i32 31) #19
  %v278.4 = fadd contract float %v278.3, %v133.4
  %v134 = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %v278.4, i32 0, i32 31) #19
  %v280 = fmul contract float %v10, %v134
  %v281 = icmp eq i64 %v72174, 0
  br i1 %v281, label %bb26, label %bb22

bb15:                                             ; preds = %bb15.lr.ph, %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v97164 = phi i64 [ %v41, %bb15.lr.ph ], [ %v129, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %v96163 = phi float [ 0.000000e+00, %bb15.lr.ph ], [ %v128, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %v100 = shl nuw nsw i64 %v97164, 1
  %v105 = getelementptr i8, ptr %5, i64 %v100
  %v106 = load i8, ptr %v105, align 1
  %v107 = zext i8 %v106 to i16
  %v113 = getelementptr i8, ptr %v105, i64 1
  %v114 = load i8, ptr %v113, align 1
  %v115 = zext i8 %v114 to i16
  %v118 = shl nuw i16 %v115, 8
  %v124 = getelementptr float, ptr %1, i64 %v97164
  %v125 = load float, ptr %v124, align 4
  %v4.i = lshr i16 %v115, 7
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v115, 2
  %v10.i = and i16 %v9.i, 31
  %v118.masked = and i16 %v118, 768
  %v12.i10 = or disjoint i16 %v118.masked, %v107
  %v13.i = zext nneg i16 %v12.i10 to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb15
  %v15.i = icmp eq i16 %v12.i10, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %6 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %6
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb15
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v41.i, %v38.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb15
  %v44.i = shl nuw i32 %v6.i, 31
  %7 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %7 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v127 = fmul contract float %v125, %v55.i
  %v128 = fadd contract float %v96163, %v127
  %v129 = add nuw nsw i64 %v97164, 32
  %v98.not = icmp samesign ult i64 %v129, %v49
  br i1 %v98.not, label %bb15, label %bb18.preheader

bb22:                                             ; preds = %bb18.preheader
  %v135 = fcmp ule float %v280, %v70172
  br i1 %v135, label %bb26, label %bb23

bb23:                                             ; preds = %bb22
  %v137 = fsub contract float %v70172, %v280
  %8 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %8, 0
  %9 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v137, float 0x3F777313A0000000, float 5.000000e-01) #20
  %10 = tail call float @llvm.fma.f32(float %v137, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %10, float %9
  %11 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %12 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %12, float %11
  %13 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %14 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %14, float %13
  %15 = fadd float %.04.i, 0xC168000FE0000000
  %16 = fneg float %15
  %17 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v137, float 0x3FF7154760000000, float %16) #20
  %18 = tail call float @llvm.fma.f32(float %v137, float 0x3FF7154760000000, float %16)
  %.0.i = select i1 %.not.i, float %18, float %17
  %19 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v137, float 0x3E54AE0C00000000, float %.0.i) #20
  %20 = tail call float @llvm.fma.f32(float %v137, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %20, float %19
  %21 = bitcast float %.04.i to i32
  %22 = shl i32 %21, 23
  %23 = bitcast i32 %22 to float
  %24 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %25 = fmul float %24, %23
  br label %bb26

bb26:                                             ; preds = %bb23, %bb22, %bb18.preheader
  %v141 = phi float [ %v280, %bb18.preheader ], [ %v280, %bb23 ], [ %v70172, %bb22 ]
  %v142 = phi float [ 0.000000e+00, %bb18.preheader ], [ %25, %bb23 ], [ 1.000000e+00, %bb22 ]
  %v143 = fsub contract float %v280, %v141
  %26 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i3 = icmp eq i32 %26, 0
  %27 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v143, float 0x3F777313A0000000, float 5.000000e-01) #20
  %28 = tail call float @llvm.fma.f32(float %v143, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i4 = select i1 %.not.i3, float %28, float %27
  %29 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i4) #20
  %30 = tail call float @llvm.nvvm.saturate.f(float %.02.i4) #20
  %.03.i5 = select i1 %.not.i3, float %30, float %29
  %31 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i5, float 2.520000e+02, float 0x4168000020000000) #20
  %32 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i5, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i6 = select i1 %.not.i3, float %32, float %31
  %33 = fadd float %.04.i6, 0xC168000FE0000000
  %34 = fneg float %33
  %35 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v143, float 0x3FF7154760000000, float %34) #20
  %36 = tail call float @llvm.fma.f32(float %v143, float 0x3FF7154760000000, float %34)
  %.0.i7 = select i1 %.not.i3, float %36, float %35
  %37 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v143, float 0x3E54AE0C00000000, float %.0.i7) #20
  %38 = tail call float @llvm.fma.f32(float %v143, float 0x3E54AE0C00000000, float %.0.i7)
  %.01.i8 = select i1 %.not.i3, float %38, float %37
  %39 = bitcast float %.04.i6 to i32
  %40 = shl i32 %39, 23
  %41 = bitcast i32 %40 to float
  %42 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i8)
  %43 = fmul float %42, %41
  %v282 = fmul contract float %v71173, %v142
  %v283 = fadd contract float %v282, %43
  %v285 = add i64 %v284, %v91
  %v286 = add i64 %v285, %v90
  br i1 %v98.not162, label %bb27, label %bb29

bb27:                                             ; preds = %bb26
  %gep = getelementptr i8, ptr %invariant.gep, i64 %v286
  %v151 = load i8, ptr %gep, align 1
  %v152 = zext i8 %v151 to i16
  %v156 = getelementptr i8, ptr %gep, i64 1
  %v157 = load i8, ptr %v156, align 1
  %v158 = zext i8 %v157 to i16
  %v161 = shl nuw i16 %v158, 8
  %v163 = fmul contract float %v66168, %v142
  %v4.i11 = lshr i16 %v158, 7
  %v6.i12 = zext nneg i16 %v4.i11 to i32
  %v9.i13 = lshr i16 %v158, 2
  %v10.i14 = and i16 %v9.i13, 31
  %v161.masked = and i16 %v161, 768
  %v12.i15 = or disjoint i16 %v161.masked, %v152
  %v13.i16 = zext nneg i16 %v12.i15 to i32
  switch i16 %v10.i14, label %bb10.i39 [
    i16 0, label %bb1.i24
    i16 31, label %bb9.i17
  ]

bb1.i24:                                          ; preds = %bb27
  %v15.i25 = icmp eq i16 %v12.i15, 0
  br i1 %v15.i25, label %bb2.i37, label %bb6.i26

bb2.i37:                                          ; preds = %bb1.i24
  %v17.i38 = shl nuw i32 %v6.i12, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit46

bb6.i26:                                          ; preds = %bb1.i24
  %v13.masked.numleadingzeros.i27 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i16, i1 true)
  %v13.masked.leadingonepos.i28 = xor i32 %v13.masked.numleadingzeros.i27, 31
  %bb5.tripcount.i29 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i28
  %v23.i30 = shl nuw nsw i32 %v13.i16, %bb5.tripcount.i29
  %v27.i31 = shl nuw i32 %v6.i12, 31
  %44 = shl nuw nsw i32 %v13.masked.numleadingzeros.i27, 23
  %reass.sub186 = sub i32 %v27.i31, %44
  %v31.i33 = add i32 %reass.sub186, 1124073472
  %v25.i34 = shl i32 %v23.i30, 13
  %v33.i35 = and i32 %v25.i34, 8380416
  %v34.i36 = or disjoint i32 %v33.i35, %v31.i33
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit46

bb9.i17:                                          ; preds = %bb27
  %v38.i18 = shl nuw i32 %v6.i12, 31
  %v41.i19 = shl nuw nsw i32 %v13.i16, 13
  %v39.i20 = or disjoint i32 %v41.i19, %v38.i18
  %v42.i21 = or disjoint i32 %v39.i20, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit46

bb10.i39:                                         ; preds = %bb27
  %v44.i40 = shl nuw i32 %v6.i12, 31
  %45 = add nuw nsw i16 %v10.i14, 112
  %v46.i41 = zext nneg i16 %45 to i32
  %v48.i42 = shl nuw nsw i32 %v46.i41, 23
  %v49.i43 = or disjoint i32 %v48.i42, %v44.i40
  %v51.i44 = shl nuw nsw i32 %v13.i16, 13
  %v52.i45 = or disjoint i32 %v49.i43, %v51.i44
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit46

cuda_kernels__oxide_kernels__f16_to_f32.exit46:   ; preds = %bb2.i37, %bb6.i26, %bb9.i17, %bb10.i39
  %v54.i22 = phi i32 [ %v34.i36, %bb6.i26 ], [ %v17.i38, %bb2.i37 ], [ %v42.i21, %bb9.i17 ], [ %v52.i45, %bb10.i39 ]
  %v55.i23 = bitcast i32 %v54.i22 to float
  %v165 = fmul contract float %43, %v55.i23
  %v166 = fadd contract float %v163, %v165
  br label %bb29

bb29:                                             ; preds = %bb26, %cuda_kernels__oxide_kernels__f16_to_f32.exit46
  %v167 = phi float [ %v166, %cuda_kernels__oxide_kernels__f16_to_f32.exit46 ], [ %v66168, %bb26 ]
  br i1 %v169.not, label %bb30, label %bb32

bb30:                                             ; preds = %bb29
  %gep181 = getelementptr i8, ptr %invariant.gep180, i64 %v286
  %v177 = load i8, ptr %gep181, align 1
  %v178 = zext i8 %v177 to i16
  %v182 = getelementptr i8, ptr %gep181, i64 1
  %v183 = load i8, ptr %v182, align 1
  %v184 = zext i8 %v183 to i16
  %v187 = shl nuw i16 %v184, 8
  %v189 = fmul contract float %v67169, %v142
  %v4.i47 = lshr i16 %v184, 7
  %v6.i48 = zext nneg i16 %v4.i47 to i32
  %v9.i49 = lshr i16 %v184, 2
  %v10.i50 = and i16 %v9.i49, 31
  %v187.masked = and i16 %v187, 768
  %v12.i51 = or disjoint i16 %v187.masked, %v178
  %v13.i52 = zext nneg i16 %v12.i51 to i32
  switch i16 %v10.i50, label %bb10.i75 [
    i16 0, label %bb1.i60
    i16 31, label %bb9.i53
  ]

bb1.i60:                                          ; preds = %bb30
  %v15.i61 = icmp eq i16 %v12.i51, 0
  br i1 %v15.i61, label %bb2.i73, label %bb6.i62

bb2.i73:                                          ; preds = %bb1.i60
  %v17.i74 = shl nuw i32 %v6.i48, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit82

bb6.i62:                                          ; preds = %bb1.i60
  %v13.masked.numleadingzeros.i63 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i52, i1 true)
  %v13.masked.leadingonepos.i64 = xor i32 %v13.masked.numleadingzeros.i63, 31
  %bb5.tripcount.i65 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i64
  %v23.i66 = shl nuw nsw i32 %v13.i52, %bb5.tripcount.i65
  %v27.i67 = shl nuw i32 %v6.i48, 31
  %46 = shl nuw nsw i32 %v13.masked.numleadingzeros.i63, 23
  %reass.sub187 = sub i32 %v27.i67, %46
  %v31.i69 = add i32 %reass.sub187, 1124073472
  %v25.i70 = shl i32 %v23.i66, 13
  %v33.i71 = and i32 %v25.i70, 8380416
  %v34.i72 = or disjoint i32 %v33.i71, %v31.i69
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit82

bb9.i53:                                          ; preds = %bb30
  %v38.i54 = shl nuw i32 %v6.i48, 31
  %v41.i55 = shl nuw nsw i32 %v13.i52, 13
  %v39.i56 = or disjoint i32 %v41.i55, %v38.i54
  %v42.i57 = or disjoint i32 %v39.i56, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit82

bb10.i75:                                         ; preds = %bb30
  %v44.i76 = shl nuw i32 %v6.i48, 31
  %47 = add nuw nsw i16 %v10.i50, 112
  %v46.i77 = zext nneg i16 %47 to i32
  %v48.i78 = shl nuw nsw i32 %v46.i77, 23
  %v49.i79 = or disjoint i32 %v48.i78, %v44.i76
  %v51.i80 = shl nuw nsw i32 %v13.i52, 13
  %v52.i81 = or disjoint i32 %v49.i79, %v51.i80
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit82

cuda_kernels__oxide_kernels__f16_to_f32.exit82:   ; preds = %bb2.i73, %bb6.i62, %bb9.i53, %bb10.i75
  %v54.i58 = phi i32 [ %v34.i72, %bb6.i62 ], [ %v17.i74, %bb2.i73 ], [ %v42.i57, %bb9.i53 ], [ %v52.i81, %bb10.i75 ]
  %v55.i59 = bitcast i32 %v54.i58 to float
  %v191 = fmul contract float %43, %v55.i59
  %v192 = fadd contract float %v189, %v191
  br label %bb32

bb32:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit82, %bb29
  %v193 = phi float [ %v67169, %bb29 ], [ %v192, %cuda_kernels__oxide_kernels__f16_to_f32.exit82 ]
  br i1 %v195.not, label %bb33, label %bb35

bb33:                                             ; preds = %bb32
  %gep183 = getelementptr i8, ptr %invariant.gep182, i64 %v286
  %v203 = load i8, ptr %gep183, align 1
  %v204 = zext i8 %v203 to i16
  %v208 = getelementptr i8, ptr %gep183, i64 1
  %v209 = load i8, ptr %v208, align 1
  %v210 = zext i8 %v209 to i16
  %v213 = shl nuw i16 %v210, 8
  %v215 = fmul contract float %v68170, %v142
  %v4.i83 = lshr i16 %v210, 7
  %v6.i84 = zext nneg i16 %v4.i83 to i32
  %v9.i85 = lshr i16 %v210, 2
  %v10.i86 = and i16 %v9.i85, 31
  %v213.masked = and i16 %v213, 768
  %v12.i87 = or disjoint i16 %v213.masked, %v204
  %v13.i88 = zext nneg i16 %v12.i87 to i32
  switch i16 %v10.i86, label %bb10.i111 [
    i16 0, label %bb1.i96
    i16 31, label %bb9.i89
  ]

bb1.i96:                                          ; preds = %bb33
  %v15.i97 = icmp eq i16 %v12.i87, 0
  br i1 %v15.i97, label %bb2.i109, label %bb6.i98

bb2.i109:                                         ; preds = %bb1.i96
  %v17.i110 = shl nuw i32 %v6.i84, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit118

bb6.i98:                                          ; preds = %bb1.i96
  %v13.masked.numleadingzeros.i99 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i88, i1 true)
  %v13.masked.leadingonepos.i100 = xor i32 %v13.masked.numleadingzeros.i99, 31
  %bb5.tripcount.i101 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i100
  %v23.i102 = shl nuw nsw i32 %v13.i88, %bb5.tripcount.i101
  %v27.i103 = shl nuw i32 %v6.i84, 31
  %48 = shl nuw nsw i32 %v13.masked.numleadingzeros.i99, 23
  %reass.sub188 = sub i32 %v27.i103, %48
  %v31.i105 = add i32 %reass.sub188, 1124073472
  %v25.i106 = shl i32 %v23.i102, 13
  %v33.i107 = and i32 %v25.i106, 8380416
  %v34.i108 = or disjoint i32 %v33.i107, %v31.i105
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit118

bb9.i89:                                          ; preds = %bb33
  %v38.i90 = shl nuw i32 %v6.i84, 31
  %v41.i91 = shl nuw nsw i32 %v13.i88, 13
  %v39.i92 = or disjoint i32 %v41.i91, %v38.i90
  %v42.i93 = or disjoint i32 %v39.i92, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit118

bb10.i111:                                        ; preds = %bb33
  %v44.i112 = shl nuw i32 %v6.i84, 31
  %49 = add nuw nsw i16 %v10.i86, 112
  %v46.i113 = zext nneg i16 %49 to i32
  %v48.i114 = shl nuw nsw i32 %v46.i113, 23
  %v49.i115 = or disjoint i32 %v48.i114, %v44.i112
  %v51.i116 = shl nuw nsw i32 %v13.i88, 13
  %v52.i117 = or disjoint i32 %v49.i115, %v51.i116
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit118

cuda_kernels__oxide_kernels__f16_to_f32.exit118:  ; preds = %bb2.i109, %bb6.i98, %bb9.i89, %bb10.i111
  %v54.i94 = phi i32 [ %v34.i108, %bb6.i98 ], [ %v17.i110, %bb2.i109 ], [ %v42.i93, %bb9.i89 ], [ %v52.i117, %bb10.i111 ]
  %v55.i95 = bitcast i32 %v54.i94 to float
  %v217 = fmul contract float %43, %v55.i95
  %v218 = fadd contract float %v215, %v217
  br label %bb35

bb35:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit118, %bb32
  %v219 = phi float [ %v68170, %bb32 ], [ %v218, %cuda_kernels__oxide_kernels__f16_to_f32.exit118 ]
  br i1 %v221.not, label %bb36, label %bb38

bb36:                                             ; preds = %bb35
  %gep185 = getelementptr i8, ptr %invariant.gep184, i64 %v286
  %v229 = load i8, ptr %gep185, align 1
  %v230 = zext i8 %v229 to i16
  %v234 = getelementptr i8, ptr %gep185, i64 1
  %v235 = load i8, ptr %v234, align 1
  %v236 = zext i8 %v235 to i16
  %v239 = shl nuw i16 %v236, 8
  %v241 = fmul contract float %v69171, %v142
  %v4.i119 = lshr i16 %v236, 7
  %v6.i120 = zext nneg i16 %v4.i119 to i32
  %v9.i121 = lshr i16 %v236, 2
  %v10.i122 = and i16 %v9.i121, 31
  %v239.masked = and i16 %v239, 768
  %v12.i123 = or disjoint i16 %v239.masked, %v230
  %v13.i124 = zext nneg i16 %v12.i123 to i32
  switch i16 %v10.i122, label %bb10.i147 [
    i16 0, label %bb1.i132
    i16 31, label %bb9.i125
  ]

bb1.i132:                                         ; preds = %bb36
  %v15.i133 = icmp eq i16 %v12.i123, 0
  br i1 %v15.i133, label %bb2.i145, label %bb6.i134

bb2.i145:                                         ; preds = %bb1.i132
  %v17.i146 = shl nuw i32 %v6.i120, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit154

bb6.i134:                                         ; preds = %bb1.i132
  %v13.masked.numleadingzeros.i135 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i124, i1 true)
  %v13.masked.leadingonepos.i136 = xor i32 %v13.masked.numleadingzeros.i135, 31
  %bb5.tripcount.i137 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i136
  %v23.i138 = shl nuw nsw i32 %v13.i124, %bb5.tripcount.i137
  %v27.i139 = shl nuw i32 %v6.i120, 31
  %50 = shl nuw nsw i32 %v13.masked.numleadingzeros.i135, 23
  %reass.sub189 = sub i32 %v27.i139, %50
  %v31.i141 = add i32 %reass.sub189, 1124073472
  %v25.i142 = shl i32 %v23.i138, 13
  %v33.i143 = and i32 %v25.i142, 8380416
  %v34.i144 = or disjoint i32 %v33.i143, %v31.i141
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit154

bb9.i125:                                         ; preds = %bb36
  %v38.i126 = shl nuw i32 %v6.i120, 31
  %v41.i127 = shl nuw nsw i32 %v13.i124, 13
  %v39.i128 = or disjoint i32 %v41.i127, %v38.i126
  %v42.i129 = or disjoint i32 %v39.i128, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit154

bb10.i147:                                        ; preds = %bb36
  %v44.i148 = shl nuw i32 %v6.i120, 31
  %51 = add nuw nsw i16 %v10.i122, 112
  %v46.i149 = zext nneg i16 %51 to i32
  %v48.i150 = shl nuw nsw i32 %v46.i149, 23
  %v49.i151 = or disjoint i32 %v48.i150, %v44.i148
  %v51.i152 = shl nuw nsw i32 %v13.i124, 13
  %v52.i153 = or disjoint i32 %v49.i151, %v51.i152
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit154

cuda_kernels__oxide_kernels__f16_to_f32.exit154:  ; preds = %bb2.i145, %bb6.i134, %bb9.i125, %bb10.i147
  %v54.i130 = phi i32 [ %v34.i144, %bb6.i134 ], [ %v17.i146, %bb2.i145 ], [ %v42.i129, %bb9.i125 ], [ %v52.i153, %bb10.i147 ]
  %v55.i131 = bitcast i32 %v54.i130 to float
  %v243 = fmul contract float %43, %v55.i131
  %v244 = fadd contract float %v241, %v243
  br label %bb38

bb38:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit154, %bb35
  %v245 = phi float [ %v69171, %bb35 ], [ %v244, %cuda_kernels__oxide_kernels__f16_to_f32.exit154 ]
  %v246 = add nuw nsw i64 %v72174, 1
  %exitcond.not = icmp eq i64 %v246, %v73
  br i1 %exitcond.not, label %bb39, label %bb12

bb39:                                             ; preds = %bb38, %bb6
  %v66.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v167, %bb38 ]
  %v67.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v193, %bb38 ]
  %v68.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v219, %bb38 ]
  %v69.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v245, %bb38 ]
  %v71.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v283, %bb38 ]
  %v247 = fdiv contract float 1.000000e+00, %v71.lcssa
  %v248.not = icmp ult i32 %v40, %v8
  br i1 %v248.not, label %bb40, label %bb41

bb40:                                             ; preds = %bb39
  %v253 = fmul contract float %v66.lcssa, %v247
  %52 = getelementptr inbounds nuw float, ptr %v16, i64 %v65
  %v252 = getelementptr inbounds nuw float, ptr %52, i64 %v41
  store float %v253, ptr %v252, align 4
  br label %bb41

bb41:                                             ; preds = %bb40, %bb39
  %v254 = or disjoint i64 %v41, 32
  %v255.not = icmp samesign ult i64 %v254, %v49
  br i1 %v255.not, label %bb42, label %bb44

bb42:                                             ; preds = %bb41
  %v261 = fmul contract float %v67.lcssa, %v247
  %53 = getelementptr inbounds nuw float, ptr %v16, i64 %v65
  %54 = getelementptr inbounds nuw float, ptr %53, i64 %v41
  %v260 = getelementptr inbounds nuw i8, ptr %54, i64 128
  store float %v261, ptr %v260, align 4
  br label %bb44

bb44:                                             ; preds = %bb41, %bb42
  %v262 = or disjoint i64 %v41, 64
  %v263.not = icmp samesign ult i64 %v262, %v49
  br i1 %v263.not, label %bb45, label %bb47

bb45:                                             ; preds = %bb44
  %v269 = fmul contract float %v68.lcssa, %v247
  %55 = getelementptr inbounds nuw float, ptr %v16, i64 %v65
  %56 = getelementptr inbounds nuw float, ptr %55, i64 %v41
  %v268 = getelementptr inbounds nuw i8, ptr %56, i64 256
  store float %v269, ptr %v268, align 4
  br label %bb47

bb47:                                             ; preds = %bb44, %bb45
  %v270 = or disjoint i64 %v41, 96
  %v271.not = icmp samesign ult i64 %v270, %v49
  br i1 %v271.not, label %bb48, label %bb51

bb48:                                             ; preds = %bb47
  %v277 = fmul contract float %v69.lcssa, %v247
  %57 = getelementptr inbounds nuw float, ptr %v16, i64 %v65
  %58 = getelementptr inbounds nuw float, ptr %57, i64 %v41
  %v276 = getelementptr inbounds nuw i8, ptr %58, i64 384
  store float %v277, ptr %v276, align 4
  br label %bb51

bb51:                                             ; preds = %bb48, %bb47, %entry
  ret void

bb58:                                             ; preds = %bb12.lr.ph
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @embedding_f32(ptr readonly captures(none) %v0, i64 %v1, i32 %v2, i32 %v3, ptr writeonly captures(address_is_null) %v4, i64 %v5) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v17 = zext i32 %v3 to i64
  %v18.not = icmp ult i64 %v22.i, %v17
  br i1 %v18.not, label %bb3, label %bb8

bb3:                                              ; preds = %entry
  %v20 = zext i32 %v2 to i64
  %v21 = mul nuw i64 %v17, %v20
  %v22 = add nuw i64 %v21, %v22.i
  %v31.not = icmp ult i64 %v22.i, %v5
  %v34 = getelementptr inbounds nuw float, ptr %v4, i64 %v22.i
  %v451 = icmp ne ptr %v4, null
  %v45 = select i1 %v31.not, i1 %v451, i1 false
  br i1 %v45, label %bb4, label %bb8

bb4:                                              ; preds = %bb3
  %v26 = icmp ult i64 %v22, %v1
  br i1 %v26, label %bb5, label %bb16

bb5:                                              ; preds = %bb4
  %v28 = getelementptr inbounds float, ptr %v0, i64 %v22
  %v29 = load float, ptr %v28, align 4
  store float %v29, ptr %v34, align 4
  br label %bb8

bb8:                                              ; preds = %bb3, %bb5, %entry
  ret void

bb16:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @embedding_q4k_row(ptr readonly captures(none) %v0, i64 %v1, i32 %v2, i32 %v3, i32 %v4, ptr writeonly captures(address_is_null) %v5, i64 %v6) #0 {
entry:
  %v19 = alloca [8 x i8], align 4
  %v20 = alloca [8 x i8], align 4
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i10 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i11 = icmp eq i32 %v4.i9, 1
  %v7.i12 = icmp eq i32 %v6.i10, 1
  %v8.not.not.i = and i1 %v5.i11, %v7.i12
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i13 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i13
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v23 = zext i32 %v3 to i64
  %v24.not = icmp ult i64 %v22.i, %v23
  br i1 %v24.not, label %bb3, label %bb32

bb3:                                              ; preds = %entry
  %v26 = mul i32 %v4, 144
  %v27 = zext i32 %v26 to i64
  %v28 = zext i32 %v2 to i64
  %v29 = mul nuw i64 %v27, %v28
  %v301 = lshr i64 %v22.i, 8
  %v32 = mul nuw nsw i64 %v301, 144
  %v33 = add nuw i64 %v29, %v32
  %v35 = icmp ult i64 %v33, %v1
  br i1 %v35, label %bb4, label %bb40

bb4:                                              ; preds = %bb3
  %v39 = or disjoint i64 %v33, 1
  %v40 = icmp ult i64 %v39, %v1
  br i1 %v40, label %bb5, label %bb41

bb5:                                              ; preds = %bb4
  %v37 = getelementptr inbounds i8, ptr %v0, i64 %v33
  %v38 = load i8, ptr %v37, align 1
  %v42 = getelementptr inbounds i8, ptr %v0, i64 %v39
  %v43 = load i8, ptr %v42, align 1
  %v47 = alloca [2 x i8], align 2
  store i8 %v38, ptr %v47, align 2
  %v47.repack2 = getelementptr inbounds nuw i8, ptr %v47, i64 1
  store i8 %v43, ptr %v47.repack2, align 1
  %v48 = load i16, ptr %v47, align 2
  %v49 = or disjoint i64 %v33, 2
  %v50 = icmp ult i64 %v49, %v1
  br i1 %v50, label %bb6, label %bb42

bb6:                                              ; preds = %bb5
  %v54 = or disjoint i64 %v33, 3
  %v55 = icmp ult i64 %v54, %v1
  br i1 %v55, label %bb7, label %bb43

bb7:                                              ; preds = %bb6
  %v52 = getelementptr inbounds i8, ptr %v0, i64 %v49
  %v53 = load i8, ptr %v52, align 1
  %v57 = getelementptr inbounds i8, ptr %v0, i64 %v54
  %v58 = load i8, ptr %v57, align 1
  %v62 = alloca [2 x i8], align 2
  store i8 %v53, ptr %v62, align 2
  %v62.repack4 = getelementptr inbounds nuw i8, ptr %v62, i64 1
  store i8 %v58, ptr %v62.repack4, align 1
  %v63 = load i16, ptr %v62, align 2
  %v4.i14 = lshr i16 %v48, 15
  %v6.i15 = zext nneg i16 %v4.i14 to i32
  %v9.i = lshr i16 %v48, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v48, 1023
  %v13.i16 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb7
  %v15.i17 = icmp eq i16 %v12.i, 0
  br i1 %v15.i17, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i18 = shl nuw i32 %v6.i15, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i16, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i16, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i15, 31
  %0 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %0
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb7
  %v38.i = shl nuw i32 %v6.i15, 31
  %v41.i = shl nuw nsw i32 %v13.i16, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb7
  %v44.i = shl nuw i32 %v6.i15, 31
  %1 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %1 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i16, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i18, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v4.i19 = lshr i16 %v63, 15
  %v6.i20 = zext nneg i16 %v4.i19 to i32
  %v9.i21 = lshr i16 %v63, 10
  %v10.i22 = and i16 %v9.i21, 31
  %v12.i23 = and i16 %v63, 1023
  %v13.i24 = zext nneg i16 %v12.i23 to i32
  switch i16 %v10.i22, label %bb10.i47 [
    i16 0, label %bb1.i32
    i16 31, label %bb9.i25
  ]

bb1.i32:                                          ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v15.i33 = icmp eq i16 %v12.i23, 0
  br i1 %v15.i33, label %bb2.i45, label %bb6.i34

bb2.i45:                                          ; preds = %bb1.i32
  %v17.i46 = shl nuw i32 %v6.i20, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

bb6.i34:                                          ; preds = %bb1.i32
  %v13.masked.numleadingzeros.i35 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i24, i1 true)
  %v13.masked.leadingonepos.i36 = xor i32 %v13.masked.numleadingzeros.i35, 31
  %bb5.tripcount.i37 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i36
  %v23.i38 = shl nuw nsw i32 %v13.i24, %bb5.tripcount.i37
  %v27.i39 = shl nuw i32 %v6.i20, 31
  %2 = shl nuw nsw i32 %v13.masked.numleadingzeros.i35, 23
  %reass.sub55 = sub i32 %v27.i39, %2
  %v31.i41 = add i32 %reass.sub55, 1124073472
  %v25.i42 = shl i32 %v23.i38, 13
  %v33.i43 = and i32 %v25.i42, 8380416
  %v34.i44 = or disjoint i32 %v33.i43, %v31.i41
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

bb9.i25:                                          ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v38.i26 = shl nuw i32 %v6.i20, 31
  %v41.i27 = shl nuw nsw i32 %v13.i24, 13
  %v39.i28 = or disjoint i32 %v38.i26, %v41.i27
  %v42.i29 = or disjoint i32 %v39.i28, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

bb10.i47:                                         ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v44.i48 = shl nuw i32 %v6.i20, 31
  %3 = add nuw nsw i16 %v10.i22, 112
  %v46.i49 = zext nneg i16 %3 to i32
  %v48.i50 = shl nuw nsw i32 %v46.i49, 23
  %v49.i51 = or disjoint i32 %v48.i50, %v44.i48
  %v51.i52 = shl nuw nsw i32 %v13.i24, 13
  %v52.i53 = or disjoint i32 %v49.i51, %v51.i52
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit54

cuda_kernels__oxide_kernels__f16_to_f32.exit54:   ; preds = %bb2.i45, %bb6.i34, %bb9.i25, %bb10.i47
  %v54.i30 = phi i32 [ %v34.i44, %bb6.i34 ], [ %v17.i46, %bb2.i45 ], [ %v42.i29, %bb9.i25 ], [ %v52.i53, %bb10.i47 ]
  %v55.i31 = bitcast i32 %v54.i30 to float
  %v66 = or disjoint i64 %v33, 4
  %v67 = icmp ult i64 %v66, %v1
  br i1 %v67, label %bb10, label %bb44

bb10:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit54
  %v69 = getelementptr inbounds i8, ptr %v0, i64 %v66
  %v70 = load i8, ptr %v69, align 1
  %v71 = or disjoint i64 %v33, 5
  %v72 = icmp ult i64 %v71, %v1
  br i1 %v72, label %bb11, label %bb45

bb11:                                             ; preds = %bb10
  %v74 = getelementptr inbounds i8, ptr %v0, i64 %v71
  %v75 = load i8, ptr %v74, align 1
  %v76 = or disjoint i64 %v33, 6
  %v77 = icmp ult i64 %v76, %v1
  br i1 %v77, label %bb12, label %bb46

bb12:                                             ; preds = %bb11
  %v79 = getelementptr inbounds i8, ptr %v0, i64 %v76
  %v80 = load i8, ptr %v79, align 1
  %v81 = or disjoint i64 %v33, 7
  %v82 = icmp ult i64 %v81, %v1
  br i1 %v82, label %bb13, label %bb47

bb13:                                             ; preds = %bb12
  %v84 = getelementptr inbounds i8, ptr %v0, i64 %v81
  %v85 = load i8, ptr %v84, align 1
  %v86 = or disjoint i64 %v33, 8
  %v87 = icmp ult i64 %v86, %v1
  br i1 %v87, label %bb14, label %bb48

bb14:                                             ; preds = %bb13
  %v89 = getelementptr inbounds i8, ptr %v0, i64 %v86
  %v90 = load i8, ptr %v89, align 1
  %v91 = or disjoint i64 %v33, 9
  %v92 = icmp ult i64 %v91, %v1
  br i1 %v92, label %bb15, label %bb49

bb15:                                             ; preds = %bb14
  %v94 = getelementptr inbounds i8, ptr %v0, i64 %v91
  %v95 = load i8, ptr %v94, align 1
  %v96 = or disjoint i64 %v33, 10
  %v97 = icmp ult i64 %v96, %v1
  br i1 %v97, label %bb16, label %bb50

bb16:                                             ; preds = %bb15
  %v99 = getelementptr inbounds i8, ptr %v0, i64 %v96
  %v100 = load i8, ptr %v99, align 1
  %v101 = or disjoint i64 %v33, 11
  %v102 = icmp ult i64 %v101, %v1
  br i1 %v102, label %bb17, label %bb51

bb17:                                             ; preds = %bb16
  %v104 = getelementptr inbounds i8, ptr %v0, i64 %v101
  %v105 = load i8, ptr %v104, align 1
  %v106 = or disjoint i64 %v33, 12
  %v107 = icmp ult i64 %v106, %v1
  br i1 %v107, label %bb18, label %bb52

bb18:                                             ; preds = %bb17
  %v109 = getelementptr inbounds i8, ptr %v0, i64 %v106
  %v110 = load i8, ptr %v109, align 1
  %v111 = or disjoint i64 %v33, 13
  %v112 = icmp ult i64 %v111, %v1
  br i1 %v112, label %bb19, label %bb53

bb19:                                             ; preds = %bb18
  %v114 = getelementptr inbounds i8, ptr %v0, i64 %v111
  %v115 = load i8, ptr %v114, align 1
  %v116 = or disjoint i64 %v33, 14
  %v117 = icmp ult i64 %v116, %v1
  br i1 %v117, label %bb20, label %bb54

bb20:                                             ; preds = %bb19
  %v121 = or disjoint i64 %v33, 15
  %v122 = icmp ult i64 %v121, %v1
  br i1 %v122, label %bb21, label %bb55

bb21:                                             ; preds = %bb20
  %v119 = getelementptr inbounds i8, ptr %v0, i64 %v116
  %v120 = load i8, ptr %v119, align 1
  %v124 = getelementptr inbounds i8, ptr %v0, i64 %v121
  %v125 = load i8, ptr %v124, align 1
  %v43.sroa.4.0.insert.ext.i = zext i8 %v85 to i32
  %v43.sroa.4.0.insert.shift.i = shl nuw i32 %v43.sroa.4.0.insert.ext.i, 24
  %v43.sroa.3.0.insert.ext.i = zext i8 %v80 to i32
  %v43.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.3.0.insert.ext.i, 16
  %v43.sroa.2.0.insert.ext.i = zext i8 %v75 to i32
  %v43.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.2.0.insert.ext.i, 8
  %v43.sroa.0.0.insert.ext.i = zext i8 %v70 to i32
  %v43.sroa.3.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.shift.i, %v43.sroa.0.0.insert.ext.i
  %v43.sroa.2.0.insert.insert.i = or disjoint i32 %v43.sroa.3.0.insert.insert.i, %v43.sroa.3.0.insert.shift.i
  %v43.sroa.0.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.insert.i, %v43.sroa.4.0.insert.shift.i
  %v51.sroa.4.0.insert.ext.i = zext i8 %v105 to i32
  %v51.sroa.4.0.insert.shift.i = shl nuw i32 %v51.sroa.4.0.insert.ext.i, 24
  %v51.sroa.3.0.insert.ext.i = zext i8 %v100 to i32
  %v51.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.3.0.insert.ext.i, 16
  %v51.sroa.2.0.insert.ext.i = zext i8 %v95 to i32
  %v51.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.2.0.insert.ext.i, 8
  %v51.sroa.0.0.insert.ext.i = zext i8 %v90 to i32
  %v51.sroa.3.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.shift.i, %v51.sroa.0.0.insert.ext.i
  %v51.sroa.2.0.insert.insert.i = or disjoint i32 %v51.sroa.3.0.insert.insert.i, %v51.sroa.3.0.insert.shift.i
  %v51.sroa.0.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.insert.i, %v51.sroa.4.0.insert.shift.i
  %v59.sroa.4.0.insert.ext.i = zext i8 %v125 to i32
  %v59.sroa.4.0.insert.shift.i = shl nuw i32 %v59.sroa.4.0.insert.ext.i, 24
  %v59.sroa.3.0.insert.ext.i = zext i8 %v120 to i32
  %v59.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.3.0.insert.ext.i, 16
  %v59.sroa.2.0.insert.ext.i = zext i8 %v115 to i32
  %v59.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.2.0.insert.ext.i, 8
  %v59.sroa.0.0.insert.ext.i = zext i8 %v110 to i32
  %v59.sroa.3.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.shift.i, %v59.sroa.0.0.insert.ext.i
  %v59.sroa.2.0.insert.insert.i = or disjoint i32 %v59.sroa.3.0.insert.insert.i, %v59.sroa.3.0.insert.shift.i
  %v59.sroa.0.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.insert.i, %v59.sroa.4.0.insert.shift.i
  %v65.i = lshr i32 %v59.sroa.0.0.insert.insert.i, 4
  %v66.i = and i32 %v65.i, 252645135
  %4 = lshr i32 %v51.sroa.0.0.insert.insert.i, 2
  %v73.i = and i32 %4, 808464432
  %v81.i = and i32 %v59.sroa.0.0.insert.insert.i, 252645135
  %5 = lshr i32 %v43.sroa.0.0.insert.insert.i, 2
  %v88.i = and i32 %5, 808464432
  %v94.i = and i32 %v43.sroa.0.0.insert.insert.i, 1061109567
  %v98.sroa.2.0.extract.shift.i = lshr i32 %v94.i, 8
  %v98.sroa.4.0.extract.shift.i = lshr i32 %v94.i, 24
  %v98.sroa.3.0.extract.shift.i = lshr i32 %v94.i, 16
  %v98.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v98.sroa.4.0.extract.shift.i to i8
  %v98.sroa.3.0.extract.trunc.i = trunc i32 %v98.sroa.3.0.extract.shift.i to i8
  %6 = insertelement <4 x i32> poison, i32 %v94.i, i64 0
  %7 = insertelement <4 x i32> %6, i32 %v98.sroa.2.0.extract.shift.i, i64 1
  %8 = trunc <4 x i32> %7 to <4 x i8>
  %9 = insertelement <4 x i8> %8, i8 %v98.sroa.3.0.extract.trunc.i, i64 2
  %10 = insertelement <4 x i8> %9, i8 %v98.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %10, ptr %v19, align 4
  %v127.fca.4.gep = getelementptr inbounds nuw i8, ptr %v19, i64 4
  %v89.i = or disjoint i32 %v81.i, %v88.i
  %v102.sroa.2.0.extract.shift.i = lshr i32 %v89.i, 8
  %v102.sroa.4.0.extract.shift.i = lshr i32 %v89.i, 24
  %v102.sroa.3.0.extract.shift.i = lshr i32 %v89.i, 16
  %v102.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v102.sroa.4.0.extract.shift.i to i8
  %v102.sroa.3.0.extract.trunc.i = trunc i32 %v102.sroa.3.0.extract.shift.i to i8
  %11 = insertelement <4 x i32> poison, i32 %v89.i, i64 0
  %12 = insertelement <4 x i32> %11, i32 %v102.sroa.2.0.extract.shift.i, i64 1
  %13 = trunc <4 x i32> %12 to <4 x i8>
  %14 = insertelement <4 x i8> %13, i8 %v102.sroa.3.0.extract.trunc.i, i64 2
  %15 = insertelement <4 x i8> %14, i8 %v102.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %15, ptr %v127.fca.4.gep, align 4
  %v78.i = and i32 %v51.sroa.0.0.insert.insert.i, 1061109567
  %v106.sroa.2.0.extract.shift.i = lshr i32 %v78.i, 8
  %v106.sroa.4.0.extract.shift.i = lshr i32 %v78.i, 24
  %v106.sroa.3.0.extract.shift.i = lshr i32 %v78.i, 16
  %v106.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v106.sroa.4.0.extract.shift.i to i8
  %v106.sroa.3.0.extract.trunc.i = trunc i32 %v106.sroa.3.0.extract.shift.i to i8
  %16 = insertelement <4 x i32> poison, i32 %v78.i, i64 0
  %17 = insertelement <4 x i32> %16, i32 %v106.sroa.2.0.extract.shift.i, i64 1
  %18 = trunc <4 x i32> %17 to <4 x i8>
  %19 = insertelement <4 x i8> %18, i8 %v106.sroa.3.0.extract.trunc.i, i64 2
  %20 = insertelement <4 x i8> %19, i8 %v106.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %20, ptr %v20, align 4
  %v128.fca.4.gep = getelementptr inbounds nuw i8, ptr %v20, i64 4
  %v74.i = or disjoint i32 %v66.i, %v73.i
  %v110.sroa.2.0.extract.shift.i = lshr i32 %v74.i, 8
  %v110.sroa.4.0.extract.shift.i = lshr i32 %v74.i, 24
  %v110.sroa.3.0.extract.shift.i = lshr i32 %v74.i, 16
  %v110.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v110.sroa.4.0.extract.shift.i to i8
  %v110.sroa.3.0.extract.trunc.i = trunc i32 %v110.sroa.3.0.extract.shift.i to i8
  %21 = insertelement <4 x i32> poison, i32 %v74.i, i64 0
  %22 = insertelement <4 x i32> %21, i32 %v110.sroa.2.0.extract.shift.i, i64 1
  %23 = trunc <4 x i32> %22 to <4 x i8>
  %24 = insertelement <4 x i8> %23, i8 %v110.sroa.3.0.extract.trunc.i, i64 2
  %25 = insertelement <4 x i8> %24, i8 %v110.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %25, ptr %v128.fca.4.gep, align 4
  %v130 = and i64 %v22.i, 31
  %v31 = lshr i64 %v22.i, 5
  %v1296 = and i64 %v31, 7
  %v132 = getelementptr inbounds nuw i8, ptr %v19, i64 %v1296
  %v133 = load i8, ptr %v132, align 1
  %v134 = uitofp i8 %v133 to float
  %v135 = getelementptr inbounds nuw i8, ptr %v20, i64 %v1296
  %v136 = load i8, ptr %v135, align 1
  %v137 = uitofp i8 %v136 to float
  %v140 = and i64 %v22.i, 32
  %26 = lshr i64 %v22.i, 1
  %v141 = and i64 %26, 96
  %v138 = or disjoint i64 %v141, 16
  %v142 = add nuw i64 %v138, %v33
  %v143.not.not = icmp eq i64 %v140, 0
  %v145 = add nuw i64 %v142, %v130
  %v146 = icmp ult i64 %v145, %v1
  br i1 %v143.not.not, label %bb24, label %bb26

bb24:                                             ; preds = %bb21
  br i1 %v146, label %bb25, label %bb57

bb25:                                             ; preds = %bb24
  %v148 = getelementptr inbounds i8, ptr %v0, i64 %v145
  %v149 = load i8, ptr %v148, align 1
  %v150 = and i8 %v149, 15
  br label %bb28

bb26:                                             ; preds = %bb21
  br i1 %v146, label %bb27, label %bb58

bb27:                                             ; preds = %bb26
  %v155 = getelementptr inbounds i8, ptr %v0, i64 %v145
  %v156 = load i8, ptr %v155, align 1
  %v159 = lshr i8 %v156, 4
  br label %bb28

bb28:                                             ; preds = %bb27, %bb25
  %v161.in = phi i8 [ %v150, %bb25 ], [ %v159, %bb27 ]
  %v169 = icmp ult i64 %v22.i, %v6
  %or.cond.not = select i1 %.v18.i, i1 %v169, i1 false
  %v1838 = icmp ne ptr %v5, null
  %v183 = select i1 %or.cond.not, i1 %v1838, i1 false
  br i1 %v183, label %bb29, label %bb32

bb29:                                             ; preds = %bb28
  %v164 = fmul contract float %v55.i, %v134
  %v161 = uitofp nneg i8 %v161.in to float
  %v165 = fmul contract float %v164, %v161
  %v166 = fmul contract float %v55.i31, %v137
  %v167 = fsub contract float %v165, %v166
  %v172 = getelementptr inbounds nuw float, ptr %v5, i64 %v22.i
  store float %v167, ptr %v172, align 4
  br label %bb32

bb32:                                             ; preds = %bb28, %bb29, %entry
  ret void

bb40:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb44:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit54
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb48:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb49:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb50:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb51:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb52:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb53:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb54:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb55:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb57:                                             ; preds = %bb24
  tail call void @llvm.trap() #19
  unreachable

bb58:                                             ; preds = %bb26
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @embedding_q6k_row(ptr readonly captures(none) %v0, i64 %v1, i32 %v2, i32 %v3, i32 %v4, ptr writeonly captures(address_is_null) %v5, i64 %v6) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i11 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i12 = icmp eq i32 %v4.i10, 1
  %v7.i13 = icmp eq i32 %v6.i11, 1
  %v8.not.not.i = and i1 %v5.i12, %v7.i13
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i14 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i14
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v20 = zext i32 %v3 to i64
  %v21.not = icmp ult i64 %v22.i, %v20
  br i1 %v21.not, label %bb3, label %bb29

bb3:                                              ; preds = %entry
  %v23 = mul i32 %v4, 210
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v2 to i64
  %v26 = mul nuw i64 %v24, %v25
  %v272 = lshr i64 %v22.i, 8
  %v29 = mul nuw nsw i64 %v272, 210
  %v30 = add nuw i64 %v26, %v29
  %v31 = add nuw i64 %v30, 208
  %v33 = icmp ult i64 %v31, %v1
  br i1 %v33, label %bb4, label %bb37

bb4:                                              ; preds = %bb3
  %v37 = add nuw i64 %v30, 209
  %v38 = icmp ult i64 %v37, %v1
  br i1 %v38, label %bb5, label %bb38

bb5:                                              ; preds = %bb4
  %v35 = getelementptr inbounds i8, ptr %v0, i64 %v31
  %v36 = load i8, ptr %v35, align 1
  %v40 = getelementptr inbounds i8, ptr %v0, i64 %v37
  %v41 = load i8, ptr %v40, align 1
  %v45 = alloca [2 x i8], align 2
  store i8 %v36, ptr %v45, align 2
  %v45.repack3 = getelementptr inbounds nuw i8, ptr %v45, i64 1
  store i8 %v41, ptr %v45.repack3, align 1
  %v46 = load i16, ptr %v45, align 2
  %v4.i15 = lshr i16 %v46, 15
  %v6.i16 = zext nneg i16 %v4.i15 to i32
  %v9.i = lshr i16 %v46, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v46, 1023
  %v13.i17 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb5
  %v15.i18 = icmp eq i16 %v12.i, 0
  br i1 %v15.i18, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i19 = shl nuw i32 %v6.i16, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i17, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i17, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i16, 31
  %0 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %0
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb5
  %v38.i = shl nuw i32 %v6.i16, 31
  %v41.i = shl nuw nsw i32 %v13.i17, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb5
  %v44.i = shl nuw i32 %v6.i16, 31
  %1 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %1 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i17, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i19, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v28 = lshr i64 %v22.i, 7
  %v485 = and i64 %v28, 1
  %v50 = shl nuw nsw i64 %v485, 6
  %v51 = add nuw i64 %v30, %v50
  %v53 = shl nuw nsw i64 %v485, 5
  %v56 = shl nuw nsw i64 %v485, 3
  %v58 = and i64 %v22.i, 31
  %v49 = lshr i64 %v22.i, 5
  %v596 = and i64 %v49, 3
  %v607 = lshr i64 %v58, 4
  %switch.idx.cast = trunc nuw nsw i64 %v596 to i8
  %switch.idx.mult = shl nuw nsw i8 %switch.idx.cast, 1
  switch i64 %v596, label %bb16 [
    i64 0, label %bb17
    i64 2, label %bb17
  ]

bb16:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v68 = or disjoint i64 %v58, 32
  br label %bb17

bb17:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb16
  %v69 = phi i64 [ %v68, %bb16 ], [ %v58, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v58, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %or.cond = icmp samesign ult i64 %v596, 2
  %v72 = add nuw i64 %v69, %v51
  %v73 = icmp ult i64 %v72, %v1
  br i1 %or.cond, label %bb19, label %bb21

bb19:                                             ; preds = %bb17
  br i1 %v73, label %bb20, label %bb39

bb20:                                             ; preds = %bb19
  %v75 = getelementptr inbounds i8, ptr %v0, i64 %v72
  %v76 = load i8, ptr %v75, align 1
  %v77 = and i8 %v76, 15
  br label %bb23

bb21:                                             ; preds = %bb17
  br i1 %v73, label %bb22, label %bb40

bb22:                                             ; preds = %bb21
  %v82 = getelementptr inbounds i8, ptr %v0, i64 %v72
  %v83 = load i8, ptr %v82, align 1
  %v86 = lshr i8 %v83, 4
  br label %bb23

bb23:                                             ; preds = %bb22, %bb20
  %v88.in = phi i8 [ %v77, %bb20 ], [ %v86, %bb22 ]
  %v52 = or disjoint i64 %v58, %v53
  %v54 = or disjoint i64 %v52, 128
  %v89 = add nuw i64 %v54, %v30
  %v90 = icmp ult i64 %v89, %v1
  br i1 %v90, label %bb24, label %bb41

bb24:                                             ; preds = %bb23
  %v92 = getelementptr inbounds i8, ptr %v0, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v96 = lshr i8 %v93, %switch.idx.mult
  %v97 = shl i8 %v96, 4
  %2 = and i8 %v97, 48
  %v1018 = or disjoint i8 %2, %v88.in
  %v101 = zext nneg i8 %v1018 to i32
  %v102 = add nsw i32 %v101, -32
  %v104 = shl nuw nsw i64 %v596, 1
  %v103 = or disjoint i64 %v607, %v56
  %v55 = or disjoint i64 %v103, %v104
  %v57 = or disjoint i64 %v55, 192
  %v105 = add nuw i64 %v57, %v30
  %v106 = icmp ult i64 %v105, %v1
  br i1 %v106, label %bb25, label %bb42

bb25:                                             ; preds = %bb24
  %v118 = icmp ult i64 %v22.i, %v6
  %or.cond1.not = select i1 %.v18.i, i1 %v118, i1 false
  %v1329 = icmp ne ptr %v5, null
  %v132 = select i1 %or.cond1.not, i1 %v1329, i1 false
  br i1 %v132, label %bb26, label %bb29

bb26:                                             ; preds = %bb25
  %v121 = getelementptr inbounds nuw float, ptr %v5, i64 %v22.i
  %v108 = getelementptr inbounds i8, ptr %v0, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v111 = sitofp i8 %v109 to float
  %v114 = fmul contract float %v55.i, %v111
  %v115 = sitofp i32 %v102 to float
  %v116 = fmul contract float %v114, %v115
  store float %v116, ptr %v121, align 4
  br label %bb29

bb29:                                             ; preds = %bb25, %bb26, %entry
  ret void

bb37:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb38:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb39:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb23
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb24
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @embedding_q6k_rows(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(address_is_null) %v7, i64 %v8) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i10 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i11 = icmp eq i32 %v4.i9, 1
  %v7.i12 = icmp eq i32 %v6.i10, 1
  %v8.not.not.i = and i1 %v5.i11, %v7.i12
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i13 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i13
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v25 = zext i32 %v4 to i64
  %v26 = zext i32 %v5 to i64
  %v27 = mul nuw i64 %v26, %v25
  %v28.not = icmp ult i64 %v22.i, %v27
  br i1 %v28.not, label %bb3, label %bb23

bb3:                                              ; preds = %entry
  %v30.not = icmp eq i32 %v5, 0
  br i1 %v30.not, label %bb31, label %bb4

bb4:                                              ; preds = %bb3
  %v26.frozen = freeze i64 %v26
  %v32 = udiv i64 %v22.i, %v26.frozen
  %0 = mul i64 %v32, %v26.frozen
  %v33.decomposed = sub i64 %v22.i, %0
  %v39 = icmp ult i64 %v32, %v3
  br i1 %v39, label %bb5, label %bb32

bb5:                                              ; preds = %bb4
  %v361 = lshr i64 %v33.decomposed, 8
  %v34 = zext i32 %v6 to i64
  %v41 = getelementptr inbounds nuw i32, ptr %v2, i64 %v32
  %v42 = load i32, ptr %v41, align 4
  %v43 = zext i32 %v42 to i64
  %v44 = mul nuw i64 %v43, %v34
  %reass.add = add nuw i64 %v44, %v361
  %reass.mul = mul i64 %reass.add, 210
  %v47 = add i64 %reass.mul, 208
  %v49 = icmp ult i64 %v47, %v1
  br i1 %v49, label %bb6, label %bb33

bb6:                                              ; preds = %bb5
  %v53 = add i64 %reass.mul, 209
  %v54 = icmp ult i64 %v53, %v1
  br i1 %v54, label %bb7, label %bb34

bb7:                                              ; preds = %bb6
  %v51 = getelementptr inbounds i8, ptr %v0, i64 %v47
  %v52 = load i8, ptr %v51, align 1
  %v56 = getelementptr inbounds i8, ptr %v0, i64 %v53
  %v57 = load i8, ptr %v56, align 1
  %v61 = alloca [2 x i8], align 2
  store i8 %v52, ptr %v61, align 2
  %v61.repack2 = getelementptr inbounds nuw i8, ptr %v61, i64 1
  store i8 %v57, ptr %v61.repack2, align 1
  %v62 = load i16, ptr %v61, align 2
  %v4.i14 = lshr i16 %v62, 15
  %v6.i15 = zext nneg i16 %v4.i14 to i32
  %v9.i = lshr i16 %v62, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v62, 1023
  %v13.i16 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb7
  %v15.i17 = icmp eq i16 %v12.i, 0
  br i1 %v15.i17, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i18 = shl nuw i32 %v6.i15, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i16, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i16, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i15, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb7
  %v38.i = shl nuw i32 %v6.i15, 31
  %v41.i = shl nuw nsw i32 %v13.i16, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb7
  %v44.i = shl nuw i32 %v6.i15, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i16, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i18, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v37 = lshr i64 %v33.decomposed, 7
  %v644 = and i64 %v37, 1
  %v66 = shl nuw nsw i64 %v644, 6
  %v67 = add i64 %reass.mul, %v66
  %v69 = shl nuw nsw i64 %v644, 5
  %v72 = shl nuw nsw i64 %v644, 3
  %v74 = and i64 %v33.decomposed, 31
  %v65 = lshr i64 %v33.decomposed, 5
  %v755 = and i64 %v65, 3
  %v76 = trunc nuw nsw i64 %v755 to i8
  %v77 = shl nuw nsw i8 %v76, 1
  switch i64 %v755, label %bb11 [
    i64 0, label %bb12
    i64 2, label %bb12
  ]

bb11:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v80 = or disjoint i64 %v74, 32
  br label %bb12

bb12:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb11
  %v81 = phi i64 [ %v80, %bb11 ], [ %v74, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v74, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %v82 = icmp samesign ugt i64 %v755, 1
  %v91 = add i64 %v81, %v67
  %v92 = icmp ult i64 %v91, %v1
  br i1 %v82, label %bb15, label %bb13

bb13:                                             ; preds = %bb12
  br i1 %v92, label %bb14, label %bb35

bb14:                                             ; preds = %bb13
  %v87 = getelementptr inbounds i8, ptr %v0, i64 %v91
  %v88 = load i8, ptr %v87, align 1
  %v89 = and i8 %v88, 15
  br label %bb17

bb15:                                             ; preds = %bb12
  br i1 %v92, label %bb16, label %bb36

bb16:                                             ; preds = %bb15
  %v94 = getelementptr inbounds i8, ptr %v0, i64 %v91
  %v95 = load i8, ptr %v94, align 1
  %v98 = lshr i8 %v95, 4
  br label %bb17

bb17:                                             ; preds = %bb16, %bb14
  %v100.in = phi i8 [ %v89, %bb14 ], [ %v98, %bb16 ]
  %v68 = or disjoint i64 %v74, %v69
  %v70 = or disjoint i64 %v68, 128
  %v101 = add i64 %v70, %reass.mul
  %v102 = icmp ult i64 %v101, %v1
  br i1 %v102, label %bb18, label %bb37

bb18:                                             ; preds = %bb17
  %v104 = getelementptr inbounds i8, ptr %v0, i64 %v101
  %v105 = load i8, ptr %v104, align 1
  %v108 = lshr i8 %v105, %v77
  %v109 = shl i8 %v108, 4
  %3 = and i8 %v109, 48
  %v1136 = or disjoint i8 %3, %v100.in
  %v113 = zext nneg i8 %v1136 to i32
  %v114 = add nsw i32 %v113, -32
  %v131 = icmp ult i64 %v22.i, %v8
  %or.cond.not = select i1 %.v18.i, i1 %v131, i1 false
  %v134 = getelementptr inbounds nuw float, ptr %v7, i64 %v22.i
  %v1458 = icmp ne ptr %v7, null
  %v145 = select i1 %or.cond.not, i1 %v1458, i1 false
  br i1 %v145, label %bb19, label %bb23

bb19:                                             ; preds = %bb18
  %v1177 = lshr i64 %v74, 4
  %v119 = shl nuw nsw i64 %v755, 1
  %v118 = or disjoint i64 %v1177, %v72
  %v71 = or disjoint i64 %v118, %v119
  %v73 = or disjoint i64 %v71, 192
  %v120 = add i64 %v73, %reass.mul
  %v121 = icmp ult i64 %v120, %v1
  br i1 %v121, label %bb20, label %bb38

bb20:                                             ; preds = %bb19
  %v123 = getelementptr inbounds i8, ptr %v0, i64 %v120
  %v124 = load i8, ptr %v123, align 1
  %v126 = sitofp i8 %v124 to float
  %v127 = fmul contract float %v55.i, %v126
  %v128 = sitofp i32 %v114 to float
  %v129 = fmul contract float %v127, %v128
  store float %v129, ptr %v134, align 4
  br label %bb23

bb23:                                             ; preds = %bb18, %bb20, %entry
  ret void

bb31:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb32:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable

bb34:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb35:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb36:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb37:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb38:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @embedding_q8_0_row(ptr readonly captures(none) %v0, i64 %v1, i32 %v2, i32 %v3, i32 %v4, ptr writeonly captures(address_is_null) %v5, i64 %v6) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i6 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i7 = icmp eq i32 %v4.i5, 1
  %v7.i8 = icmp eq i32 %v6.i6, 1
  %v8.not.not.i = and i1 %v5.i7, %v7.i8
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i9 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i9
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v20 = zext i32 %v3 to i64
  %v21.not = icmp ult i64 %v22.i, %v20
  br i1 %v21.not, label %bb3, label %bb11

bb3:                                              ; preds = %entry
  %v23 = zext i32 %v4 to i64
  %v251 = lshr i64 %v22.i, 5
  %v26 = and i64 %v22.i, 31
  %v27 = zext i32 %v2 to i64
  %v28 = mul nuw i64 %v23, %v27
  %reass.add = add nuw i64 %v28, %v251
  %reass.mul = mul i64 %reass.add, 34
  %v32 = icmp ult i64 %reass.mul, %v1
  br i1 %v32, label %bb4, label %bb19

bb4:                                              ; preds = %bb3
  %v36 = or disjoint i64 %reass.mul, 1
  %v37 = icmp ult i64 %v36, %v1
  br i1 %v37, label %bb5, label %bb20

bb5:                                              ; preds = %bb4
  %v34 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v35 = load i8, ptr %v34, align 1
  %v39 = getelementptr inbounds i8, ptr %v0, i64 %v36
  %v40 = load i8, ptr %v39, align 1
  %v44 = alloca [2 x i8], align 2
  store i8 %v35, ptr %v44, align 2
  %v44.repack2 = getelementptr inbounds nuw i8, ptr %v44, i64 1
  store i8 %v40, ptr %v44.repack2, align 1
  %v45 = load i16, ptr %v44, align 2
  %v4.i10 = lshr i16 %v45, 15
  %v6.i11 = zext nneg i16 %v4.i10 to i32
  %v9.i = lshr i16 %v45, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v45, 1023
  %v13.i12 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb5
  %v15.i13 = icmp eq i16 %v12.i, 0
  br i1 %v15.i13, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i14 = shl nuw i32 %v6.i11, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i12, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i12, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i11, 31
  %0 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %0
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb5
  %v38.i = shl nuw i32 %v6.i11, 31
  %v41.i = shl nuw nsw i32 %v13.i12, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb5
  %v44.i = shl nuw i32 %v6.i11, 31
  %1 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %1 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i12, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i14, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v59 = icmp ult i64 %v22.i, %v6
  %or.cond.not = select i1 %.v18.i, i1 %v59, i1 false
  %v62 = getelementptr inbounds nuw float, ptr %v5, i64 %v22.i
  %v734 = icmp ne ptr %v5, null
  %v73 = select i1 %or.cond.not, i1 %v734, i1 false
  br i1 %v73, label %bb7, label %bb11

bb7:                                              ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v49 = add nuw nsw i64 %v26, 2
  %v50 = add i64 %v49, %reass.mul
  %v51 = icmp ult i64 %v50, %v1
  br i1 %v51, label %bb8, label %bb21

bb8:                                              ; preds = %bb7
  %v53 = getelementptr inbounds i8, ptr %v0, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v56 = sitofp i8 %v54 to float
  %v57 = fmul contract float %v55.i, %v56
  store float %v57, ptr %v62, align 4
  br label %bb11

bb11:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb8, %entry
  ret void

bb19:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb20:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb21:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define ptx_kernel void @fill_u32(i32 %v0, ptr writeonly captures(address_is_null) %v1, i64 %v2) #4 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v13 = icmp ult i64 %v22.i, %v2
  %or.cond.not = select i1 %.v18.i, i1 %v13, i1 false
  %v271 = icmp ne ptr %v1, null
  %v27 = select i1 %or.cond.not, i1 %v271, i1 false
  br i1 %v27, label %bb2, label %bb4

bb2:                                              ; preds = %entry
  %v16 = getelementptr inbounds i32, ptr %v1, i64 %v22.i
  store i32 %v0, ptr %v16, align 4
  br label %bb4

bb4:                                              ; preds = %entry, %bb2
  ret void
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @kv_write_row(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i3 = icmp eq i32 %v4.i1, 1
  %v7.i4 = icmp eq i32 %v6.i2, 1
  %v8.not.not.i = and i1 %v5.i3, %v7.i4
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i5 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i5
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v34 = trunc i64 %v22.i to i32
  %v35.not = icmp ugt i32 %v11, %v34
  br i1 %v35.not, label %bb3, label %bb11

bb3:                                              ; preds = %entry
  %v37.not = icmp eq i32 %v12, 0
  br i1 %v37.not, label %bb12, label %bb4

bb4:                                              ; preds = %bb3
  %v7.frozen = freeze i32 %v7
  %v12.frozen = freeze i32 %v12
  %v39 = udiv i32 %v7.frozen, %v12.frozen
  %v41 = mul i32 %v9, %v6
  %v42 = add i32 %v39, %v41
  %v43 = zext i32 %v42 to i64
  %v45 = icmp ugt i64 %v5, %v43
  br i1 %v45, label %bb5, label %bb13

bb5:                                              ; preds = %bb4
  %v62 = and i64 %v22.i, 4294967295
  %v64 = icmp ult i64 %v62, %v1
  br i1 %v64, label %bb9, label %bb14

bb9:                                              ; preds = %bb5
  %v47 = getelementptr inbounds nuw i32, ptr %v4, i64 %v43
  %v48 = load i32, ptr %v47, align 4
  %v49 = zext i32 %v48 to i64
  %v56 = zext i32 %v13 to i64
  %v57 = mul nuw i64 %v49, %v56
  %v52 = icmp eq i32 %v8, 0
  %v50 = zext i32 %v10 to i64
  %v51 = shl nuw nsw i64 %v50, 1
  %v53 = zext i32 %v12 to i64
  %v54 = mul i64 %v51, %v53
  %v55 = select i1 %v52, i64 0, i64 %v54
  %0 = mul i32 %v39, %v12.frozen
  %v40.decomposed = sub i32 %v7.frozen, %0
  %v59 = zext i32 %v40.decomposed to i64
  %v60 = mul i64 %v51, %v59
  %v66 = getelementptr inbounds nuw float, ptr %v0, i64 %v62
  %v6714 = load i32, ptr %v66, align 4
  %v4.i7 = lshr i32 %v6714, 16
  %v5.i8 = and i32 %v4.i7, 32768
  %v7.i9 = lshr i32 %v6714, 23
  %v8.i = and i32 %v7.i9, 255
  %v10.i = and i32 %v6714, 8388607
  %v11.i = icmp eq i32 %v8.i, 255
  br i1 %v11.i, label %bb1.i, label %bb5.i

bb1.i:                                            ; preds = %bb9
  %v13.i12 = icmp eq i32 %v10.i, 0
  %..i = select i1 %v13.i12, i32 0, i32 512
  %v12.i = or disjoint i32 %..i, %v5.i8
  %1 = trunc nuw i32 %v12.i to i16
  %v16.i13 = or disjoint i16 %1, 31744
  br label %cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit

bb5.i:                                            ; preds = %bb9
  %v19.i = icmp samesign ult i32 %v8.i, 143
  br i1 %v19.i, label %bb7.i, label %bb6.i

bb6.i:                                            ; preds = %bb5.i
  %2 = trunc nuw i32 %v5.i8 to i16
  %v22.i10 = or disjoint i16 %2, 31744
  br label %cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit

bb7.i:                                            ; preds = %bb5.i
  %v23.i = icmp samesign ugt i32 %v8.i, 112
  br i1 %v23.i, label %bb9.i, label %bb8.i

bb8.i:                                            ; preds = %bb7.i
  %v25.i = icmp samesign ugt i32 %v8.i, 101
  br i1 %v25.i, label %bb11.i, label %bb10.i

bb9.i:                                            ; preds = %bb7.i
  %v18.i11 = shl nuw nsw i32 %v7.i9, 10
  %v31.i = lshr i32 %v10.i, 13
  %v33.i = lshr i32 %v6714, 12
  %v33.lobit.i = and i32 %v33.i, 1
  %v32.i = or disjoint i32 %v31.i, 16384
  %v29.i = add nuw nsw i32 %v32.i, %v33.lobit.i
  %v49.i = add nuw nsw i32 %v29.i, %v18.i11
  %v50.i = or i32 %v49.i, %v5.i8
  %v51.i = trunc i32 %v50.i to i16
  br label %cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit

bb10.i:                                           ; preds = %bb8.i
  %v35.i = trunc nuw i32 %v5.i8 to i16
  br label %cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit

bb11.i:                                           ; preds = %bb8.i
  %v36.i = or disjoint i32 %v10.i, 8388608
  %v37.i = sub nsw i32 17, %v7.i9
  %v38.i = and i32 %v37.i, 31
  %v39.i = lshr i32 %v36.i, %v38.i
  %v41.i = lshr i32 %v39.i, 13
  %v42.i = lshr i32 %v39.i, 12
  %v42.lobit.i = and i32 %v42.i, 1
  %v45.i = add nuw nsw i32 %v42.lobit.i, %v41.i
  %v46.i = or disjoint i32 %v45.i, %v5.i8
  %v47.i = trunc nuw i32 %v46.i to i16
  br label %cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit

cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit: ; preds = %bb1.i, %bb6.i, %bb9.i, %bb10.i, %bb11.i
  %v53.i = phi i16 [ %v16.i13, %bb1.i ], [ %v22.i10, %bb6.i ], [ %v51.i, %bb9.i ], [ %v35.i, %bb10.i ], [ %v47.i, %bb11.i ]
  %v70 = trunc i16 %v53.i to i8
  %v73 = lshr i16 %v53.i, 8
  %v74 = trunc nuw i16 %v73 to i8
  %v75 = shl nuw nsw i64 %v62, 1
  %3 = getelementptr i8, ptr %v2, i64 %v57
  %4 = getelementptr i8, ptr %3, i64 %v55
  %5 = getelementptr i8, ptr %4, i64 %v60
  %v78 = getelementptr i8, ptr %5, i64 %v75
  store i8 %v70, ptr %v78, align 1
  %v80 = getelementptr i8, ptr %v78, i64 1
  store i8 %v74, ptr %v80, align 1
  br label %bb11

bb11:                                             ; preds = %entry, %cuda_kernels__oxide_kernels__kernels__f32_to_f16_bits.exit
  ret void

bb12:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb13:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb14:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_count_assignments(ptr readonly captures(none) %v0, i64 %v1, ptr captures(none) %v2, i64 %v3) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i3 = icmp eq i32 %v4.i1, 1
  %v7.i4 = icmp eq i32 %v6.i2, 1
  %v8.not.not.i = and i1 %v5.i3, %v7.i4
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i5 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i5
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v14.not = icmp ult i64 %v22.i, %v1
  br i1 %v14.not, label %bb2, label %bb5

bb2:                                              ; preds = %entry
  %v17 = getelementptr inbounds i32, ptr %v0, i64 %v22.i
  %v18 = load i32, ptr %v17, align 4
  %v19 = zext i32 %v18 to i64
  %v21 = icmp ugt i64 %v3, %v19
  br i1 %v21, label %bb3, label %bb6

bb3:                                              ; preds = %bb2
  %v23 = getelementptr inbounds nuw { { i32 } }, ptr %v2, i64 %v19
  %v24 = atomicrmw add ptr %v23, i32 1 syncscope("device") monotonic, align 4
  br label %bb5

bb5:                                              ; preds = %bb3, %entry
  ret void

bb6:                                              ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nofree norecurse nosync nounwind memory(argmem: readwrite)
define ptx_kernel void @moe_prefix_offsets(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3, ptr writeonly captures(none) %v4, i64 %v5) #5 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i3 = icmp eq i32 %v4.i1, 1
  %v7.i4 = icmp eq i32 %v6.i2, 1
  %v8.not.not.i = and i1 %v5.i3, %v7.i4
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i5 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i5
  %0 = or i64 %v17.i, %v7.i
  %v186 = icmp eq i64 %0, 0
  %v18 = select i1 %.v18.i, i1 %v186, i1 false
  %v22.not7 = icmp ne i64 %v1, 0
  %or.cond = select i1 %v18, i1 %v22.not7, i1 false
  br i1 %or.cond, label %bb5.preheader, label %bb8

bb5.preheader:                                    ; preds = %entry
  %xtraiter = and i64 %v1, 1
  %1 = icmp eq i64 %v1, 1
  br i1 %1, label %bb5.epil.preheader, label %bb5.preheader.new

bb5.preheader.new:                                ; preds = %bb5.preheader
  %unroll_iter = and i64 %v1, -2
  br label %bb5

bb5:                                              ; preds = %bb5, %bb5.preheader.new
  %v209 = phi i64 [ 0, %bb5.preheader.new ], [ %v33.1, %bb5 ]
  %v198 = phi i32 [ 0, %bb5.preheader.new ], [ %v32.1, %bb5 ]
  %niter = phi i64 [ 0, %bb5.preheader.new ], [ %niter.next.1, %bb5 ]
  %v25 = getelementptr inbounds i32, ptr %v2, i64 %v209
  store i32 %v198, ptr %v25, align 4
  %v27 = getelementptr inbounds i32, ptr %v4, i64 %v209
  store i32 0, ptr %v27, align 4
  %v30 = getelementptr inbounds i32, ptr %v0, i64 %v209
  %v31 = load i32, ptr %v30, align 4
  %v32 = add i32 %v31, %v198
  %v33 = or disjoint i64 %v209, 1
  %v25.1 = getelementptr inbounds i32, ptr %v2, i64 %v33
  store i32 %v32, ptr %v25.1, align 4
  %v27.1 = getelementptr inbounds i32, ptr %v4, i64 %v33
  store i32 0, ptr %v27.1, align 4
  %v30.1 = getelementptr inbounds i32, ptr %v0, i64 %v33
  %v31.1 = load i32, ptr %v30.1, align 4
  %v32.1 = add i32 %v31.1, %v32
  %v33.1 = add nuw i64 %v209, 2
  %niter.next.1 = add i64 %niter, 2
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter
  br i1 %niter.ncmp.1, label %bb8.loopexit.unr-lcssa, label %bb5

bb8.loopexit.unr-lcssa:                           ; preds = %bb5
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb8, label %bb5.epil.preheader

bb5.epil.preheader:                               ; preds = %bb8.loopexit.unr-lcssa, %bb5.preheader
  %v209.epil.init = phi i64 [ 0, %bb5.preheader ], [ %v33.1, %bb8.loopexit.unr-lcssa ]
  %v198.epil.init = phi i32 [ 0, %bb5.preheader ], [ %v32.1, %bb8.loopexit.unr-lcssa ]
  %lcmp.mod10 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod10)
  %v25.epil = getelementptr inbounds i32, ptr %v2, i64 %v209.epil.init
  store i32 %v198.epil.init, ptr %v25.epil, align 4
  %v27.epil = getelementptr inbounds i32, ptr %v4, i64 %v209.epil.init
  store i32 0, ptr %v27.epil, align 4
  br label %bb8

bb8:                                              ; preds = %bb5.epil.preheader, %bb8.loopexit.unr-lcssa, %entry
  ret void
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_q4k_project(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, ptr writeonly captures(address_is_null) %v13, i64 %v14) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i18 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i19 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i20 = icmp eq i32 %v4.i18, 1
  %v7.i21 = icmp eq i32 %v6.i19, 1
  %v8.not.not.i22 = and i1 %v5.i20, %v7.i21
  %v13.i23 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i24 = icmp eq i32 %v13.i23, 1
  %v15.i25 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i26 = icmp eq i32 %v15.i25, 1
  %v17.i27 = and i1 %v14.i24, %v16.i26
  %.v18.i28 = and i1 %v8.not.not.i22, %v17.i27
  %v22.i = select i1 %.v18.i28, i64 %v18.i, i64 -1
  %v37 = zext i32 %v6 to i64
  %v38 = zext i32 %v7 to i64
  %v39 = mul nuw i64 %v38, %v37
  %v40 = zext i32 %v8 to i64
  %v41 = mul i64 %v39, %v40
  %v42.not = icmp ult i64 %v22.i, %v41
  br i1 %v42.not, label %bb3, label %bb15

bb3:                                              ; preds = %entry
  %v44.not = icmp eq i32 %v8, 0
  br i1 %v44.not, label %bb23, label %bb4

bb4:                                              ; preds = %bb3
  %v40.frozen = freeze i64 %v40
  %v47 = udiv i64 %v22.i, %v40.frozen
  %0 = mul i64 %v47, %v40.frozen
  %v46.decomposed = sub i64 %v22.i, %0
  %v48.not = icmp eq i32 %v7, 0
  br i1 %v48.not, label %bb24, label %bb5

bb5:                                              ; preds = %bb4
  %v52 = icmp ult i64 %v47, %v5
  br i1 %v52, label %bb6, label %bb25

bb6:                                              ; preds = %bb5
  %v50 = udiv i64 %v47, %v38
  %v54 = getelementptr inbounds i32, ptr %v4, i64 %v47
  %v55 = load i32, ptr %v54, align 4
  %v56 = zext i32 %v55 to i64
  %v571 = lshr i32 %v9, 8
  %narrow = mul nuw i32 %v571, 144
  %v59 = zext i32 %narrow to i64
  %v60 = zext i32 %v10 to i64
  %v62 = icmp eq i32 %v12, 0
  %v65 = zext i32 %v9 to i64
  %v50.v47 = select i1 %v62, i64 %v50, i64 %v47
  %v66 = mul i64 %v50.v47, %v65
  %v68 = mul nuw i64 %v56, %v60
  %v69 = zext i32 %v11 to i64
  %v70 = add nuw nsw i64 %v46.decomposed, %v69
  %reass.add = add nuw i64 %v70, %v68
  %reass.mul = mul i64 %reass.add, %v59
  %v77 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v0, i64 %v1, i64 %reass.mul, ptr %v2, i64 %v3, i64 %v66, i32 %v571) #19
  %v2.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i6 = zext nneg i32 %v2.i3 to i64
  %v6.i7 = zext nneg i32 %v3.i4 to i64
  %v17.i8 = mul nuw nsw i64 %v5.i6, %v6.i7
  %v7.i9 = zext nneg i32 %v4.i5 to i64
  %v18.i10 = add nuw nsw i64 %v17.i8, %v7.i9
  %v4.i13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i14 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i15 = icmp eq i32 %v4.i13, 1
  %v7.i16 = icmp eq i32 %v6.i14, 1
  %v8.not.not.i = and i1 %v5.i15, %v7.i16
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i17 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i17
  %v22.i12 = select i1 %.v18.i, i64 %v18.i10, i64 -1
  %v83 = icmp ult i64 %v22.i12, %v14
  %or.cond.not = select i1 %.v18.i, i1 %v83, i1 false
  %v972 = icmp ne ptr %v13, null
  %v97 = select i1 %or.cond.not, i1 %v972, i1 false
  br i1 %v97, label %bb12, label %bb15

bb12:                                             ; preds = %bb6
  %v86 = getelementptr inbounds float, ptr %v13, i64 %v22.i12
  store float %v77, ptr %v86, align 4
  br label %bb15

bb15:                                             ; preds = %bb6, %bb12, %entry
  ret void

bb23:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb24:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb25:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @moe_q4k_project_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14, ptr writeonly captures(none) %v15, i64 %v16) #6 {
entry:
  %v39 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v40 = zext nneg i32 %v39 to i64
  %v41 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v42 = zext nneg i32 %v41 to i64
  %v43 = zext i32 %v8 to i64
  %v44 = zext i32 %v9 to i64
  %v45 = mul nuw i64 %v44, %v43
  %v46 = zext i32 %v10 to i64
  %v47 = mul i64 %v45, %v46
  %v48.not = icmp ugt i64 %v47, %v42
  br i1 %v48.not, label %bb4, label %bb30

bb4:                                              ; preds = %entry
  %v50.not = icmp eq i32 %v10, 0
  br i1 %v50.not, label %bb31, label %bb5

bb5:                                              ; preds = %bb4
  %v10.frozen = freeze i32 %v10
  %v533 = udiv i32 %v41, %v10.frozen
  %0 = mul i32 %v533, %v10.frozen
  %v522.decomposed = sub i32 %v41, %0
  %v52.zext = zext nneg i32 %v522.decomposed to i64
  %v53.zext = zext nneg i32 %v533 to i64
  %v55 = icmp ugt i64 %v7, %v53.zext
  br i1 %v55, label %bb6, label %bb32

bb6:                                              ; preds = %bb5
  %v57 = getelementptr inbounds nuw i32, ptr %v6, i64 %v53.zext
  %v58 = load i32, ptr %v57, align 4
  %v59 = zext i32 %v58 to i64
  %v60.not = icmp eq i32 %v9, 0
  br i1 %v60.not, label %bb33, label %bb7

bb7:                                              ; preds = %bb6
  %v64 = icmp ugt i64 %v5, %v59
  br i1 %v64, label %bb8, label %bb34

bb8:                                              ; preds = %bb7
  %1 = lshr i32 %v11, 8
  %v70 = zext nneg i32 %1 to i64
  %v78.not7 = icmp samesign ult i32 %v39, %1
  br i1 %v78.not7, label %bb13.lr.ph, label %bb15

bb13.lr.ph:                                       ; preds = %bb8
  %v74 = icmp eq i32 %v14, 0
  %v624 = udiv i32 %v58, %v9
  %v62.zext = zext i32 %v624 to i64
  %v62.v59 = select i1 %v74, i64 %v62.zext, i64 %v59
  %v72 = zext i32 %v12 to i64
  %v66 = getelementptr inbounds nuw i32, ptr %v4, i64 %v59
  %v67 = load i32, ptr %v66, align 4
  %v68 = zext i32 %v67 to i64
  %v80 = mul nuw i64 %v68, %v72
  %v81 = zext i32 %v13 to i64
  %v82 = add nuw nsw i64 %v52.zext, %v81
  %reass.add = add nuw i64 %v82, %v80
  %reass.mul = mul i64 %reass.add, %v70
  %v87 = mul nuw nsw i64 %v62.v59, %v70
  br label %bb13

bb13:                                             ; preds = %bb13.lr.ph, %bb13
  %v779 = phi i64 [ %v40, %bb13.lr.ph ], [ %v96, %bb13 ]
  %v768 = phi float [ 0.000000e+00, %bb13.lr.ph ], [ %v95, %bb13 ]
  %reass.add5 = add i64 %v779, %reass.mul
  %reass.mul6 = mul i64 %reass.add5, 144
  %v88 = add nuw nsw i64 %v779, %v87
  %v89 = shl nuw i64 %v88, 8
  %v94 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v0, i64 %v1, i64 %reass.mul6, ptr %v2, i64 %v3, i64 %v89, i32 1) #19
  %v95 = fadd contract float %v768, %v94
  %v96 = add nuw nsw i64 %v779, 32
  %v78.not = icmp samesign ult i64 %v96, %v70
  br i1 %v78.not, label %bb13, label %bb15

bb15:                                             ; preds = %bb13, %bb8
  %v76.lcssa = phi float [ 0.000000e+00, %bb8 ], [ %v95, %bb13 ]
  %v97 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_12, i64 %v40
  store float %v76.lcssa, ptr addrspace(3) %v97, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v102.not = icmp samesign ult i32 %v39, 16
  br i1 %v102.not, label %bb20, label %bb24

bb20:                                             ; preds = %bb15
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v97, i64 64
  %v107 = load float, ptr addrspace(3) %gep, align 4
  %v109 = load float, ptr addrspace(3) %v97, align 4
  %v110 = fadd contract float %v107, %v109
  store float %v110, ptr addrspace(3) %v97, align 4
  br label %bb24

bb24:                                             ; preds = %bb15, %bb20
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v102.not.1 = icmp samesign ult i32 %v39, 8
  br i1 %v102.not.1, label %bb20.1, label %bb24.1

bb20.1:                                           ; preds = %bb24
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v97, i64 32
  %v107.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v109.1 = load float, ptr addrspace(3) %v97, align 4
  %v110.1 = fadd contract float %v107.1, %v109.1
  store float %v110.1, ptr addrspace(3) %v97, align 4
  br label %bb24.1

bb24.1:                                           ; preds = %bb20.1, %bb24
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v102.not.2 = icmp samesign ult i32 %v39, 4
  br i1 %v102.not.2, label %bb20.2, label %bb24.2

bb20.2:                                           ; preds = %bb24.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v97, i64 16
  %v107.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v109.2 = load float, ptr addrspace(3) %v97, align 4
  %v110.2 = fadd contract float %v107.2, %v109.2
  store float %v110.2, ptr addrspace(3) %v97, align 4
  br label %bb24.2

bb24.2:                                           ; preds = %bb20.2, %bb24.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v102.not.3 = icmp samesign ult i32 %v39, 2
  br i1 %v102.not.3, label %bb20.3, label %bb24.3

bb20.3:                                           ; preds = %bb24.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v97, i64 8
  %v107.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v109.3 = load float, ptr addrspace(3) %v97, align 4
  %v110.3 = fadd contract float %v107.3, %v109.3
  store float %v110.3, ptr addrspace(3) %v97, align 4
  br label %bb24.3

bb24.3:                                           ; preds = %bb20.3, %bb24.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v102.not.4 = icmp eq i32 %v39, 0
  br i1 %v102.not.4, label %bb20.4, label %bb24.4

bb20.4:                                           ; preds = %bb24.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v97, i64 4
  %v107.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v109.4 = load float, ptr addrspace(3) %v97, align 4
  %v110.4 = fadd contract float %v107.4, %v109.4
  store float %v110.4, ptr addrspace(3) %v97, align 4
  br label %bb24.4

bb24.4:                                           ; preds = %bb20.4, %bb24.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v113 = icmp eq i32 %v39, 0
  br i1 %v113, label %bb27, label %bb30

bb27:                                             ; preds = %bb24.4
  %v117 = mul nuw i64 %v59, %v46
  %2 = getelementptr float, ptr %v15, i64 %v117
  %v120 = getelementptr float, ptr %2, i64 %v52.zext
  %v116 = load float, ptr addrspace(3) @__shared_mem_12, align 4
  store float %v116, ptr %v120, align 4
  br label %bb30

bb30:                                             ; preds = %bb24.4, %bb27, %entry
  ret void

bb31:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb32:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb34:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_q5_0_project(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr writeonly captures(address_is_null) %v10, i64 %v11) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i11 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i12 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i13 = icmp eq i32 %v4.i11, 1
  %v7.i14 = icmp eq i32 %v6.i12, 1
  %v8.not.not.i = and i1 %v5.i13, %v7.i14
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i15 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i15
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v33 = zext i32 %v6 to i64
  %v34 = zext i32 %v7 to i64
  %v35 = mul nuw i64 %v34, %v33
  %v36 = zext i32 %v8 to i64
  %v37 = mul i64 %v35, %v36
  %v38.not = icmp ult i64 %v22.i, %v37
  br i1 %v38.not, label %bb3, label %bb25

bb3:                                              ; preds = %entry
  %v40.not = icmp eq i32 %v8, 0
  br i1 %v40.not, label %bb33, label %bb4

bb4:                                              ; preds = %bb3
  %v36.frozen = freeze i64 %v36
  %v43 = udiv i64 %v22.i, %v36.frozen
  %v45 = icmp ult i64 %v43, %v5
  br i1 %v45, label %bb5, label %bb34

bb5:                                              ; preds = %bb4
  %0 = mul i64 %v43, %v36.frozen
  %v42.decomposed = sub i64 %v22.i, %0
  %v47 = getelementptr inbounds i32, ptr %v4, i64 %v43
  %v48 = load i32, ptr %v47, align 4
  %v49 = zext i32 %v48 to i64
  %v50 = zext i32 %v9 to i64
  %v511 = lshr i64 %v50, 5
  %v53 = mul nuw i64 %v49, %v36
  %v54 = add nuw i64 %v53, %v42.decomposed
  %v55 = mul i64 %v54, %v511
  %v56 = mul i64 %v43, %v50
  %v59.not35.not = icmp eq i64 %v511, 0
  br i1 %v59.not35.not, label %bb21, label %bb7

bb7:                                              ; preds = %bb5, %bb20
  %v5837 = phi i64 [ %v164, %bb20 ], [ 0, %bb5 ]
  %v5736 = phi float [ %v162, %bb20 ], [ 0.000000e+00, %bb5 ]
  %reass.add = add i64 %v5837, %v55
  %reass.mul = mul i64 %reass.add, 22
  %v64 = icmp ult i64 %reass.mul, %v1
  br i1 %v64, label %bb8, label %bb35

bb8:                                              ; preds = %bb7
  %v68 = or disjoint i64 %reass.mul, 1
  %v69 = icmp ult i64 %v68, %v1
  br i1 %v69, label %bb9, label %bb36

bb9:                                              ; preds = %bb8
  %v66 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v67 = load i8, ptr %v66, align 1
  %v71 = getelementptr inbounds i8, ptr %v0, i64 %v68
  %v72 = load i8, ptr %v71, align 1
  %v76 = alloca [2 x i8], align 2
  store i8 %v67, ptr %v76, align 2
  %v76.repack2 = getelementptr inbounds nuw i8, ptr %v76, i64 1
  store i8 %v72, ptr %v76.repack2, align 1
  %v77 = load i16, ptr %v76, align 2
  %v4.i16 = lshr i16 %v77, 15
  %v6.i17 = zext nneg i16 %v4.i16 to i32
  %v9.i = lshr i16 %v77, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v77, 1023
  %v13.i18 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb9
  %v15.i19 = icmp eq i16 %v12.i, 0
  br i1 %v15.i19, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i20 = shl nuw i32 %v6.i17, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i18, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i18, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i17, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb9
  %v38.i = shl nuw i32 %v6.i17, 31
  %v41.i = shl nuw nsw i32 %v13.i18, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb9
  %v44.i = shl nuw i32 %v6.i17, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i18, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i20, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v79 = add nuw i64 %reass.mul, 2
  %v80 = icmp ult i64 %v79, %v1
  br i1 %v80, label %bb11, label %bb37

bb11:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v82 = getelementptr inbounds i8, ptr %v0, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = add nuw i64 %reass.mul, 3
  %v85 = icmp ult i64 %v84, %v1
  br i1 %v85, label %bb12, label %bb38

bb12:                                             ; preds = %bb11
  %v87 = getelementptr inbounds i8, ptr %v0, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = add nuw i64 %reass.mul, 4
  %v90 = icmp ult i64 %v89, %v1
  br i1 %v90, label %bb13, label %bb39

bb13:                                             ; preds = %bb12
  %v94 = add nuw i64 %reass.mul, 5
  %v95 = icmp ult i64 %v94, %v1
  br i1 %v95, label %bb14, label %bb40

bb14:                                             ; preds = %bb13
  %v92 = getelementptr inbounds i8, ptr %v0, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v97 = getelementptr inbounds i8, ptr %v0, i64 %v94
  %v98 = load i8, ptr %v97, align 1
  %v104 = alloca [4 x i8], align 4
  store i8 %v83, ptr %v104, align 4
  %v104.repack4 = getelementptr inbounds nuw i8, ptr %v104, i64 1
  store i8 %v88, ptr %v104.repack4, align 1
  %v104.repack6 = getelementptr inbounds nuw i8, ptr %v104, i64 2
  store i8 %v93, ptr %v104.repack6, align 2
  %v104.repack8 = getelementptr inbounds nuw i8, ptr %v104, i64 3
  store i8 %v98, ptr %v104.repack8, align 1
  %v105 = load i32, ptr %v104, align 4
  %v106 = shl i64 %v5837, 5
  %v107 = add i64 %v106, %v56
  %v112 = add nuw i64 %reass.mul, 6
  br label %bb16

bb16:                                             ; preds = %bb14, %bb19
  %v10934 = phi i64 [ 0, %bb14 ], [ %v163, %bb19 ]
  %v10833 = phi float [ %v5736, %bb14 ], [ %v162, %bb19 ]
  %v113 = add nuw i64 %v112, %v10934
  %v114 = icmp ult i64 %v113, %v1
  br i1 %v114, label %bb17, label %bb41

bb17:                                             ; preds = %bb16
  %v145 = add nuw i64 %v107, %v10934
  %v147 = icmp ult i64 %v145, %v3
  br i1 %v147, label %bb18, label %bb42

bb18:                                             ; preds = %bb17
  %v155 = add i64 %v145, 16
  %v156 = icmp ult i64 %v155, %v3
  br i1 %v156, label %bb19, label %bb43

bb19:                                             ; preds = %bb18
  %3 = trunc nuw nsw i64 %v10934 to i32
  %v130 = or disjoint i32 %3, 16
  %v132 = lshr i32 %v105, %v130
  %v133 = shl nuw nsw i32 %v132, 4
  %v136 = and i32 %v133, 16
  %v116 = getelementptr inbounds i8, ptr %v0, i64 %v113
  %v117 = load i8, ptr %v116, align 1
  %v139 = lshr i8 %v117, 4
  %v140 = zext nneg i8 %v139 to i32
  %v141 = add nsw i32 %v136, -16
  %v142 = or disjoint i32 %v141, %v140
  %v152 = sitofp i32 %v142 to float
  %v153 = fmul contract float %v55.i, %v152
  %v120 = lshr i32 %v105, %3
  %v121 = shl i32 %v120, 4
  %v124 = and i32 %v121, 16
  %v125 = and i8 %v117, 15
  %v126 = zext nneg i8 %v125 to i32
  %v127 = add nsw i32 %v124, -16
  %v128 = or disjoint i32 %v127, %v126
  %v143 = sitofp i32 %v128 to float
  %v144 = fmul contract float %v55.i, %v143
  %v149 = getelementptr inbounds float, ptr %v2, i64 %v145
  %v150 = load float, ptr %v149, align 4
  %v151 = fmul contract float %v150, %v144
  %v158 = getelementptr inbounds float, ptr %v2, i64 %v155
  %v159 = load float, ptr %v158, align 4
  %v160 = fmul contract float %v159, %v153
  %v161 = fadd contract float %v151, %v160
  %v162 = fadd contract float %v10833, %v161
  %v163 = add nuw nsw i64 %v10934, 1
  %exitcond = icmp eq i64 %v163, 16
  br i1 %exitcond, label %bb20, label %bb16

bb20:                                             ; preds = %bb19
  %v164 = add nuw nsw i64 %v5837, 1
  %exitcond38.not = icmp eq i64 %v164, %v511
  br i1 %exitcond38.not, label %bb21, label %bb7

bb21:                                             ; preds = %bb20, %bb5
  %v57.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v162, %bb20 ]
  %v168 = icmp ult i64 %v22.i, %v11
  %or.cond.not = select i1 %.v18.i, i1 %v168, i1 false
  %v18210 = icmp ne ptr %v10, null
  %v182 = select i1 %or.cond.not, i1 %v18210, i1 false
  br i1 %v182, label %bb22, label %bb25

bb22:                                             ; preds = %bb21
  %v171 = getelementptr inbounds float, ptr %v10, i64 %v22.i
  store float %v57.lcssa, ptr %v171, align 4
  br label %bb25

bb25:                                             ; preds = %bb21, %bb22, %entry
  ret void

bb33:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb34:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb35:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb36:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb37:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  tail call void @llvm.trap() #19
  unreachable

bb38:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb39:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @moe_q5_0_project_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr writeonly captures(none) %v12, i64 %v13) #6 {
entry:
  %v35 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v36 = zext nneg i32 %v35 to i64
  %v37 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v38 = zext nneg i32 %v37 to i64
  %v39 = zext i32 %v8 to i64
  %v40 = zext i32 %v9 to i64
  %v41 = mul nuw i64 %v40, %v39
  %v42 = zext i32 %v10 to i64
  %v43 = mul i64 %v41, %v42
  %v44.not = icmp ugt i64 %v43, %v38
  br i1 %v44.not, label %bb4, label %bb38

bb4:                                              ; preds = %entry
  %v46.not = icmp eq i32 %v10, 0
  br i1 %v46.not, label %bb39, label %bb5

bb5:                                              ; preds = %bb4
  %v10.frozen = freeze i32 %v10
  %v4911 = udiv i32 %v37, %v10.frozen
  %0 = mul i32 %v4911, %v10.frozen
  %v4810.decomposed = sub i32 %v37, %0
  %v48.zext = zext nneg i32 %v4810.decomposed to i64
  %v49.zext = zext nneg i32 %v4911 to i64
  %v51 = icmp ugt i64 %v7, %v49.zext
  br i1 %v51, label %bb6, label %bb40

bb6:                                              ; preds = %bb5
  %v53 = getelementptr inbounds nuw i32, ptr %v6, i64 %v49.zext
  %v54 = load i32, ptr %v53, align 4
  %v55 = zext i32 %v54 to i64
  %v57 = icmp ugt i64 %v5, %v55
  br i1 %v57, label %bb7, label %bb41

bb7:                                              ; preds = %bb6
  %1 = lshr i32 %v11, 5
  %v63 = zext nneg i32 %1 to i64
  %v67.not26 = icmp samesign ult i32 %v35, %1
  br i1 %v67.not26, label %bb9.lr.ph, label %bb23

bb9.lr.ph:                                        ; preds = %bb7
  %v59 = getelementptr inbounds nuw i32, ptr %v4, i64 %v55
  %v60 = load i32, ptr %v59, align 4
  %v61 = zext i32 %v60 to i64
  %v69 = mul nuw i64 %v61, %v42
  %v70 = add nuw i64 %v69, %v48.zext
  %v71 = mul i64 %v70, %v63
  %v117 = mul nuw nsw i64 %v55, %v63
  br label %bb9

bb9:                                              ; preds = %bb9.lr.ph, %bb22
  %v6628 = phi i64 [ %v36, %bb9.lr.ph ], [ %v176, %bb22 ]
  %v6527 = phi float [ 0.000000e+00, %bb9.lr.ph ], [ %v174, %bb22 ]
  %reass.add = add i64 %v6628, %v71
  %reass.mul = mul i64 %reass.add, 22
  %v75 = icmp ult i64 %reass.mul, %v1
  br i1 %v75, label %bb10, label %bb42

bb10:                                             ; preds = %bb9
  %v79 = or disjoint i64 %reass.mul, 1
  %v80 = icmp ult i64 %v79, %v1
  br i1 %v80, label %bb11, label %bb43

bb11:                                             ; preds = %bb10
  %v77 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v78 = load i8, ptr %v77, align 1
  %v82 = getelementptr inbounds i8, ptr %v0, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v87 = alloca [2 x i8], align 2
  store i8 %v78, ptr %v87, align 2
  %v87.repack1 = getelementptr inbounds nuw i8, ptr %v87, i64 1
  store i8 %v83, ptr %v87.repack1, align 1
  %v88 = load i16, ptr %v87, align 2
  %v4.i = lshr i16 %v88, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v88, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v88, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb11
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %2 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %2
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb11
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb11
  %v44.i = shl nuw i32 %v6.i, 31
  %3 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %3 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v90 = add nuw i64 %reass.mul, 2
  %v91 = icmp ult i64 %v90, %v1
  br i1 %v91, label %bb13, label %bb44

bb13:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v93 = getelementptr inbounds i8, ptr %v0, i64 %v90
  %v94 = load i8, ptr %v93, align 1
  %v95 = add nuw i64 %reass.mul, 3
  %v96 = icmp ult i64 %v95, %v1
  br i1 %v96, label %bb14, label %bb45

bb14:                                             ; preds = %bb13
  %v98 = getelementptr inbounds i8, ptr %v0, i64 %v95
  %v99 = load i8, ptr %v98, align 1
  %v100 = add nuw i64 %reass.mul, 4
  %v101 = icmp ult i64 %v100, %v1
  br i1 %v101, label %bb15, label %bb46

bb15:                                             ; preds = %bb14
  %v105 = add nuw i64 %reass.mul, 5
  %v106 = icmp ult i64 %v105, %v1
  br i1 %v106, label %bb16, label %bb47

bb16:                                             ; preds = %bb15
  %v103 = getelementptr inbounds i8, ptr %v0, i64 %v100
  %v104 = load i8, ptr %v103, align 1
  %v108 = getelementptr inbounds i8, ptr %v0, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v115 = alloca [4 x i8], align 4
  store i8 %v94, ptr %v115, align 4
  %v115.repack3 = getelementptr inbounds nuw i8, ptr %v115, i64 1
  store i8 %v99, ptr %v115.repack3, align 1
  %v115.repack5 = getelementptr inbounds nuw i8, ptr %v115, i64 2
  store i8 %v104, ptr %v115.repack5, align 2
  %v115.repack7 = getelementptr inbounds nuw i8, ptr %v115, i64 3
  store i8 %v109, ptr %v115.repack7, align 1
  %v116 = load i32, ptr %v115, align 4
  %v118 = add nuw i64 %v6628, %v117
  %v119 = shl i64 %v118, 5
  %v124 = add nuw i64 %reass.mul, 6
  br label %bb18

bb18:                                             ; preds = %bb16, %bb21
  %v12125 = phi i64 [ 0, %bb16 ], [ %v175, %bb21 ]
  %v12024 = phi float [ %v6527, %bb16 ], [ %v174, %bb21 ]
  %v125 = add nuw i64 %v124, %v12125
  %v126 = icmp ult i64 %v125, %v1
  br i1 %v126, label %bb19, label %bb48

bb19:                                             ; preds = %bb18
  %v157 = add nuw nsw i64 %v12125, %v119
  %v159 = icmp ult i64 %v157, %v3
  br i1 %v159, label %bb20, label %bb49

bb20:                                             ; preds = %bb19
  %v167 = or disjoint i64 %v157, 16
  %v168 = icmp ult i64 %v167, %v3
  br i1 %v168, label %bb21, label %bb50

bb21:                                             ; preds = %bb20
  %4 = trunc nuw nsw i64 %v12125 to i32
  %v142 = or disjoint i32 %4, 16
  %v144 = lshr i32 %v116, %v142
  %v145 = shl nuw nsw i32 %v144, 4
  %v148 = and i32 %v145, 16
  %v128 = getelementptr inbounds i8, ptr %v0, i64 %v125
  %v129 = load i8, ptr %v128, align 1
  %v151 = lshr i8 %v129, 4
  %v152 = zext nneg i8 %v151 to i32
  %v153 = add nsw i32 %v148, -16
  %v154 = or disjoint i32 %v153, %v152
  %v164 = sitofp i32 %v154 to float
  %v165 = fmul contract float %v55.i, %v164
  %v132 = lshr i32 %v116, %4
  %v133 = shl i32 %v132, 4
  %v136 = and i32 %v133, 16
  %v137 = and i8 %v129, 15
  %v138 = zext nneg i8 %v137 to i32
  %v139 = add nsw i32 %v136, -16
  %v140 = or disjoint i32 %v139, %v138
  %v155 = sitofp i32 %v140 to float
  %v156 = fmul contract float %v55.i, %v155
  %v161 = getelementptr inbounds float, ptr %v2, i64 %v157
  %v162 = load float, ptr %v161, align 4
  %v163 = fmul contract float %v162, %v156
  %v170 = getelementptr inbounds float, ptr %v2, i64 %v167
  %v171 = load float, ptr %v170, align 4
  %v172 = fmul contract float %v171, %v165
  %v173 = fadd contract float %v163, %v172
  %v174 = fadd contract float %v12024, %v173
  %v175 = add nuw nsw i64 %v12125, 1
  %exitcond = icmp eq i64 %v175, 16
  br i1 %exitcond, label %bb22, label %bb18

bb22:                                             ; preds = %bb21
  %v176 = add nuw nsw i64 %v6628, 32
  %v67.not = icmp samesign ult i64 %v176, %v63
  br i1 %v67.not, label %bb9, label %bb23

bb23:                                             ; preds = %bb22, %bb7
  %v65.lcssa = phi float [ 0.000000e+00, %bb7 ], [ %v174, %bb22 ]
  %v177 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_6, i64 %v36
  store float %v65.lcssa, ptr addrspace(3) %v177, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v182.not = icmp samesign ult i32 %v35, 16
  br i1 %v182.not, label %bb28, label %bb32

bb28:                                             ; preds = %bb23
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v177, i64 64
  %v187 = load float, ptr addrspace(3) %gep, align 4
  %v189 = load float, ptr addrspace(3) %v177, align 4
  %v190 = fadd contract float %v187, %v189
  store float %v190, ptr addrspace(3) %v177, align 4
  br label %bb32

bb32:                                             ; preds = %bb23, %bb28
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v182.not.1 = icmp samesign ult i32 %v35, 8
  br i1 %v182.not.1, label %bb28.1, label %bb32.1

bb28.1:                                           ; preds = %bb32
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v177, i64 32
  %v187.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v189.1 = load float, ptr addrspace(3) %v177, align 4
  %v190.1 = fadd contract float %v187.1, %v189.1
  store float %v190.1, ptr addrspace(3) %v177, align 4
  br label %bb32.1

bb32.1:                                           ; preds = %bb28.1, %bb32
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v182.not.2 = icmp samesign ult i32 %v35, 4
  br i1 %v182.not.2, label %bb28.2, label %bb32.2

bb28.2:                                           ; preds = %bb32.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v177, i64 16
  %v187.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v189.2 = load float, ptr addrspace(3) %v177, align 4
  %v190.2 = fadd contract float %v187.2, %v189.2
  store float %v190.2, ptr addrspace(3) %v177, align 4
  br label %bb32.2

bb32.2:                                           ; preds = %bb28.2, %bb32.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v182.not.3 = icmp samesign ult i32 %v35, 2
  br i1 %v182.not.3, label %bb28.3, label %bb32.3

bb28.3:                                           ; preds = %bb32.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v177, i64 8
  %v187.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v189.3 = load float, ptr addrspace(3) %v177, align 4
  %v190.3 = fadd contract float %v187.3, %v189.3
  store float %v190.3, ptr addrspace(3) %v177, align 4
  br label %bb32.3

bb32.3:                                           ; preds = %bb28.3, %bb32.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v182.not.4 = icmp eq i32 %v35, 0
  br i1 %v182.not.4, label %bb28.4, label %bb32.4

bb28.4:                                           ; preds = %bb32.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v177, i64 4
  %v187.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v189.4 = load float, ptr addrspace(3) %v177, align 4
  %v190.4 = fadd contract float %v187.4, %v189.4
  store float %v190.4, ptr addrspace(3) %v177, align 4
  br label %bb32.4

bb32.4:                                           ; preds = %bb28.4, %bb32.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v193 = icmp eq i32 %v35, 0
  br i1 %v193, label %bb35, label %bb38

bb35:                                             ; preds = %bb32.4
  %v197 = mul nuw i64 %v55, %v42
  %5 = getelementptr float, ptr %v12, i64 %v197
  %v200 = getelementptr float, ptr %5, i64 %v48.zext
  %v196 = load float, ptr addrspace(3) @__shared_mem_6, align 4
  store float %v196, ptr %v200, align 4
  br label %bb38

bb38:                                             ; preds = %bb32.4, %bb35, %entry
  ret void

bb39:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb44:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb48:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb49:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb50:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_q6k_project(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr writeonly captures(address_is_null) %v10, i64 %v11) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i18 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i19 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i20 = icmp eq i32 %v4.i18, 1
  %v7.i21 = icmp eq i32 %v6.i19, 1
  %v8.not.not.i22 = and i1 %v5.i20, %v7.i21
  %v13.i23 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i24 = icmp eq i32 %v13.i23, 1
  %v15.i25 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i26 = icmp eq i32 %v15.i25, 1
  %v17.i27 = and i1 %v14.i24, %v16.i26
  %.v18.i28 = and i1 %v8.not.not.i22, %v17.i27
  %v22.i = select i1 %.v18.i28, i64 %v18.i, i64 -1
  %v31 = zext i32 %v6 to i64
  %v32 = zext i32 %v7 to i64
  %v33 = mul nuw i64 %v32, %v31
  %v34 = zext i32 %v8 to i64
  %v35 = mul i64 %v33, %v34
  %v36.not = icmp ult i64 %v22.i, %v35
  br i1 %v36.not, label %bb3, label %bb10

bb3:                                              ; preds = %entry
  %v38.not = icmp eq i32 %v8, 0
  br i1 %v38.not, label %bb18, label %bb4

bb4:                                              ; preds = %bb3
  %v34.frozen = freeze i64 %v34
  %v41 = udiv i64 %v22.i, %v34.frozen
  %v43 = icmp ult i64 %v41, %v5
  br i1 %v43, label %bb5, label %bb19

bb5:                                              ; preds = %bb4
  %0 = mul i64 %v41, %v34.frozen
  %v40.decomposed = sub i64 %v22.i, %0
  %v45 = getelementptr inbounds i32, ptr %v4, i64 %v41
  %v46 = load i32, ptr %v45, align 4
  %v47 = zext i32 %v46 to i64
  %v481 = lshr i32 %v9, 8
  %narrow = mul nuw i32 %v481, 210
  %v50 = zext i32 %narrow to i64
  %v52 = mul nuw i64 %v47, %v34
  %reass.add = add nuw i64 %v52, %v40.decomposed
  %reass.mul = mul i64 %reass.add, %v50
  %v55 = zext i32 %v9 to i64
  %v56 = mul i64 %v41, %v55
  %v61 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k(ptr %v0, i64 %v1, i64 %reass.mul, ptr %v2, i64 %v3, i64 %v56, i32 %v481) #19
  %v2.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i6 = zext nneg i32 %v2.i3 to i64
  %v6.i7 = zext nneg i32 %v3.i4 to i64
  %v17.i8 = mul nuw nsw i64 %v5.i6, %v6.i7
  %v7.i9 = zext nneg i32 %v4.i5 to i64
  %v18.i10 = add nuw nsw i64 %v17.i8, %v7.i9
  %v4.i13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i14 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i15 = icmp eq i32 %v4.i13, 1
  %v7.i16 = icmp eq i32 %v6.i14, 1
  %v8.not.not.i = and i1 %v5.i15, %v7.i16
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i17 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i17
  %v22.i12 = select i1 %.v18.i, i64 %v18.i10, i64 -1
  %v67 = icmp ult i64 %v22.i12, %v11
  %or.cond.not = select i1 %.v18.i, i1 %v67, i1 false
  %v812 = icmp ne ptr %v10, null
  %v81 = select i1 %or.cond.not, i1 %v812, i1 false
  br i1 %v81, label %bb8, label %bb10

bb8:                                              ; preds = %bb5
  %v70 = getelementptr inbounds float, ptr %v10, i64 %v22.i12
  store float %v61, ptr %v70, align 4
  br label %bb10

bb10:                                             ; preds = %bb5, %entry, %bb8
  ret void

bb18:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb19:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_q8_0_project(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, ptr writeonly captures(address_is_null) %v10, i64 %v11) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i6 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i7 = icmp eq i32 %v4.i5, 1
  %v7.i8 = icmp eq i32 %v6.i6, 1
  %v8.not.not.i = and i1 %v5.i7, %v7.i8
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i9 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i9
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v32 = zext i32 %v6 to i64
  %v33 = zext i32 %v7 to i64
  %v34 = mul nuw i64 %v33, %v32
  %v35 = zext i32 %v8 to i64
  %v36 = mul i64 %v34, %v35
  %v37.not = icmp ult i64 %v22.i, %v36
  br i1 %v37.not, label %bb3, label %bb20

bb3:                                              ; preds = %entry
  %v39.not = icmp eq i32 %v8, 0
  br i1 %v39.not, label %bb28, label %bb4

bb4:                                              ; preds = %bb3
  %v35.frozen = freeze i64 %v35
  %v42 = udiv i64 %v22.i, %v35.frozen
  %v44 = icmp ult i64 %v42, %v5
  br i1 %v44, label %bb5, label %bb29

bb5:                                              ; preds = %bb4
  %0 = mul i64 %v42, %v35.frozen
  %v41.decomposed = sub i64 %v22.i, %0
  %v46 = getelementptr inbounds i32, ptr %v4, i64 %v42
  %v47 = load i32, ptr %v46, align 4
  %v48 = zext i32 %v47 to i64
  %v49 = zext i32 %v9 to i64
  %v501 = lshr i64 %v49, 5
  %v52 = mul nuw i64 %v48, %v35
  %v53 = add nuw i64 %v52, %v41.decomposed
  %v54 = mul i64 %v53, %v501
  %v55 = mul i64 %v42, %v49
  %v58.not23.not = icmp eq i64 %v501, 0
  br i1 %v58.not23.not, label %bb16, label %bb7

bb7:                                              ; preds = %bb5, %bb15
  %v5725 = phi i64 [ %v102, %bb15 ], [ 0, %bb5 ]
  %v5624 = phi float [ %v100, %bb15 ], [ 0.000000e+00, %bb5 ]
  %reass.add = add i64 %v5725, %v54
  %reass.mul = mul i64 %reass.add, 34
  %v63 = icmp ult i64 %reass.mul, %v1
  br i1 %v63, label %bb8, label %bb30

bb8:                                              ; preds = %bb7
  %v67 = or disjoint i64 %reass.mul, 1
  %v68 = icmp ult i64 %v67, %v1
  br i1 %v68, label %bb9, label %bb31

bb9:                                              ; preds = %bb8
  %v65 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v66 = load i8, ptr %v65, align 1
  %v70 = getelementptr inbounds i8, ptr %v0, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v75 = alloca [2 x i8], align 2
  store i8 %v66, ptr %v75, align 2
  %v75.repack2 = getelementptr inbounds nuw i8, ptr %v75, i64 1
  store i8 %v71, ptr %v75.repack2, align 1
  %v76 = load i16, ptr %v75, align 2
  %v4.i10 = lshr i16 %v76, 15
  %v6.i11 = zext nneg i16 %v4.i10 to i32
  %v9.i = lshr i16 %v76, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v76, 1023
  %v13.i12 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb9
  %v15.i13 = icmp eq i16 %v12.i, 0
  br i1 %v15.i13, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i14 = shl nuw i32 %v6.i11, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i12, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i12, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i11, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb9
  %v38.i = shl nuw i32 %v6.i11, 31
  %v41.i = shl nuw nsw i32 %v13.i12, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb9
  %v44.i = shl nuw i32 %v6.i11, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i12, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i14, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v78 = shl i64 %v5725, 5
  %v79 = add i64 %v78, %v55
  %v84 = add nuw i64 %reass.mul, 2
  br label %bb12

bb12:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb14
  %v8122 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v101, %bb14 ]
  %v8021 = phi float [ %v5624, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v100, %bb14 ]
  %v85 = add nuw i64 %v84, %v8122
  %v86 = icmp ult i64 %v85, %v1
  br i1 %v86, label %bb13, label %bb32

bb13:                                             ; preds = %bb12
  %v93 = add nuw i64 %v79, %v8122
  %v95 = icmp ult i64 %v93, %v3
  br i1 %v95, label %bb14, label %bb33

bb14:                                             ; preds = %bb13
  %v88 = getelementptr inbounds i8, ptr %v0, i64 %v85
  %v89 = load i8, ptr %v88, align 1
  %v91 = sitofp i8 %v89 to float
  %v92 = fmul contract float %v55.i, %v91
  %v97 = getelementptr inbounds float, ptr %v2, i64 %v93
  %v98 = load float, ptr %v97, align 4
  %v99 = fmul contract float %v98, %v92
  %v100 = fadd contract float %v8021, %v99
  %v101 = add nuw nsw i64 %v8122, 1
  %exitcond = icmp eq i64 %v101, 32
  br i1 %exitcond, label %bb15, label %bb12

bb15:                                             ; preds = %bb14
  %v102 = add nuw nsw i64 %v5725, 1
  %exitcond26.not = icmp eq i64 %v102, %v501
  br i1 %exitcond26.not, label %bb16, label %bb7

bb16:                                             ; preds = %bb15, %bb5
  %v56.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v100, %bb15 ]
  %v106 = icmp ult i64 %v22.i, %v11
  %or.cond.not = select i1 %.v18.i, i1 %v106, i1 false
  %v1204 = icmp ne ptr %v10, null
  %v120 = select i1 %or.cond.not, i1 %v1204, i1 false
  br i1 %v120, label %bb17, label %bb20

bb17:                                             ; preds = %bb16
  %v109 = getelementptr inbounds float, ptr %v10, i64 %v22.i
  store float %v56.lcssa, ptr %v109, align 4
  br label %bb20

bb20:                                             ; preds = %bb16, %bb17, %entry
  ret void

bb28:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb29:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb30:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb31:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb32:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @moe_q8_0_project_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr writeonly captures(none) %v12, i64 %v13) #6 {
entry:
  %v34 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v35 = zext nneg i32 %v34 to i64
  %v36 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v37 = zext nneg i32 %v36 to i64
  %v38 = zext i32 %v8 to i64
  %v39 = zext i32 %v9 to i64
  %v40 = mul nuw i64 %v39, %v38
  %v41 = zext i32 %v10 to i64
  %v42 = mul i64 %v40, %v41
  %v43.not = icmp ugt i64 %v42, %v37
  br i1 %v43.not, label %bb4, label %bb33

bb4:                                              ; preds = %entry
  %v45.not = icmp eq i32 %v10, 0
  br i1 %v45.not, label %bb34, label %bb5

bb5:                                              ; preds = %bb4
  %v10.frozen = freeze i32 %v10
  %v485 = udiv i32 %v36, %v10.frozen
  %0 = mul i32 %v485, %v10.frozen
  %v474.decomposed = sub i32 %v36, %0
  %v47.zext = zext nneg i32 %v474.decomposed to i64
  %v48.zext = zext nneg i32 %v485 to i64
  %v50 = icmp ugt i64 %v7, %v48.zext
  br i1 %v50, label %bb6, label %bb35

bb6:                                              ; preds = %bb5
  %v52 = getelementptr inbounds nuw i32, ptr %v6, i64 %v48.zext
  %v53 = load i32, ptr %v52, align 4
  %v54 = zext i32 %v53 to i64
  %v56 = icmp ugt i64 %v5, %v54
  br i1 %v56, label %bb7, label %bb36

bb7:                                              ; preds = %bb6
  %1 = lshr i32 %v11, 5
  %v62 = zext nneg i32 %1 to i64
  %v66.not14 = icmp samesign ult i32 %v34, %1
  br i1 %v66.not14, label %bb9.lr.ph, label %bb18

bb9.lr.ph:                                        ; preds = %bb7
  %v58 = getelementptr inbounds nuw i32, ptr %v4, i64 %v54
  %v59 = load i32, ptr %v58, align 4
  %v60 = zext i32 %v59 to i64
  %v68 = mul nuw i64 %v60, %v41
  %v69 = add nuw i64 %v68, %v47.zext
  %v70 = mul i64 %v69, %v62
  %v89 = mul nuw nsw i64 %v54, %v62
  br label %bb9

bb9:                                              ; preds = %bb9.lr.ph, %bb17
  %v6516 = phi i64 [ %v35, %bb9.lr.ph ], [ %v114, %bb17 ]
  %v6415 = phi float [ 0.000000e+00, %bb9.lr.ph ], [ %v112, %bb17 ]
  %reass.add = add i64 %v6516, %v70
  %reass.mul = mul i64 %reass.add, 34
  %v74 = icmp ult i64 %reass.mul, %v1
  br i1 %v74, label %bb10, label %bb37

bb10:                                             ; preds = %bb9
  %v78 = or disjoint i64 %reass.mul, 1
  %v79 = icmp ult i64 %v78, %v1
  br i1 %v79, label %bb11, label %bb38

bb11:                                             ; preds = %bb10
  %v76 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v77 = load i8, ptr %v76, align 1
  %v81 = getelementptr inbounds i8, ptr %v0, i64 %v78
  %v82 = load i8, ptr %v81, align 1
  %v86 = alloca [2 x i8], align 2
  store i8 %v77, ptr %v86, align 2
  %v86.repack1 = getelementptr inbounds nuw i8, ptr %v86, i64 1
  store i8 %v82, ptr %v86.repack1, align 1
  %v87 = load i16, ptr %v86, align 2
  %v4.i = lshr i16 %v87, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v87, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v87, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb11
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %2 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %2
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb11
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb11
  %v44.i = shl nuw i32 %v6.i, 31
  %3 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %3 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v90 = add nuw i64 %v6516, %v89
  %v91 = shl i64 %v90, 5
  %v96 = add nuw i64 %reass.mul, 2
  br label %bb14

bb14:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb16
  %v9313 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v113, %bb16 ]
  %v9212 = phi float [ %v6415, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v112, %bb16 ]
  %v97 = add nuw i64 %v96, %v9313
  %v98 = icmp ult i64 %v97, %v1
  br i1 %v98, label %bb15, label %bb39

bb15:                                             ; preds = %bb14
  %v105 = add nuw nsw i64 %v9313, %v91
  %v107 = icmp ult i64 %v105, %v3
  br i1 %v107, label %bb16, label %bb40

bb16:                                             ; preds = %bb15
  %v100 = getelementptr inbounds i8, ptr %v0, i64 %v97
  %v101 = load i8, ptr %v100, align 1
  %v103 = sitofp i8 %v101 to float
  %v104 = fmul contract float %v55.i, %v103
  %v109 = getelementptr inbounds float, ptr %v2, i64 %v105
  %v110 = load float, ptr %v109, align 4
  %v111 = fmul contract float %v110, %v104
  %v112 = fadd contract float %v9212, %v111
  %v113 = add nuw nsw i64 %v9313, 1
  %exitcond = icmp eq i64 %v113, 32
  br i1 %exitcond, label %bb17, label %bb14

bb17:                                             ; preds = %bb16
  %v114 = add nuw nsw i64 %v6516, 32
  %v66.not = icmp samesign ult i64 %v114, %v62
  br i1 %v66.not, label %bb9, label %bb18

bb18:                                             ; preds = %bb17, %bb7
  %v64.lcssa = phi float [ 0.000000e+00, %bb7 ], [ %v112, %bb17 ]
  %v115 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_7, i64 %v35
  store float %v64.lcssa, ptr addrspace(3) %v115, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v120.not = icmp samesign ult i32 %v34, 16
  br i1 %v120.not, label %bb23, label %bb27

bb23:                                             ; preds = %bb18
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v115, i64 64
  %v125 = load float, ptr addrspace(3) %gep, align 4
  %v127 = load float, ptr addrspace(3) %v115, align 4
  %v128 = fadd contract float %v125, %v127
  store float %v128, ptr addrspace(3) %v115, align 4
  br label %bb27

bb27:                                             ; preds = %bb18, %bb23
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v120.not.1 = icmp samesign ult i32 %v34, 8
  br i1 %v120.not.1, label %bb23.1, label %bb27.1

bb23.1:                                           ; preds = %bb27
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v115, i64 32
  %v125.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v127.1 = load float, ptr addrspace(3) %v115, align 4
  %v128.1 = fadd contract float %v125.1, %v127.1
  store float %v128.1, ptr addrspace(3) %v115, align 4
  br label %bb27.1

bb27.1:                                           ; preds = %bb23.1, %bb27
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v120.not.2 = icmp samesign ult i32 %v34, 4
  br i1 %v120.not.2, label %bb23.2, label %bb27.2

bb23.2:                                           ; preds = %bb27.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v115, i64 16
  %v125.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v127.2 = load float, ptr addrspace(3) %v115, align 4
  %v128.2 = fadd contract float %v125.2, %v127.2
  store float %v128.2, ptr addrspace(3) %v115, align 4
  br label %bb27.2

bb27.2:                                           ; preds = %bb23.2, %bb27.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v120.not.3 = icmp samesign ult i32 %v34, 2
  br i1 %v120.not.3, label %bb23.3, label %bb27.3

bb23.3:                                           ; preds = %bb27.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v115, i64 8
  %v125.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v127.3 = load float, ptr addrspace(3) %v115, align 4
  %v128.3 = fadd contract float %v125.3, %v127.3
  store float %v128.3, ptr addrspace(3) %v115, align 4
  br label %bb27.3

bb27.3:                                           ; preds = %bb23.3, %bb27.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v120.not.4 = icmp eq i32 %v34, 0
  br i1 %v120.not.4, label %bb23.4, label %bb27.4

bb23.4:                                           ; preds = %bb27.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v115, i64 4
  %v125.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v127.4 = load float, ptr addrspace(3) %v115, align 4
  %v128.4 = fadd contract float %v125.4, %v127.4
  store float %v128.4, ptr addrspace(3) %v115, align 4
  br label %bb27.4

bb27.4:                                           ; preds = %bb23.4, %bb27.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v131 = icmp eq i32 %v34, 0
  br i1 %v131, label %bb30, label %bb33

bb30:                                             ; preds = %bb27.4
  %v135 = mul nuw i64 %v54, %v41
  %4 = getelementptr float, ptr %v12, i64 %v135
  %v138 = getelementptr float, ptr %4, i64 %v47.zext
  %v134 = load float, ptr addrspace(3) @__shared_mem_7, align 4
  store float %v134, ptr %v138, align 4
  br label %bb33

bb33:                                             ; preds = %bb27.4, %bb30, %entry
  ret void

bb34:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb35:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable

bb36:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb37:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb38:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb39:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_route_topk(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, float %v12, ptr captures(none) %v13, i64 %v14, ptr captures(none) %v15, i64 %v16) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i3 = icmp eq i32 %v4.i1, 1
  %v7.i4 = icmp eq i32 %v6.i2, 1
  %v8.not.not.i = and i1 %v5.i3, %v7.i4
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i5 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i5
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v42 = zext i32 %v6 to i64
  %v43.not = icmp ult i64 %v22.i, %v42
  br i1 %v43.not, label %bb3, label %bb70

bb3:                                              ; preds = %entry
  %v45 = zext i32 %v7 to i64
  %v46 = mul nuw i64 %v22.i, %v45
  %v49 = zext i32 %v8 to i64
  %v50.not48.not = icmp eq i32 %v8, 0
  br i1 %v50.not48.not, label %bb18, label %bb5.lr.ph

bb5.lr.ph:                                        ; preds = %bb3
  %v53.not = icmp ult i64 %v5, %v49
  %v62.not45.not = icmp eq i32 %v7, 0
  %0 = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v46)
  %1 = add nsw i64 %v45, -1
  %umin = tail call i64 @llvm.umin.i64(i64 %0, i64 %1)
  %2 = freeze i64 %umin
  %invariant.gep115 = getelementptr float, ptr %v0, i64 %v46
  %xtraiter = and i64 %v45, 3
  %3 = icmp ult i32 %v7, 4
  %unroll_iter = and i64 %v45, 4294967292
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  %lcmp.mod126 = icmp ne i64 %xtraiter, 0
  br label %bb5

bb5:                                              ; preds = %bb5.lr.ph, %bb14
  %indvars.iv83 = phi i64 [ 0, %bb5.lr.ph ], [ %indvars.iv.next84, %bb14 ]
  %indvars.iv = phi i64 [ 0, %bb5.lr.ph ], [ %indvars.iv.next, %bb14 ]
  %v4850 = phi i64 [ 0, %bb5.lr.ph ], [ %v83, %bb14 ]
  %v4749 = phi float [ 0xC7EFFFFFE0000000, %bb5.lr.ph ], [ %v47.v60, %bb14 ]
  %umax = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv)
  %4 = add i64 %umax, %indvars.iv83
  %umin86 = tail call i64 @llvm.umin.i64(i64 %2, i64 %4)
  br i1 %v53.not, label %bb9, label %bb6

bb6:                                              ; preds = %bb5
  %v57 = getelementptr inbounds nuw float, ptr %v4, i64 %v4850
  %v58 = load float, ptr %v57, align 4
  br label %bb9

bb9:                                              ; preds = %bb5, %bb6
  %v59 = phi float [ %v58, %bb6 ], [ 0.000000e+00, %bb5 ]
  br i1 %v62.not45.not, label %bb14, label %bb11.lr.ph

bb11.lr.ph:                                       ; preds = %bb9
  %v64 = mul nuw i64 %v4850, %v45
  %.not.not = icmp ugt i64 %4, %2
  br i1 %.not.not, label %bb11.lr.ph.split, label %bb75

bb11.lr.ph.split:                                 ; preds = %bb11.lr.ph
  %.not = icmp eq i64 %0, %umin86
  br i1 %.not, label %bb76, label %bb11.lr.ph.split.split

bb11.lr.ph.split.split:                           ; preds = %bb11.lr.ph.split
  %invariant.gep = getelementptr float, ptr %v2, i64 %v64
  br i1 %3, label %bb11.epil.preheader, label %bb11

bb11:                                             ; preds = %bb11.lr.ph.split.split, %bb11
  %v6147 = phi i64 [ %v79.3, %bb11 ], [ 0, %bb11.lr.ph.split.split ]
  %v6046 = phi float [ %v78.3, %bb11 ], [ %v59, %bb11.lr.ph.split.split ]
  %niter = phi i64 [ %niter.next.3, %bb11 ], [ 0, %bb11.lr.ph.split.split ]
  %gep = getelementptr float, ptr %invariant.gep, i64 %v6147
  %v70 = load float, ptr %gep, align 4
  %gep116 = getelementptr float, ptr %invariant.gep115, i64 %v6147
  %v76 = load float, ptr %gep116, align 4
  %v77 = fmul contract float %v70, %v76
  %v78 = fadd contract float %v6046, %v77
  %v79 = or disjoint i64 %v6147, 1
  %gep.1 = getelementptr float, ptr %invariant.gep, i64 %v79
  %v70.1 = load float, ptr %gep.1, align 4
  %gep116.1 = getelementptr float, ptr %invariant.gep115, i64 %v79
  %v76.1 = load float, ptr %gep116.1, align 4
  %v77.1 = fmul contract float %v70.1, %v76.1
  %v78.1 = fadd contract float %v78, %v77.1
  %v79.1 = or disjoint i64 %v6147, 2
  %gep.2 = getelementptr float, ptr %invariant.gep, i64 %v79.1
  %v70.2 = load float, ptr %gep.2, align 4
  %gep116.2 = getelementptr float, ptr %invariant.gep115, i64 %v79.1
  %v76.2 = load float, ptr %gep116.2, align 4
  %v77.2 = fmul contract float %v70.2, %v76.2
  %v78.2 = fadd contract float %v78.1, %v77.2
  %v79.2 = or disjoint i64 %v6147, 3
  %gep.3 = getelementptr float, ptr %invariant.gep, i64 %v79.2
  %v70.3 = load float, ptr %gep.3, align 4
  %gep116.3 = getelementptr float, ptr %invariant.gep115, i64 %v79.2
  %v76.3 = load float, ptr %gep116.3, align 4
  %v77.3 = fmul contract float %v70.3, %v76.3
  %v78.3 = fadd contract float %v78.2, %v77.3
  %v79.3 = add nuw nsw i64 %v6147, 4
  %niter.next.3 = add i64 %niter, 4
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3, label %bb14.loopexit.unr-lcssa, label %bb11

bb14.loopexit.unr-lcssa:                          ; preds = %bb11
  br i1 %lcmp.mod.not, label %bb14, label %bb11.epil.preheader

bb11.epil.preheader:                              ; preds = %bb14.loopexit.unr-lcssa, %bb11.lr.ph.split.split
  %v6147.epil.init = phi i64 [ 0, %bb11.lr.ph.split.split ], [ %v79.3, %bb14.loopexit.unr-lcssa ]
  %v6046.epil.init = phi float [ %v59, %bb11.lr.ph.split.split ], [ %v78.3, %bb14.loopexit.unr-lcssa ]
  tail call void @llvm.assume(i1 %lcmp.mod126)
  br label %bb11.epil

bb11.epil:                                        ; preds = %bb11.epil, %bb11.epil.preheader
  %v6147.epil = phi i64 [ %v6147.epil.init, %bb11.epil.preheader ], [ %v79.epil, %bb11.epil ]
  %v6046.epil = phi float [ %v6046.epil.init, %bb11.epil.preheader ], [ %v78.epil, %bb11.epil ]
  %epil.iter = phi i64 [ 0, %bb11.epil.preheader ], [ %epil.iter.next, %bb11.epil ]
  %gep.epil = getelementptr float, ptr %invariant.gep, i64 %v6147.epil
  %v70.epil = load float, ptr %gep.epil, align 4
  %gep116.epil = getelementptr float, ptr %invariant.gep115, i64 %v6147.epil
  %v76.epil = load float, ptr %gep116.epil, align 4
  %v77.epil = fmul contract float %v70.epil, %v76.epil
  %v78.epil = fadd contract float %v6046.epil, %v77.epil
  %v79.epil = add nuw nsw i64 %v6147.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %bb14, label %bb11.epil, !llvm.loop !10

bb14:                                             ; preds = %bb14.loopexit.unr-lcssa, %bb11.epil, %bb9
  %v60.lcssa = phi float [ %v59, %bb9 ], [ %v78.3, %bb14.loopexit.unr-lcssa ], [ %v78.epil, %bb11.epil ]
  %v80.inv = fcmp ogt float %v60.lcssa, %v4749
  %v47.v60 = select i1 %v80.inv, float %v60.lcssa, float %v4749
  %v83 = add nuw nsw i64 %v4850, 1
  %indvars.iv.next = add nuw i64 %indvars.iv, %v45
  %indvars.iv.next84 = sub i64 %indvars.iv83, %v45
  %exitcond87.not = icmp eq i64 %v83, %v49
  br i1 %exitcond87.not, label %bb18, label %bb5

bb18:                                             ; preds = %bb14, %bb3
  %v47.lcssa = phi float [ 0xC7EFFFFFE0000000, %bb3 ], [ %v47.v60, %bb14 ]
  %v84 = icmp ne i32 %v10, 2
  %v87.not57 = icmp ne i32 %v8, 0
  %or.cond82 = and i1 %v84, %v87.not57
  br i1 %or.cond82, label %bb21.lr.ph, label %bb32.split

bb21.lr.ph:                                       ; preds = %bb18
  %v90.not = icmp ult i64 %v5, %v49
  %v99.not53.not = icmp eq i32 %v7, 0
  %5 = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v46)
  %6 = add nsw i64 %v45, -1
  %umin94 = tail call i64 @llvm.umin.i64(i64 %5, i64 %6)
  %7 = freeze i64 %umin94
  %invariant.gep119 = getelementptr float, ptr %v0, i64 %v46
  %xtraiter127 = and i64 %v45, 3
  %8 = icmp ult i32 %v7, 4
  %unroll_iter132 = and i64 %v45, 4294967292
  %lcmp.mod129.not = icmp eq i64 %xtraiter127, 0
  %lcmp.mod131 = icmp ne i64 %xtraiter127, 0
  br label %bb21

bb21:                                             ; preds = %bb21.lr.ph, %bb30
  %indvars.iv91 = phi i64 [ 0, %bb21.lr.ph ], [ %indvars.iv.next92, %bb30 ]
  %indvars.iv88 = phi i64 [ 0, %bb21.lr.ph ], [ %indvars.iv.next89, %bb30 ]
  %v8659 = phi i64 [ 0, %bb21.lr.ph ], [ %v212, %bb30 ]
  %v8558 = phi float [ 0.000000e+00, %bb21.lr.ph ], [ %v211, %bb30 ]
  %umax90 = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv88)
  %9 = add i64 %umax90, %indvars.iv91
  %umin95 = tail call i64 @llvm.umin.i64(i64 %7, i64 %9)
  br i1 %v90.not, label %bb25, label %bb22

bb22:                                             ; preds = %bb21
  %v94 = getelementptr inbounds nuw float, ptr %v4, i64 %v8659
  %v95 = load float, ptr %v94, align 4
  br label %bb25

bb25:                                             ; preds = %bb21, %bb22
  %v96 = phi float [ %v95, %bb22 ], [ 0.000000e+00, %bb21 ]
  br i1 %v99.not53.not, label %bb30, label %bb27.lr.ph

bb27.lr.ph:                                       ; preds = %bb25
  %v101 = mul nuw i64 %v8659, %v45
  %.not111.not = icmp ugt i64 %9, %7
  br i1 %.not111.not, label %bb27.lr.ph.split, label %bb78

bb27.lr.ph.split:                                 ; preds = %bb27.lr.ph
  %.not112 = icmp eq i64 %5, %umin95
  br i1 %.not112, label %bb79, label %bb27.lr.ph.split.split

bb27.lr.ph.split.split:                           ; preds = %bb27.lr.ph.split
  %invariant.gep117 = getelementptr float, ptr %v2, i64 %v101
  br i1 %8, label %bb27.epil.preheader, label %bb27

bb27:                                             ; preds = %bb27.lr.ph.split.split, %bb27
  %v9855 = phi i64 [ %v116.3, %bb27 ], [ 0, %bb27.lr.ph.split.split ]
  %v9754 = phi float [ %v115.3, %bb27 ], [ %v96, %bb27.lr.ph.split.split ]
  %niter133 = phi i64 [ %niter133.next.3, %bb27 ], [ 0, %bb27.lr.ph.split.split ]
  %gep118 = getelementptr float, ptr %invariant.gep117, i64 %v9855
  %v107 = load float, ptr %gep118, align 4
  %gep120 = getelementptr float, ptr %invariant.gep119, i64 %v9855
  %v113 = load float, ptr %gep120, align 4
  %v114 = fmul contract float %v107, %v113
  %v115 = fadd contract float %v9754, %v114
  %v116 = or disjoint i64 %v9855, 1
  %gep118.1 = getelementptr float, ptr %invariant.gep117, i64 %v116
  %v107.1 = load float, ptr %gep118.1, align 4
  %gep120.1 = getelementptr float, ptr %invariant.gep119, i64 %v116
  %v113.1 = load float, ptr %gep120.1, align 4
  %v114.1 = fmul contract float %v107.1, %v113.1
  %v115.1 = fadd contract float %v115, %v114.1
  %v116.1 = or disjoint i64 %v9855, 2
  %gep118.2 = getelementptr float, ptr %invariant.gep117, i64 %v116.1
  %v107.2 = load float, ptr %gep118.2, align 4
  %gep120.2 = getelementptr float, ptr %invariant.gep119, i64 %v116.1
  %v113.2 = load float, ptr %gep120.2, align 4
  %v114.2 = fmul contract float %v107.2, %v113.2
  %v115.2 = fadd contract float %v115.1, %v114.2
  %v116.2 = or disjoint i64 %v9855, 3
  %gep118.3 = getelementptr float, ptr %invariant.gep117, i64 %v116.2
  %v107.3 = load float, ptr %gep118.3, align 4
  %gep120.3 = getelementptr float, ptr %invariant.gep119, i64 %v116.2
  %v113.3 = load float, ptr %gep120.3, align 4
  %v114.3 = fmul contract float %v107.3, %v113.3
  %v115.3 = fadd contract float %v115.2, %v114.3
  %v116.3 = add nuw nsw i64 %v9855, 4
  %niter133.next.3 = add i64 %niter133, 4
  %niter133.ncmp.3 = icmp eq i64 %niter133.next.3, %unroll_iter132
  br i1 %niter133.ncmp.3, label %bb30.loopexit.unr-lcssa, label %bb27

bb30.loopexit.unr-lcssa:                          ; preds = %bb27
  br i1 %lcmp.mod129.not, label %bb30, label %bb27.epil.preheader

bb27.epil.preheader:                              ; preds = %bb30.loopexit.unr-lcssa, %bb27.lr.ph.split.split
  %v9855.epil.init = phi i64 [ 0, %bb27.lr.ph.split.split ], [ %v116.3, %bb30.loopexit.unr-lcssa ]
  %v9754.epil.init = phi float [ %v96, %bb27.lr.ph.split.split ], [ %v115.3, %bb30.loopexit.unr-lcssa ]
  tail call void @llvm.assume(i1 %lcmp.mod131)
  br label %bb27.epil

bb27.epil:                                        ; preds = %bb27.epil, %bb27.epil.preheader
  %v9855.epil = phi i64 [ %v9855.epil.init, %bb27.epil.preheader ], [ %v116.epil, %bb27.epil ]
  %v9754.epil = phi float [ %v9754.epil.init, %bb27.epil.preheader ], [ %v115.epil, %bb27.epil ]
  %epil.iter128 = phi i64 [ 0, %bb27.epil.preheader ], [ %epil.iter128.next, %bb27.epil ]
  %gep118.epil = getelementptr float, ptr %invariant.gep117, i64 %v9855.epil
  %v107.epil = load float, ptr %gep118.epil, align 4
  %gep120.epil = getelementptr float, ptr %invariant.gep119, i64 %v9855.epil
  %v113.epil = load float, ptr %gep120.epil, align 4
  %v114.epil = fmul contract float %v107.epil, %v113.epil
  %v115.epil = fadd contract float %v9754.epil, %v114.epil
  %v116.epil = add nuw nsw i64 %v9855.epil, 1
  %epil.iter128.next = add i64 %epil.iter128, 1
  %epil.iter128.cmp.not = icmp eq i64 %epil.iter128.next, %xtraiter127
  br i1 %epil.iter128.cmp.not, label %bb30, label %bb27.epil, !llvm.loop !11

bb30:                                             ; preds = %bb30.loopexit.unr-lcssa, %bb27.epil, %bb25
  %v97.lcssa = phi float [ %v96, %bb25 ], [ %v115.3, %bb30.loopexit.unr-lcssa ], [ %v115.epil, %bb27.epil ]
  %v117 = fsub contract float %v97.lcssa, %v47.lcssa
  %10 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %10, 0
  %11 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v117, float 0x3F777313A0000000, float 5.000000e-01) #20
  %12 = tail call float @llvm.fma.f32(float %v117, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %12, float %11
  %13 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %14 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %14, float %13
  %15 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %16 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %16, float %15
  %17 = fadd float %.04.i, 0xC168000FE0000000
  %18 = fneg float %17
  %19 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v117, float 0x3FF7154760000000, float %18) #20
  %20 = tail call float @llvm.fma.f32(float %v117, float 0x3FF7154760000000, float %18)
  %.0.i = select i1 %.not.i, float %20, float %19
  %21 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v117, float 0x3E54AE0C00000000, float %.0.i) #20
  %22 = tail call float @llvm.fma.f32(float %v117, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %22, float %21
  %23 = bitcast float %.04.i to i32
  %24 = shl i32 %23, 23
  %25 = bitcast i32 %24 to float
  %26 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %27 = fmul float %26, %25
  %v211 = fadd contract float %v8558, %27
  %v212 = add nuw nsw i64 %v8659, 1
  %indvars.iv.next89 = add nuw i64 %indvars.iv88, %v45
  %indvars.iv.next92 = sub i64 %indvars.iv91, %v45
  %exitcond97.not = icmp eq i64 %v212, %v49
  br i1 %exitcond97.not, label %bb32.split, label %bb21

bb32.split:                                       ; preds = %bb30, %bb18
  %v119 = phi float [ 0.000000e+00, %bb18 ], [ %v211, %bb30 ]
  %v122 = zext i32 %v9 to i64
  %v123.not75.not = icmp eq i32 %v9, 0
  br i1 %v123.not75.not, label %bb70, label %bb35.preheader.lr.ph

bb35.preheader.lr.ph:                             ; preds = %bb32.split
  %v134 = mul nuw i64 %v22.i, %v122
  %28 = getelementptr i32, ptr %v13, i64 %v134
  %v146.not = icmp ult i64 %v5, %v49
  %v155.not65.not = icmp eq i32 %v7, 0
  %29 = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v46)
  %30 = add nsw i64 %v45, -1
  %umin105 = tail call i64 @llvm.umin.i64(i64 %29, i64 %30)
  %31 = freeze i64 %umin105
  %invariant.gep123 = getelementptr float, ptr %v0, i64 %v46
  %xtraiter141 = and i64 %v45, 3
  %32 = icmp ult i32 %v7, 4
  %unroll_iter146 = and i64 %v45, 4294967292
  %lcmp.mod143.not = icmp eq i64 %xtraiter141, 0
  %lcmp.mod145 = icmp ne i64 %xtraiter141, 0
  br label %bb35.preheader

bb35.preheader:                                   ; preds = %bb35.preheader.lr.ph, %bb60
  %v12177 = phi i64 [ 0, %bb35.preheader.lr.ph ], [ %v195, %bb60 ]
  %v12076 = phi float [ 0.000000e+00, %bb35.preheader.lr.ph ], [ %v194, %bb60 ]
  br i1 %v50.not48.not, label %bb60, label %bb37.preheader.lr.ph

bb37.preheader.lr.ph:                             ; preds = %bb35.preheader
  %v132.not61.not = icmp eq i64 %v12177, 0
  %xtraiter134 = and i64 %v12177, 3
  %33 = icmp samesign ult i64 %v12177, 4
  %unroll_iter139 = and i64 %v12177, 9223372036854775804
  %lcmp.mod136.not = icmp eq i64 %xtraiter134, 0
  %lcmp.mod138 = icmp ne i64 %xtraiter134, 0
  br label %bb37.preheader

bb63.lr.ph:                                       ; preds = %bb60
  %v199 = mul nuw i64 %v22.i, %v122
  %34 = getelementptr float, ptr %v15, i64 %v199
  %v204 = icmp eq i32 %v11, 0
  %v205 = fcmp ule float %v194, 0.000000e+00
  %or.cond = select i1 %v204, i1 true, i1 %v205
  %xtraiter148 = and i64 %v122, 1
  %35 = icmp eq i32 %v9, 1
  br i1 %35, label %bb63.epil.preheader, label %bb63.lr.ph.new

bb63.lr.ph.new:                                   ; preds = %bb63.lr.ph
  %unroll_iter152 = and i64 %v122, 4294967294
  br label %bb63

bb37.preheader:                                   ; preds = %bb37.preheader.lr.ph, %bb59
  %indvars.iv102 = phi i64 [ 0, %bb37.preheader.lr.ph ], [ %indvars.iv.next103, %bb59 ]
  %indvars.iv99 = phi i64 [ 0, %bb37.preheader.lr.ph ], [ %indvars.iv.next100, %bb59 ]
  %v12772 = phi i64 [ 0, %bb37.preheader.lr.ph ], [ %v184, %bb59 ]
  %v12671 = phi float [ 0xC7EFFFFFE0000000, %bb37.preheader.lr.ph ], [ %v183, %bb59 ]
  %v12570 = phi i64 [ 0, %bb37.preheader.lr.ph ], [ %v185, %bb59 ]
  %umax101 = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv99)
  %36 = add i64 %umax101, %indvars.iv102
  %umin106 = tail call i64 @llvm.umin.i64(i64 %31, i64 %36)
  br i1 %v132.not61.not, label %bb43, label %bb38.preheader

bb38.preheader:                                   ; preds = %bb37.preheader
  br i1 %33, label %bb38.epil.preheader, label %bb38

bb38:                                             ; preds = %bb38.preheader, %bb38
  %v13163 = phi i64 [ %v143.3, %bb38 ], [ 0, %bb38.preheader ]
  %v13062 = phi i1 [ %v130..3, %bb38 ], [ false, %bb38.preheader ]
  %niter140 = phi i64 [ %niter140.next.3, %bb38 ], [ 0, %bb38.preheader ]
  %v137 = getelementptr i32, ptr %28, i64 %v13163
  %v138 = load i32, ptr %v137, align 4
  %v139 = zext i32 %v138 to i64
  %v140.not = icmp eq i64 %v12570, %v139
  %37 = getelementptr i32, ptr %28, i64 %v13163
  %v137.1 = getelementptr i8, ptr %37, i64 4
  %v138.1 = load i32, ptr %v137.1, align 4
  %v139.1 = zext i32 %v138.1 to i64
  %v140.not.1 = icmp eq i64 %v12570, %v139.1
  %38 = getelementptr i32, ptr %28, i64 %v13163
  %v137.2 = getelementptr i8, ptr %38, i64 8
  %v138.2 = load i32, ptr %v137.2, align 4
  %v139.2 = zext i32 %v138.2 to i64
  %v140.not.2 = icmp eq i64 %v12570, %v139.2
  %39 = getelementptr i32, ptr %28, i64 %v13163
  %v137.3 = getelementptr i8, ptr %39, i64 12
  %v138.3 = load i32, ptr %v137.3, align 4
  %v139.3 = zext i32 %v138.3 to i64
  %v140.not.3 = icmp eq i64 %v12570, %v139.3
  %40 = select i1 %v140.not.3, i1 true, i1 %v140.not.2
  %41 = select i1 %40, i1 true, i1 %v140.not.1
  %42 = select i1 %41, i1 true, i1 %v140.not
  %v130..3 = select i1 %42, i1 true, i1 %v13062
  %v143.3 = add nuw nsw i64 %v13163, 4
  %niter140.next.3 = add i64 %niter140, 4
  %niter140.ncmp.3 = icmp eq i64 %niter140.next.3, %unroll_iter139
  br i1 %niter140.ncmp.3, label %bb42.unr-lcssa, label %bb38

bb42.unr-lcssa:                                   ; preds = %bb38
  br i1 %lcmp.mod136.not, label %bb42, label %bb38.epil.preheader

bb38.epil.preheader:                              ; preds = %bb42.unr-lcssa, %bb38.preheader
  %v13163.epil.init = phi i64 [ 0, %bb38.preheader ], [ %v143.3, %bb42.unr-lcssa ]
  %v13062.epil.init = phi i1 [ false, %bb38.preheader ], [ %v130..3, %bb42.unr-lcssa ]
  tail call void @llvm.assume(i1 %lcmp.mod138)
  br label %bb38.epil

bb38.epil:                                        ; preds = %bb38.epil, %bb38.epil.preheader
  %v13163.epil = phi i64 [ %v143.epil, %bb38.epil ], [ %v13163.epil.init, %bb38.epil.preheader ]
  %v13062.epil = phi i1 [ %v130..epil, %bb38.epil ], [ %v13062.epil.init, %bb38.epil.preheader ]
  %epil.iter135 = phi i64 [ %epil.iter135.next, %bb38.epil ], [ 0, %bb38.epil.preheader ]
  %v137.epil = getelementptr i32, ptr %28, i64 %v13163.epil
  %v138.epil = load i32, ptr %v137.epil, align 4
  %v139.epil = zext i32 %v138.epil to i64
  %v140.not.epil = icmp eq i64 %v12570, %v139.epil
  %v130..epil = select i1 %v140.not.epil, i1 true, i1 %v13062.epil
  %v143.epil = add nuw nsw i64 %v13163.epil, 1
  %epil.iter135.next = add i64 %epil.iter135, 1
  %epil.iter135.cmp.not = icmp eq i64 %epil.iter135.next, %xtraiter134
  br i1 %epil.iter135.cmp.not, label %bb42, label %bb38.epil, !llvm.loop !12

bb42:                                             ; preds = %bb38.epil, %bb42.unr-lcssa
  %v130..lcssa = phi i1 [ %v130..3, %bb42.unr-lcssa ], [ %v130..epil, %bb38.epil ]
  br i1 %v130..lcssa, label %bb59, label %bb43

bb43:                                             ; preds = %bb37.preheader, %bb42
  br i1 %v146.not, label %bb47, label %bb44

bb44:                                             ; preds = %bb43
  %v150 = getelementptr inbounds nuw float, ptr %v4, i64 %v12570
  %v151 = load float, ptr %v150, align 4
  br label %bb47

bb47:                                             ; preds = %bb43, %bb44
  %v152 = phi float [ %v151, %bb44 ], [ 0.000000e+00, %bb43 ]
  br i1 %v155.not65.not, label %bb52, label %bb49.lr.ph

bb49.lr.ph:                                       ; preds = %bb47
  %v157 = mul nuw i64 %v12570, %v45
  %.not113.not = icmp ugt i64 %36, %31
  br i1 %.not113.not, label %bb49.lr.ph.split, label %bb81

bb49.lr.ph.split:                                 ; preds = %bb49.lr.ph
  %.not114 = icmp eq i64 %29, %umin106
  br i1 %.not114, label %bb82, label %bb49.lr.ph.split.split

bb49.lr.ph.split.split:                           ; preds = %bb49.lr.ph.split
  %invariant.gep121 = getelementptr float, ptr %v2, i64 %v157
  br i1 %32, label %bb49.epil.preheader, label %bb49

bb49:                                             ; preds = %bb49.lr.ph.split.split, %bb49
  %v15467 = phi i64 [ %v172.3, %bb49 ], [ 0, %bb49.lr.ph.split.split ]
  %v15366 = phi float [ %v171.3, %bb49 ], [ %v152, %bb49.lr.ph.split.split ]
  %niter147 = phi i64 [ %niter147.next.3, %bb49 ], [ 0, %bb49.lr.ph.split.split ]
  %gep122 = getelementptr float, ptr %invariant.gep121, i64 %v15467
  %v163 = load float, ptr %gep122, align 4
  %gep124 = getelementptr float, ptr %invariant.gep123, i64 %v15467
  %v169 = load float, ptr %gep124, align 4
  %v170 = fmul contract float %v163, %v169
  %v171 = fadd contract float %v15366, %v170
  %v172 = or disjoint i64 %v15467, 1
  %gep122.1 = getelementptr float, ptr %invariant.gep121, i64 %v172
  %v163.1 = load float, ptr %gep122.1, align 4
  %gep124.1 = getelementptr float, ptr %invariant.gep123, i64 %v172
  %v169.1 = load float, ptr %gep124.1, align 4
  %v170.1 = fmul contract float %v163.1, %v169.1
  %v171.1 = fadd contract float %v171, %v170.1
  %v172.1 = or disjoint i64 %v15467, 2
  %gep122.2 = getelementptr float, ptr %invariant.gep121, i64 %v172.1
  %v163.2 = load float, ptr %gep122.2, align 4
  %gep124.2 = getelementptr float, ptr %invariant.gep123, i64 %v172.1
  %v169.2 = load float, ptr %gep124.2, align 4
  %v170.2 = fmul contract float %v163.2, %v169.2
  %v171.2 = fadd contract float %v171.1, %v170.2
  %v172.2 = or disjoint i64 %v15467, 3
  %gep122.3 = getelementptr float, ptr %invariant.gep121, i64 %v172.2
  %v163.3 = load float, ptr %gep122.3, align 4
  %gep124.3 = getelementptr float, ptr %invariant.gep123, i64 %v172.2
  %v169.3 = load float, ptr %gep124.3, align 4
  %v170.3 = fmul contract float %v163.3, %v169.3
  %v171.3 = fadd contract float %v171.2, %v170.3
  %v172.3 = add nuw nsw i64 %v15467, 4
  %niter147.next.3 = add i64 %niter147, 4
  %niter147.ncmp.3 = icmp eq i64 %niter147.next.3, %unroll_iter146
  br i1 %niter147.ncmp.3, label %bb52.loopexit.unr-lcssa, label %bb49

bb52.loopexit.unr-lcssa:                          ; preds = %bb49
  br i1 %lcmp.mod143.not, label %bb52, label %bb49.epil.preheader

bb49.epil.preheader:                              ; preds = %bb52.loopexit.unr-lcssa, %bb49.lr.ph.split.split
  %v15467.epil.init = phi i64 [ 0, %bb49.lr.ph.split.split ], [ %v172.3, %bb52.loopexit.unr-lcssa ]
  %v15366.epil.init = phi float [ %v152, %bb49.lr.ph.split.split ], [ %v171.3, %bb52.loopexit.unr-lcssa ]
  tail call void @llvm.assume(i1 %lcmp.mod145)
  br label %bb49.epil

bb49.epil:                                        ; preds = %bb49.epil, %bb49.epil.preheader
  %v15467.epil = phi i64 [ %v15467.epil.init, %bb49.epil.preheader ], [ %v172.epil, %bb49.epil ]
  %v15366.epil = phi float [ %v15366.epil.init, %bb49.epil.preheader ], [ %v171.epil, %bb49.epil ]
  %epil.iter142 = phi i64 [ 0, %bb49.epil.preheader ], [ %epil.iter142.next, %bb49.epil ]
  %gep122.epil = getelementptr float, ptr %invariant.gep121, i64 %v15467.epil
  %v163.epil = load float, ptr %gep122.epil, align 4
  %gep124.epil = getelementptr float, ptr %invariant.gep123, i64 %v15467.epil
  %v169.epil = load float, ptr %gep124.epil, align 4
  %v170.epil = fmul contract float %v163.epil, %v169.epil
  %v171.epil = fadd contract float %v15366.epil, %v170.epil
  %v172.epil = add nuw nsw i64 %v15467.epil, 1
  %epil.iter142.next = add i64 %epil.iter142, 1
  %epil.iter142.cmp.not = icmp eq i64 %epil.iter142.next, %xtraiter141
  br i1 %epil.iter142.cmp.not, label %bb52, label %bb49.epil, !llvm.loop !13

bb52:                                             ; preds = %bb52.loopexit.unr-lcssa, %bb49.epil, %bb47
  %v153.lcssa = phi float [ %v152, %bb47 ], [ %v171.3, %bb52.loopexit.unr-lcssa ], [ %v171.epil, %bb49.epil ]
  %43 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i12 = icmp eq i32 %43, 0
  br i1 %v84, label %bb54, label %bb53

bb53:                                             ; preds = %bb52
  %v174 = fneg float %v153.lcssa
  %44 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v174, float 0x3F777313A0000000, float 5.000000e-01) #20
  %45 = tail call float @llvm.fma.f32(float %v174, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i7 = select i1 %.not.i12, float %45, float %44
  %46 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i7) #20
  %47 = tail call float @llvm.nvvm.saturate.f(float %.02.i7) #20
  %.03.i8 = select i1 %.not.i12, float %47, float %46
  %48 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i8, float 2.520000e+02, float 0x4168000020000000) #20
  %49 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i8, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i9 = select i1 %.not.i12, float %49, float %48
  %50 = fadd float %.04.i9, 0xC168000FE0000000
  %51 = fneg float %50
  %52 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v174, float 0x3FF7154760000000, float %51) #20
  %53 = tail call float @llvm.fma.f32(float %v174, float 0x3FF7154760000000, float %51)
  %.0.i10 = select i1 %.not.i12, float %53, float %52
  %54 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v174, float 0x3E54AE0C00000000, float %.0.i10) #20
  %55 = tail call float @llvm.fma.f32(float %v174, float 0x3E54AE0C00000000, float %.0.i10)
  %.01.i11 = select i1 %.not.i12, float %55, float %54
  %56 = bitcast float %.04.i9 to i32
  %57 = shl i32 %56, 23
  %58 = bitcast i32 %57 to float
  %59 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i11)
  %60 = fmul float %59, %58
  %v213 = fadd contract float %60, 1.000000e+00
  %v214 = fdiv contract float 1.000000e+00, %v213
  br label %bb55

bb54:                                             ; preds = %bb52
  %v176 = fsub contract float %v153.lcssa, %v47.lcssa
  %61 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v176, float 0x3F777313A0000000, float 5.000000e-01) #20
  %62 = tail call float @llvm.fma.f32(float %v176, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i13 = select i1 %.not.i12, float %62, float %61
  %63 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i13) #20
  %64 = tail call float @llvm.nvvm.saturate.f(float %.02.i13) #20
  %.03.i14 = select i1 %.not.i12, float %64, float %63
  %65 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i14, float 2.520000e+02, float 0x4168000020000000) #20
  %66 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i14, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i15 = select i1 %.not.i12, float %66, float %65
  %67 = fadd float %.04.i15, 0xC168000FE0000000
  %68 = fneg float %67
  %69 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v176, float 0x3FF7154760000000, float %68) #20
  %70 = tail call float @llvm.fma.f32(float %v176, float 0x3FF7154760000000, float %68)
  %.0.i16 = select i1 %.not.i12, float %70, float %69
  %71 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v176, float 0x3E54AE0C00000000, float %.0.i16) #20
  %72 = tail call float @llvm.fma.f32(float %v176, float 0x3E54AE0C00000000, float %.0.i16)
  %.01.i17 = select i1 %.not.i12, float %72, float %71
  %73 = bitcast float %.04.i15 to i32
  %74 = shl i32 %73, 23
  %75 = bitcast i32 %74 to float
  %76 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i17)
  %77 = fmul float %76, %75
  %v215 = fdiv contract float %77, %v119
  br label %bb55

bb55:                                             ; preds = %bb54, %bb53
  %v178 = phi float [ %v214, %bb53 ], [ %v215, %bb54 ]
  %v179 = fcmp ule float %v178, %v12671
  %v126.v178 = select i1 %v179, float %v12671, float %v178
  %v127.v125 = select i1 %v179, i64 %v12772, i64 %v12570
  br label %bb59

bb59:                                             ; preds = %bb55, %bb42
  %v183 = phi float [ %v12671, %bb42 ], [ %v126.v178, %bb55 ]
  %v184 = phi i64 [ %v12772, %bb42 ], [ %v127.v125, %bb55 ]
  %v185 = add nuw nsw i64 %v12570, 1
  %indvars.iv.next100 = add nuw i64 %indvars.iv99, %v45
  %indvars.iv.next103 = sub i64 %indvars.iv102, %v45
  %exitcond108.not = icmp eq i64 %v185, %v49
  br i1 %exitcond108.not, label %bb60.loopexit, label %bb37.preheader

bb60.loopexit:                                    ; preds = %bb59
  %78 = trunc i64 %v184 to i32
  br label %bb60

bb60:                                             ; preds = %bb60.loopexit, %bb35.preheader
  %v126.lcssa = phi float [ 0xC7EFFFFFE0000000, %bb35.preheader ], [ %v183, %bb60.loopexit ]
  %v127.lcssa = phi i32 [ 0, %bb35.preheader ], [ %78, %bb60.loopexit ]
  %v187 = add i64 %v12177, %v134
  %v189 = getelementptr inbounds i32, ptr %v13, i64 %v187
  store i32 %v127.lcssa, ptr %v189, align 4
  %v193 = getelementptr inbounds float, ptr %v15, i64 %v187
  store float %v126.lcssa, ptr %v193, align 4
  %v194 = fadd contract float %v12076, %v126.lcssa
  %v195 = add nuw nsw i64 %v12177, 1
  %exitcond109.not = icmp eq i64 %v195, %v122
  br i1 %exitcond109.not, label %bb63.lr.ph, label %bb35.preheader

bb63:                                             ; preds = %bb63, %bb63.lr.ph.new
  %v19681 = phi i64 [ 0, %bb63.lr.ph.new ], [ %v210.1, %bb63 ]
  %niter153 = phi i64 [ 0, %bb63.lr.ph.new ], [ %niter153.next.1, %bb63 ]
  %v202 = getelementptr float, ptr %34, i64 %v19681
  %v203 = load float, ptr %v202, align 4
  %v207 = fdiv contract float %v203, %v194
  %v208 = select i1 %or.cond, float %v203, float %v207
  %v209 = fmul contract float %v12, %v208
  store float %v209, ptr %v202, align 4
  %79 = getelementptr float, ptr %34, i64 %v19681
  %v202.1 = getelementptr i8, ptr %79, i64 4
  %v203.1 = load float, ptr %v202.1, align 4
  %v207.1 = fdiv contract float %v203.1, %v194
  %v208.1 = select i1 %or.cond, float %v203.1, float %v207.1
  %v209.1 = fmul contract float %v12, %v208.1
  store float %v209.1, ptr %v202.1, align 4
  %v210.1 = add nuw nsw i64 %v19681, 2
  %niter153.next.1 = add i64 %niter153, 2
  %niter153.ncmp.1 = icmp eq i64 %niter153.next.1, %unroll_iter152
  br i1 %niter153.ncmp.1, label %bb70.loopexit.unr-lcssa, label %bb63

bb70.loopexit.unr-lcssa:                          ; preds = %bb63
  %lcmp.mod150.not = icmp eq i64 %xtraiter148, 0
  br i1 %lcmp.mod150.not, label %bb70, label %bb63.epil.preheader

bb63.epil.preheader:                              ; preds = %bb70.loopexit.unr-lcssa, %bb63.lr.ph
  %v19681.epil.init = phi i64 [ 0, %bb63.lr.ph ], [ %v210.1, %bb70.loopexit.unr-lcssa ]
  %lcmp.mod151 = icmp ne i64 %xtraiter148, 0
  tail call void @llvm.assume(i1 %lcmp.mod151)
  %v202.epil = getelementptr float, ptr %34, i64 %v19681.epil.init
  %v203.epil = load float, ptr %v202.epil, align 4
  %v207.epil = fdiv contract float %v203.epil, %v194
  %v208.epil = select i1 %or.cond, float %v203.epil, float %v207.epil
  %v209.epil = fmul contract float %v12, %v208.epil
  store float %v209.epil, ptr %v202.epil, align 4
  br label %bb70

bb70:                                             ; preds = %bb63.epil.preheader, %bb70.loopexit.unr-lcssa, %bb32.split, %entry
  ret void

bb75:                                             ; preds = %bb11.lr.ph
  tail call void @llvm.trap() #19
  unreachable

bb76:                                             ; preds = %bb11.lr.ph.split
  tail call void @llvm.trap() #19
  unreachable

bb78:                                             ; preds = %bb27.lr.ph
  tail call void @llvm.trap() #19
  unreachable

bb79:                                             ; preds = %bb27.lr.ph.split
  tail call void @llvm.trap() #19
  unreachable

bb81:                                             ; preds = %bb49.lr.ph
  tail call void @llvm.trap() #19
  unreachable

bb82:                                             ; preds = %bb49.lr.ph.split
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_scatter_assignments(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr captures(none) %v4, i64 %v5, ptr writeonly captures(none) %v6, i64 %v7) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i3 = icmp eq i32 %v4.i1, 1
  %v7.i4 = icmp eq i32 %v6.i2, 1
  %v8.not.not.i = and i1 %v5.i3, %v7.i4
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i5 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i5
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v24.not = icmp ult i64 %v22.i, %v1
  br i1 %v24.not, label %bb4, label %bb8

bb4:                                              ; preds = %entry
  %v28 = getelementptr inbounds i32, ptr %v0, i64 %v22.i
  %v29 = load i32, ptr %v28, align 4
  %v30 = zext i32 %v29 to i64
  %v32 = icmp ugt i64 %v5, %v30
  br i1 %v32, label %bb5, label %bb10

bb5:                                              ; preds = %bb4
  %v34 = getelementptr inbounds nuw { { i32 } }, ptr %v4, i64 %v30
  %v35 = atomicrmw add ptr %v34, i32 1 syncscope("device") monotonic, align 4
  %v38 = icmp ugt i64 %v3, %v30
  br i1 %v38, label %bb7, label %bb11

bb7:                                              ; preds = %bb5
  %v36 = zext i32 %v35 to i64
  %v40 = getelementptr inbounds nuw i32, ptr %v2, i64 %v30
  %v41 = load i32, ptr %v40, align 4
  %v42 = zext i32 %v41 to i64
  %0 = getelementptr inbounds nuw i32, ptr %v6, i64 %v42
  %v45 = getelementptr inbounds nuw i32, ptr %0, i64 %v36
  %v46 = trunc i64 %v22.i to i32
  store i32 %v46, ptr %v45, align 4
  br label %bb8

bb8:                                              ; preds = %entry, %bb7
  ret void

bb10:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb11:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @moe_weighted_reduce(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(address_is_null) %v7, i64 %v8) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i17 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i18 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i19 = icmp eq i32 %v4.i17, 1
  %v7.i20 = icmp eq i32 %v6.i18, 1
  %v8.not.not.i21 = and i1 %v5.i19, %v7.i20
  %v13.i22 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i23 = icmp eq i32 %v13.i22, 1
  %v15.i24 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i25 = icmp eq i32 %v15.i24, 1
  %v17.i26 = and i1 %v14.i23, %v16.i25
  %.v18.i27 = and i1 %v8.not.not.i21, %v17.i26
  %v22.i = select i1 %.v18.i27, i64 %v18.i, i64 -1
  %v24 = zext i32 %v4 to i64
  %v25 = zext i32 %v6 to i64
  %v26 = mul nuw i64 %v25, %v24
  %v27.not = icmp ult i64 %v22.i, %v26
  br i1 %v27.not, label %bb3, label %bb14

bb3:                                              ; preds = %entry
  %v29.not = icmp eq i32 %v6, 0
  br i1 %v29.not, label %bb22, label %bb4

bb4:                                              ; preds = %bb3
  %v25.frozen = freeze i64 %v25
  %v31 = udiv i64 %v22.i, %v25.frozen
  %0 = mul i64 %v31, %v25.frozen
  %v32.decomposed = sub i64 %v22.i, %0
  %v35 = zext i32 %v5 to i64
  %v36.not30.not = icmp eq i32 %v5, 0
  br i1 %v36.not30.not, label %bb9, label %bb6.lr.ph

bb6.lr.ph:                                        ; preds = %bb4
  %v38 = mul i64 %v31, %v35
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb8
  %v3432 = phi i64 [ 0, %bb6.lr.ph ], [ %v54, %bb8 ]
  %v3331 = phi float [ 0.000000e+00, %bb6.lr.ph ], [ %v53, %bb8 ]
  %v39 = add nuw i64 %v3432, %v38
  %v41 = icmp ult i64 %v39, %v3
  br i1 %v41, label %bb7, label %bb23

bb7:                                              ; preds = %bb6
  %v45 = mul i64 %v39, %v25
  %v46 = add i64 %v45, %v32.decomposed
  %v48 = icmp ult i64 %v46, %v1
  br i1 %v48, label %bb8, label %bb24

bb8:                                              ; preds = %bb7
  %v43 = getelementptr inbounds float, ptr %v2, i64 %v39
  %v44 = load float, ptr %v43, align 4
  %v50 = getelementptr inbounds float, ptr %v0, i64 %v46
  %v51 = load float, ptr %v50, align 4
  %v52 = fmul contract float %v44, %v51
  %v53 = fadd contract float %v3331, %v52
  %v54 = add nuw nsw i64 %v3432, 1
  %exitcond.not = icmp eq i64 %v54, %v35
  br i1 %exitcond.not, label %bb9, label %bb6

bb9:                                              ; preds = %bb8, %bb4
  %v33.lcssa = phi float [ 0.000000e+00, %bb4 ], [ %v53, %bb8 ]
  %v2.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i5 = zext nneg i32 %v2.i2 to i64
  %v6.i6 = zext nneg i32 %v3.i3 to i64
  %v17.i7 = mul nuw nsw i64 %v5.i5, %v6.i6
  %v7.i8 = zext nneg i32 %v4.i4 to i64
  %v18.i9 = add nuw nsw i64 %v17.i7, %v7.i8
  %v4.i12 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i13 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i14 = icmp eq i32 %v4.i12, 1
  %v7.i15 = icmp eq i32 %v6.i13, 1
  %v8.not.not.i = and i1 %v5.i14, %v7.i15
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i16 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i16
  %v22.i11 = select i1 %.v18.i, i64 %v18.i9, i64 -1
  %v60 = icmp ult i64 %v22.i11, %v8
  %or.cond.not = select i1 %.v18.i, i1 %v60, i1 false
  %v741 = icmp ne ptr %v7, null
  %v74 = select i1 %or.cond.not, i1 %v741, i1 false
  br i1 %v74, label %bb11, label %bb14

bb11:                                             ; preds = %bb9
  %v63 = getelementptr inbounds float, ptr %v7, i64 %v22.i11
  store float %v33.lcssa, ptr %v63, align 4
  br label %bb14

bb14:                                             ; preds = %bb9, %bb11, %entry
  ret void

bb22:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb23:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb24:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @mul_f32(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr writeonly captures(address_is_null) %v4, i64 %v5) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v32 = icmp ult i64 %v22.i, %v5
  %or.cond.not = select i1 %.v18.i, i1 %v32, i1 false
  %v35 = getelementptr inbounds float, ptr %v4, i64 %v22.i
  %v461 = icmp ne ptr %v4, null
  %v46 = select i1 %or.cond.not, i1 %v461, i1 false
  br i1 %v46, label %bb2, label %bb6

bb2:                                              ; preds = %entry
  %v21 = icmp ult i64 %v22.i, %v1
  br i1 %v21, label %bb3, label %bb14

bb3:                                              ; preds = %bb2
  %v26 = icmp ult i64 %v22.i, %v3
  br i1 %v26, label %bb4, label %bb15

bb4:                                              ; preds = %bb3
  %v23 = getelementptr inbounds float, ptr %v0, i64 %v22.i
  %v24 = load float, ptr %v23, align 4
  %v28 = getelementptr inbounds float, ptr %v2, i64 %v22.i
  %v29 = load float, ptr %v28, align 4
  %v30 = fmul contract float %v24, %v29
  store float %v30, ptr %v35, align 4
  br label %bb6

bb6:                                              ; preds = %entry, %bb4
  ret void

bb14:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable

bb15:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q4k_gate_up_swiglu_multiwarp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, ptr writeonly captures(none) %v10, i64 %v11) #6 {
entry:
  %v31 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v32 = zext nneg i32 %v31 to i64
  %v33 = and i64 %v32, 31
  %v36 = lshr i64 %v32, 5
  %v37 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v38 = zext nneg i32 %v37 to i64
  %v40.not = icmp ult i32 %v37, %v8
  %v42 = icmp samesign ult i32 %v31, 128
  %or.cond = select i1 %v40.not, i1 %v42, i1 false
  br i1 %or.cond, label %bb6, label %bb38

bb6:                                              ; preds = %entry
  %v44 = zext i32 %v9 to i64
  %v49.not155 = icmp samesign ult i64 %v36, %v44
  br i1 %v49.not155, label %bb8.lr.ph, label %bb15.preheader

bb8.lr.ph:                                        ; preds = %bb6
  %v51 = mul nuw nsw i64 %v44, %v38
  %v60 = shl nuw nsw i64 %v33, 2
  %v11.i.i = and i64 %v60, 28
  %0 = trunc nuw nsw i64 %v60 to i32
  %1 = lshr i32 %0, 3
  %2 = and i32 %1, 4
  br label %bb8

bb15.preheader:                                   ; preds = %bb13, %bb6
  %v46.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v103, %bb13 ]
  %v47.lcssa = phi float [ 0.000000e+00, %bb6 ], [ %v107, %bb13 ]
  %v115 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v46.lcssa, i32 16, i32 31) #19
  %v142 = fadd contract float %v46.lcssa, %v115
  %v143 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v47.lcssa, i32 16, i32 31) #19
  %v144 = fadd contract float %v47.lcssa, %v143
  %v115.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v142, i32 8, i32 31) #19
  %v142.1 = fadd contract float %v142, %v115.1
  %v143.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v144, i32 8, i32 31) #19
  %v144.1 = fadd contract float %v144, %v143.1
  %v115.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v142.1, i32 4, i32 31) #19
  %v142.2 = fadd contract float %v142.1, %v115.2
  %v143.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v144.1, i32 4, i32 31) #19
  %v144.2 = fadd contract float %v144.1, %v143.2
  %v115.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v142.2, i32 2, i32 31) #19
  %v142.3 = fadd contract float %v142.2, %v115.3
  %v143.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v144.2, i32 2, i32 31) #19
  %v144.3 = fadd contract float %v144.2, %v143.3
  %v115.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v142.3, i32 1, i32 31) #19
  %v143.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v144.3, i32 1, i32 31) #19
  %v116.not = icmp eq i64 %v33, 0
  br i1 %v116.not, label %bb18, label %bb21

bb8:                                              ; preds = %bb8.lr.ph, %bb13
  %v48158 = phi i64 [ %v36, %bb8.lr.ph ], [ %v109, %bb13 ]
  %v47157 = phi float [ 0.000000e+00, %bb8.lr.ph ], [ %v107, %bb13 ]
  %v46156 = phi float [ 0.000000e+00, %bb8.lr.ph ], [ %v103, %bb13 ]
  %reass.add = add nuw i64 %v48158, %v51
  %reass.mul = mul i64 %reass.add, 144
  %v62 = shl i64 %v48158, 8
  %3 = getelementptr i8, ptr %v0, i64 %reass.mul
  %4 = getelementptr i8, ptr %3, i64 16
  %invariant.gep = getelementptr i8, ptr %4, i64 %v11.i.i
  %v27.i = load i8, ptr %3, align 1
  %v31.i = getelementptr i8, ptr %3, i64 1
  %v32.i = load i8, ptr %v31.i, align 1
  %v36.sroa.2.0.insert.ext.i = zext i8 %v32.i to i16
  %v36.sroa.2.0.insert.shift.i = shl nuw i16 %v36.sroa.2.0.insert.ext.i, 8
  %v36.sroa.0.0.insert.ext.i = zext i8 %v27.i to i16
  %v4.i.i = lshr i16 %v36.sroa.2.0.insert.ext.i, 7
  %v6.i.i = zext nneg i16 %v4.i.i to i32
  %v9.i.i = lshr i16 %v36.sroa.2.0.insert.ext.i, 2
  %v10.i.i = and i16 %v9.i.i, 31
  %v36.sroa.2.0.insert.shift.masked.i = and i16 %v36.sroa.2.0.insert.shift.i, 768
  %v12.i.i = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i, %v36.sroa.0.0.insert.ext.i
  %v13.i.i = zext nneg i16 %v12.i.i to i32
  %v42.i = getelementptr i8, ptr %3, i64 2
  %v43.i = load i8, ptr %v42.i, align 1
  %v47.i = getelementptr i8, ptr %3, i64 3
  %v48.i = load i8, ptr %v47.i, align 1
  %v52.sroa.2.0.insert.ext.i = zext i8 %v48.i to i16
  %v52.sroa.2.0.insert.shift.i = shl nuw i16 %v52.sroa.2.0.insert.ext.i, 8
  %v52.sroa.0.0.insert.ext.i = zext i8 %v43.i to i16
  %v4.i5.i = lshr i16 %v52.sroa.2.0.insert.ext.i, 7
  %v6.i6.i = zext nneg i16 %v4.i5.i to i32
  %v9.i7.i = lshr i16 %v52.sroa.2.0.insert.ext.i, 2
  %v10.i8.i = and i16 %v9.i7.i, 31
  %v52.sroa.2.0.insert.shift.masked.i = and i16 %v52.sroa.2.0.insert.shift.i, 768
  %v12.i9.i = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i, %v52.sroa.0.0.insert.ext.i
  %v13.i10.i = zext nneg i16 %v12.i9.i to i32
  %5 = getelementptr i8, ptr %v2, i64 %reass.mul
  %6 = getelementptr i8, ptr %5, i64 16
  %invariant.gep150 = getelementptr i8, ptr %6, i64 %v11.i.i
  %v27.i17 = load i8, ptr %5, align 1
  %v31.i18 = getelementptr i8, ptr %5, i64 1
  %v32.i19 = load i8, ptr %v31.i18, align 1
  %v36.sroa.2.0.insert.ext.i20 = zext i8 %v32.i19 to i16
  %v36.sroa.2.0.insert.shift.i21 = shl nuw i16 %v36.sroa.2.0.insert.ext.i20, 8
  %v36.sroa.0.0.insert.ext.i22 = zext i8 %v27.i17 to i16
  %v4.i.i23 = lshr i16 %v36.sroa.2.0.insert.ext.i20, 7
  %v6.i.i24 = zext nneg i16 %v4.i.i23 to i32
  %v9.i.i25 = lshr i16 %v36.sroa.2.0.insert.ext.i20, 2
  %v10.i.i26 = and i16 %v9.i.i25, 31
  %v36.sroa.2.0.insert.shift.masked.i27 = and i16 %v36.sroa.2.0.insert.shift.i21, 768
  %v12.i.i28 = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i27, %v36.sroa.0.0.insert.ext.i22
  %v13.i.i29 = zext nneg i16 %v12.i.i28 to i32
  %v42.i37 = getelementptr i8, ptr %5, i64 2
  %v43.i38 = load i8, ptr %v42.i37, align 1
  %v47.i39 = getelementptr i8, ptr %5, i64 3
  %v48.i40 = load i8, ptr %v47.i39, align 1
  %v52.sroa.2.0.insert.ext.i41 = zext i8 %v48.i40 to i16
  %v52.sroa.2.0.insert.shift.i42 = shl nuw i16 %v52.sroa.2.0.insert.ext.i41, 8
  %v52.sroa.0.0.insert.ext.i43 = zext i8 %v43.i38 to i16
  %v4.i5.i44 = lshr i16 %v52.sroa.2.0.insert.ext.i41, 7
  %v6.i6.i45 = zext nneg i16 %v4.i5.i44 to i32
  %v9.i7.i46 = lshr i16 %v52.sroa.2.0.insert.ext.i41, 2
  %v10.i8.i47 = and i16 %v9.i7.i46, 31
  %v52.sroa.2.0.insert.shift.masked.i48 = and i16 %v52.sroa.2.0.insert.shift.i42, 768
  %v12.i9.i49 = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i48, %v52.sroa.0.0.insert.ext.i43
  %v13.i10.i50 = zext nneg i16 %v12.i9.i49 to i32
  %v38.i.i = shl nuw i32 %v6.i.i, 31
  %v41.i.i = shl nuw nsw i32 %v13.i.i, 13
  %v39.i.i = or disjoint i32 %v41.i.i, %v38.i.i
  %v42.i.i = or disjoint i32 %v39.i.i, 2139095040
  %v15.i.i = icmp eq i16 %v12.i.i, 0
  %v13.masked.numleadingzeros.i.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i, i1 true)
  %v13.masked.leadingonepos.i.i = xor i32 %v13.masked.numleadingzeros.i.i, 31
  %bb5.tripcount.i.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i
  %v23.i.i = shl nuw nsw i32 %v13.i.i, %bb5.tripcount.i.i
  %reass.sub.i = or disjoint i32 %v38.i.i, 1124073472
  %7 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i, 23
  %v31.i.i = sub nuw nsw i32 %reass.sub.i, %7
  %v25.i.i = shl i32 %v23.i.i, 13
  %v33.i2.i = and i32 %v25.i.i, 8380416
  %v34.i3.i = or disjoint i32 %v31.i.i, %v33.i2.i
  %8 = add nuw nsw i16 %v10.i.i, 112
  %v46.i4.i = zext nneg i16 %8 to i32
  %v48.i.i = shl nuw nsw i32 %v46.i4.i, 23
  %v49.i.i = or disjoint i32 %v48.i.i, %v38.i.i
  %v52.i.i = or disjoint i32 %v49.i.i, %v41.i.i
  %v38.i12.i = shl nuw i32 %v6.i6.i, 31
  %v41.i13.i = shl nuw nsw i32 %v13.i10.i, 13
  %v39.i14.i = or disjoint i32 %v41.i13.i, %v38.i12.i
  %v42.i15.i = or disjoint i32 %v39.i14.i, 2139095040
  %v15.i19.i = icmp eq i16 %v12.i9.i, 0
  %v13.masked.numleadingzeros.i21.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i, i1 true)
  %v13.masked.leadingonepos.i22.i = xor i32 %v13.masked.numleadingzeros.i21.i, 31
  %bb5.tripcount.i23.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i
  %v23.i24.i = shl nuw nsw i32 %v13.i10.i, %bb5.tripcount.i23.i
  %reass.sub63.i = or disjoint i32 %v38.i12.i, 1124073472
  %9 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i, 23
  %v31.i27.i = sub nuw nsw i32 %reass.sub63.i, %9
  %v25.i28.i = shl i32 %v23.i24.i, 13
  %v33.i29.i = and i32 %v25.i28.i, 8380416
  %v34.i30.i = or disjoint i32 %v31.i27.i, %v33.i29.i
  %10 = add nuw nsw i16 %v10.i8.i, 112
  %v46.i35.i = zext nneg i16 %10 to i32
  %v48.i36.i = shl nuw nsw i32 %v46.i35.i, 23
  %v49.i37.i = or disjoint i32 %v48.i36.i, %v38.i12.i
  %v52.i39.i = or disjoint i32 %v49.i37.i, %v41.i13.i
  %11 = getelementptr i8, ptr %3, i64 8
  %12 = getelementptr i8, ptr %3, i64 4
  %13 = getelementptr i8, ptr %3, i64 12
  %v38.i.i31 = shl nuw i32 %v6.i.i24, 31
  %v41.i.i32 = shl nuw nsw i32 %v13.i.i29, 13
  %v39.i.i33 = or disjoint i32 %v41.i.i32, %v38.i.i31
  %v42.i.i34 = or disjoint i32 %v39.i.i33, 2139095040
  %v15.i.i125 = icmp eq i16 %v12.i.i28, 0
  %v13.masked.numleadingzeros.i.i127 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i29, i1 true)
  %v13.masked.leadingonepos.i.i128 = xor i32 %v13.masked.numleadingzeros.i.i127, 31
  %bb5.tripcount.i.i129 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i128
  %v23.i.i130 = shl nuw nsw i32 %v13.i.i29, %bb5.tripcount.i.i129
  %reass.sub.i132 = or disjoint i32 %v38.i.i31, 1124073472
  %14 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i127, 23
  %v31.i.i133 = sub nuw nsw i32 %reass.sub.i132, %14
  %v25.i.i134 = shl i32 %v23.i.i130, 13
  %v33.i2.i135 = and i32 %v25.i.i134, 8380416
  %v34.i3.i136 = or disjoint i32 %v31.i.i133, %v33.i2.i135
  %15 = add nuw nsw i16 %v10.i.i26, 112
  %v46.i4.i141 = zext nneg i16 %15 to i32
  %v48.i.i142 = shl nuw nsw i32 %v46.i4.i141, 23
  %v49.i.i143 = or disjoint i32 %v48.i.i142, %v38.i.i31
  %v52.i.i145 = or disjoint i32 %v49.i.i143, %v41.i.i32
  %v38.i12.i52 = shl nuw i32 %v6.i6.i45, 31
  %v41.i13.i53 = shl nuw nsw i32 %v13.i10.i50, 13
  %v39.i14.i54 = or disjoint i32 %v41.i13.i53, %v38.i12.i52
  %v42.i15.i55 = or disjoint i32 %v39.i14.i54, 2139095040
  %v15.i19.i103 = icmp eq i16 %v12.i9.i49, 0
  %v13.masked.numleadingzeros.i21.i105 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i50, i1 true)
  %v13.masked.leadingonepos.i22.i106 = xor i32 %v13.masked.numleadingzeros.i21.i105, 31
  %bb5.tripcount.i23.i107 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i106
  %v23.i24.i108 = shl nuw nsw i32 %v13.i10.i50, %bb5.tripcount.i23.i107
  %reass.sub63.i110 = or disjoint i32 %v38.i12.i52, 1124073472
  %16 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i105, 23
  %v31.i27.i111 = sub nuw nsw i32 %reass.sub63.i110, %16
  %v25.i28.i112 = shl i32 %v23.i24.i108, 13
  %v33.i29.i113 = and i32 %v25.i28.i112, 8380416
  %v34.i30.i114 = or disjoint i32 %v31.i27.i111, %v33.i29.i113
  %17 = add nuw nsw i16 %v10.i8.i47, 112
  %v46.i35.i119 = zext nneg i16 %17 to i32
  %v48.i36.i120 = shl nuw nsw i32 %v46.i35.i119, 23
  %v49.i37.i121 = or disjoint i32 %v48.i36.i120, %v38.i12.i52
  %v52.i39.i123 = or disjoint i32 %v49.i37.i121, %v41.i13.i53
  %18 = getelementptr i8, ptr %5, i64 8
  %19 = getelementptr i8, ptr %5, i64 4
  %20 = getelementptr i8, ptr %5, i64 12
  %v17.i.i.v34.i3.i = select i1 %v15.i.i, i32 %v38.i.i, i32 %v34.i3.i
  %v17.i32.i.v34.i30.i = select i1 %v15.i19.i, i32 %v38.i12.i, i32 %v34.i30.i
  %v17.i.i138.v34.i3.i136 = select i1 %v15.i.i125, i32 %v38.i.i31, i32 %v34.i3.i136
  %v17.i32.i116.v34.i30.i114 = select i1 %v15.i19.i103, i32 %v38.i12.i52, i32 %v34.i30.i114
  br label %bb10

bb10:                                             ; preds = %bb8, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146
  %v9.i41.i.not = phi i1 [ true, %bb8 ], [ false, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146 ]
  %v56154 = phi i64 [ 0, %bb8 ], [ 128, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146 ]
  %v55153 = phi float [ %v47157, %bb8 ], [ %v107, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146 ]
  %v54152 = phi float [ %v46156, %bb8 ], [ %v103, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146 ]
  %v61 = or disjoint i64 %v56154, %v60
  %v63 = or disjoint i64 %v61, %v62
  %v65 = getelementptr inbounds i8, ptr %v4, i64 %v63
  %v29.sroa.0.0.copyload = load i32, ptr %v65, align 1
  %sext = shl i32 %v29.sroa.0.0.copyload, 24
  %v87 = ashr exact i32 %sext, 24
  %21 = shl i32 %v29.sroa.0.0.copyload, 16
  %v88 = ashr i32 %21, 24
  %22 = shl i32 %v29.sroa.0.0.copyload, 8
  %v90 = ashr i32 %22, 24
  %v92 = ashr i32 %v29.sroa.0.0.copyload, 24
  %v89 = add nsw i32 %v88, %v92
  %v91 = add nsw i32 %v89, %v87
  %v93 = add nsw i32 %v91, %v90
  %v944 = lshr i64 %v63, 5
  %v98 = getelementptr inbounds nuw float, ptr %v6, i64 %v944
  %v99 = load float, ptr %v98, align 4
  %23 = lshr exact i64 %v61, 1
  %v14.i.i = and i64 %23, 96
  %gep = getelementptr i8, ptr %invariant.gep, i64 %v14.i.i
  %v9.sroa.0.0.copyload.i.i = load i32, ptr %gep, align 1
  %v32.v.i.i = lshr i32 %v9.sroa.0.0.copyload.i.i, %2
  %v32.i.i = and i32 %v32.v.i.i, 252645135
  %v33.i.i = xor i32 %v32.i.i, 134744072
  %v34.i.i = and i32 %v33.i.i, 134744072
  %24 = mul nuw i32 %v34.i.i, 30
  %v46.i.i = add nuw nsw i32 %24, %v33.i.i
  %v20.i = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i, i32 %v29.sroa.0.0.copyload, i32 0) #19
  switch i16 %v10.i.i, label %bb10.i.i [
    i16 0, label %bb1.i.i
    i16 31, label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  ]

bb1.i.i:                                          ; preds = %bb10
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

bb10.i.i:                                         ; preds = %bb10
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

cuda_kernels__oxide_kernels__f16_to_f32.exit.i:   ; preds = %bb10, %bb1.i.i, %bb10.i.i
  %v54.i.i = phi i32 [ %v52.i.i, %bb10.i.i ], [ %v17.i.i.v34.i3.i, %bb1.i.i ], [ %v42.i.i, %bb10 ]
  switch i16 %v10.i8.i, label %bb10.i33.i [
    i16 0, label %bb1.i18.i
    i16 31, label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i
  ]

bb1.i18.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i

bb10.i33.i:                                       ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i

cuda_kernels__oxide_kernels__f16_to_f32.exit40.i: ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i, %bb1.i18.i, %bb10.i33.i
  %v54.i16.i = phi i32 [ %v52.i39.i, %bb10.i33.i ], [ %v17.i32.i.v34.i30.i, %bb1.i18.i ], [ %v42.i15.i, %cuda_kernels__oxide_kernels__f16_to_f32.exit.i ]
  %v551.i = lshr i64 %v61, 5
  br i1 %v9.i41.i.not, label %bb1.i42.i, label %bb2.i46.i

bb1.i42.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i
  %v16.i.i = getelementptr i8, ptr %12, i64 %v551.i
  %v17.i43.i = load i8, ptr %v16.i.i, align 1
  %v18.i44.i = and i8 %v17.i43.i, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i

bb2.i46.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i
  %v25.i47.i = getelementptr i8, ptr %11, i64 %v551.i
  %v26.i.i = load i8, ptr %v25.i47.i, align 1
  %v27.i48.i = and i8 %v26.i.i, 15
  %v32.i49.i = getelementptr i8, ptr %3, i64 %v551.i
  %v33.i50.i = load i8, ptr %v32.i49.i, align 1
  %25 = lshr i8 %v33.i50.i, 2
  %v39.i51.i = and i8 %25, 48
  %v40.i.i = or disjoint i8 %v39.i51.i, %v27.i48.i
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i

cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i: ; preds = %bb2.i46.i, %bb1.i42.i
  %v41.i45.i = phi i8 [ %v18.i44.i, %bb1.i42.i ], [ %v40.i.i, %bb2.i46.i ]
  br i1 %v9.i41.i.not, label %bb1.i53.i, label %bb2.i57.i

bb1.i53.i:                                        ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i
  %v16.i54.i = getelementptr i8, ptr %11, i64 %v551.i
  %v17.i55.i = load i8, ptr %v16.i54.i, align 1
  %v18.i56.i = and i8 %v17.i55.i, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit

bb2.i57.i:                                        ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i
  %v19.i.i = add nsw i64 %v551.i, -4
  %v25.i58.i = getelementptr i8, ptr %13, i64 %v19.i.i
  %v26.i59.i = load i8, ptr %v25.i58.i, align 1
  %v29.i.i = lshr i8 %v26.i59.i, 4
  %v34.i60.i = getelementptr i8, ptr %11, i64 %v19.i.i
  %v35.i.i = load i8, ptr %v34.i60.i, align 1
  %26 = lshr i8 %v35.i.i, 2
  %v41.i61.i = and i8 %26, 48
  %v42.i62.i = or disjoint i8 %v41.i61.i, %v29.i.i
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit

cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit: ; preds = %bb1.i53.i, %bb2.i57.i
  %v43.i.i = phi i8 [ %v18.i56.i, %bb1.i53.i ], [ %v42.i62.i, %bb2.i57.i ]
  %v55.i.i = bitcast i32 %v54.i.i to float
  %v59.i = uitofp nneg i8 %v41.i45.i to float
  %v60.i = fmul contract float %v55.i.i, %v59.i
  %v21.i = shl nsw i32 %v93, 3
  %v22.i = add i32 %v20.i, %v21.i
  %v61.i = sitofp i32 %v22.i to float
  %v62.i = fmul contract float %v60.i, %v61.i
  %v55.i17.i = bitcast i32 %v54.i16.i to float
  %v66.i = uitofp nneg i8 %v43.i.i to float
  %v67.i = fmul contract float %v55.i17.i, %v66.i
  %v68.i = sitofp i32 %v93 to float
  %v69.i = fmul contract float %v67.i, %v68.i
  %v70.i = fsub contract float %v62.i, %v69.i
  %v71.i = fmul contract float %v99, %v70.i
  %v103 = fadd contract float %v54152, %v71.i
  %gep151 = getelementptr i8, ptr %invariant.gep150, i64 %v14.i.i
  %v9.sroa.0.0.copyload.i.i10 = load i32, ptr %gep151, align 1
  %v32.v.i.i11 = lshr i32 %v9.sroa.0.0.copyload.i.i10, %2
  %v32.i.i12 = and i32 %v32.v.i.i11, 252645135
  %v33.i.i13 = xor i32 %v32.i.i12, 134744072
  %v34.i.i14 = and i32 %v33.i.i13, 134744072
  %27 = mul nuw i32 %v34.i.i14, 30
  %v46.i.i15 = add nuw nsw i32 %27, %v33.i.i13
  %v20.i16 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i15, i32 %v29.sroa.0.0.copyload, i32 0) #19
  switch i16 %v10.i.i26, label %bb10.i.i139 [
    i16 0, label %bb1.i.i124
    i16 31, label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35
  ]

bb1.i.i124:                                       ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35

bb10.i.i139:                                      ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35

cuda_kernels__oxide_kernels__f16_to_f32.exit.i35: ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit, %bb1.i.i124, %bb10.i.i139
  %v54.i.i36 = phi i32 [ %v52.i.i145, %bb10.i.i139 ], [ %v17.i.i138.v34.i3.i136, %bb1.i.i124 ], [ %v42.i.i34, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit ]
  switch i16 %v10.i8.i47, label %bb10.i33.i117 [
    i16 0, label %bb1.i18.i102
    i16 31, label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i56
  ]

bb1.i18.i102:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i56

bb10.i33.i117:                                    ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i56

cuda_kernels__oxide_kernels__f16_to_f32.exit40.i56: ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35, %bb1.i18.i102, %bb10.i33.i117
  %v54.i16.i57 = phi i32 [ %v52.i39.i123, %bb10.i33.i117 ], [ %v17.i32.i116.v34.i30.i114, %bb1.i18.i102 ], [ %v42.i15.i55, %cuda_kernels__oxide_kernels__f16_to_f32.exit.i35 ]
  br i1 %v9.i41.i.not, label %bb1.i42.i60, label %bb2.i46.i94

bb1.i42.i60:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i56
  %v16.i.i61 = getelementptr i8, ptr %19, i64 %v551.i
  %v17.i43.i62 = load i8, ptr %v16.i.i61, align 1
  %v18.i44.i63 = and i8 %v17.i43.i62, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i64

bb2.i46.i94:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i56
  %v25.i47.i95 = getelementptr i8, ptr %18, i64 %v551.i
  %v26.i.i96 = load i8, ptr %v25.i47.i95, align 1
  %v27.i48.i97 = and i8 %v26.i.i96, 15
  %v32.i49.i98 = getelementptr i8, ptr %5, i64 %v551.i
  %v33.i50.i99 = load i8, ptr %v32.i49.i98, align 1
  %28 = lshr i8 %v33.i50.i99, 2
  %v39.i51.i100 = and i8 %28, 48
  %v40.i.i101 = or disjoint i8 %v39.i51.i100, %v27.i48.i97
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i64

cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i64: ; preds = %bb2.i46.i94, %bb1.i42.i60
  %v41.i45.i65 = phi i8 [ %v18.i44.i63, %bb1.i42.i60 ], [ %v40.i.i101, %bb2.i46.i94 ]
  br i1 %v9.i41.i.not, label %bb1.i53.i66, label %bb2.i57.i85

bb1.i53.i66:                                      ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i64
  %v16.i54.i67 = getelementptr i8, ptr %18, i64 %v551.i
  %v17.i55.i68 = load i8, ptr %v16.i54.i67, align 1
  %v18.i56.i69 = and i8 %v17.i55.i68, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146

bb2.i57.i85:                                      ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i64
  %v19.i.i86 = add nsw i64 %v551.i, -4
  %v25.i58.i87 = getelementptr i8, ptr %20, i64 %v19.i.i86
  %v26.i59.i88 = load i8, ptr %v25.i58.i87, align 1
  %v29.i.i89 = lshr i8 %v26.i59.i88, 4
  %v34.i60.i90 = getelementptr i8, ptr %18, i64 %v19.i.i86
  %v35.i.i91 = load i8, ptr %v34.i60.i90, align 1
  %29 = lshr i8 %v35.i.i91, 2
  %v41.i61.i92 = and i8 %29, 48
  %v42.i62.i93 = or disjoint i8 %v41.i61.i92, %v29.i.i89
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146

cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146: ; preds = %bb1.i53.i66, %bb2.i57.i85
  %v43.i.i70 = phi i8 [ %v18.i56.i69, %bb1.i53.i66 ], [ %v42.i62.i93, %bb2.i57.i85 ]
  %v55.i.i71 = bitcast i32 %v54.i.i36 to float
  %v59.i72 = uitofp nneg i8 %v41.i45.i65 to float
  %v60.i73 = fmul contract float %v55.i.i71, %v59.i72
  %v22.i75 = add i32 %v20.i16, %v21.i
  %v61.i76 = sitofp i32 %v22.i75 to float
  %v62.i77 = fmul contract float %v60.i73, %v61.i76
  %v55.i17.i78 = bitcast i32 %v54.i16.i57 to float
  %v66.i79 = uitofp nneg i8 %v43.i.i70 to float
  %v67.i80 = fmul contract float %v55.i17.i78, %v66.i79
  %v69.i82 = fmul contract float %v67.i80, %v68.i
  %v70.i83 = fsub contract float %v62.i77, %v69.i82
  %v71.i84 = fmul contract float %v99, %v70.i83
  %v107 = fadd contract float %v55153, %v71.i84
  br i1 %v9.i41.i.not, label %bb10, label %bb13

bb13:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit146
  %v109 = add nuw nsw i64 %v48158, 4
  %v49.not = icmp samesign ult i64 %v109, %v44
  br i1 %v49.not, label %bb8, label %bb15.preheader

bb18:                                             ; preds = %bb15.preheader
  %v144.4 = fadd contract float %v144.3, %v143.4
  %v142.4 = fadd contract float %v142.3, %v115.4
  %v119 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_16, i64 %v36
  %v118 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_15, i64 %v36
  store float %v142.4, ptr addrspace(3) %v118, align 4
  store float %v144.4, ptr addrspace(3) %v119, align 4
  br label %bb21

bb21:                                             ; preds = %bb18, %bb15.preheader
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v121 = icmp eq i64 %v36, 0
  br i1 %v121, label %bb23, label %bb38

bb23:                                             ; preds = %bb21
  %v122 = icmp samesign ugt i64 %v33, 3
  br i1 %v122, label %bb27, label %bb24

bb24:                                             ; preds = %bb23
  %v125 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_15, i64 %v33
  %v126 = load float, ptr addrspace(3) %v125, align 4
  br label %bb27

bb27:                                             ; preds = %bb23, %bb24
  %v127 = phi float [ %v126, %bb24 ], [ 0.000000e+00, %bb23 ]
  br i1 %v122, label %bb31, label %bb28

bb28:                                             ; preds = %bb27
  %v130 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_16, i64 %v33
  %v131 = load float, ptr addrspace(3) %v130, align 4
  br label %bb31

bb31:                                             ; preds = %bb27, %bb28
  %v132 = phi float [ %v131, %bb28 ], [ 0.000000e+00, %bb27 ]
  %v138 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v127, i32 16, i32 31) #19
  %v146 = fadd contract float %v127, %v138
  %v147 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v132, i32 16, i32 31) #19
  %v148 = fadd contract float %v132, %v147
  %v138.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v146, i32 8, i32 31) #19
  %v146.1 = fadd contract float %v146, %v138.1
  %v147.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v148, i32 8, i32 31) #19
  %v148.1 = fadd contract float %v148, %v147.1
  %v138.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v146.1, i32 4, i32 31) #19
  %v146.2 = fadd contract float %v146.1, %v138.2
  %v147.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v148.1, i32 4, i32 31) #19
  %v148.2 = fadd contract float %v148.1, %v147.2
  %v138.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v146.2, i32 2, i32 31) #19
  %v146.3 = fadd contract float %v146.2, %v138.3
  %v147.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v148.2, i32 2, i32 31) #19
  %v148.3 = fadd contract float %v148.2, %v147.3
  %v138.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v146.3, i32 1, i32 31) #19
  %v147.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v148.3, i32 1, i32 31) #19
  br i1 %v116.not, label %bb35, label %bb38

bb35:                                             ; preds = %bb31
  %v148.4 = fadd contract float %v148.3, %v147.4
  %v146.4 = fadd contract float %v146.3, %v138.4
  %v140 = fneg float %v146.4
  %30 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %30, 0
  %31 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v140, float 0x3F777313A0000000, float 5.000000e-01) #20
  %32 = tail call float @llvm.fma.f32(float %v140, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %32, float %31
  %33 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %34 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %34, float %33
  %35 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %36 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %36, float %35
  %37 = fadd float %.04.i, 0xC168000FE0000000
  %38 = fneg float %37
  %39 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v140, float 0x3FF7154760000000, float %38) #20
  %40 = tail call float @llvm.fma.f32(float %v140, float 0x3FF7154760000000, float %38)
  %.0.i = select i1 %.not.i, float %40, float %39
  %41 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v140, float 0x3E54AE0C00000000, float %.0.i) #20
  %42 = tail call float @llvm.fma.f32(float %v140, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %42, float %41
  %43 = bitcast float %.04.i to i32
  %44 = shl i32 %43, 23
  %45 = bitcast i32 %44 to float
  %46 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %47 = fmul float %46, %45
  %v150 = fadd contract float %47, 1.000000e+00
  %v151 = fdiv contract float %v146.4, %v150
  %v153 = getelementptr inbounds nuw float, ptr %v10, i64 %v38
  %v154 = fmul contract float %v148.4, %v151
  store float %v154, ptr %v153, align 4
  br label %bb38

bb38:                                             ; preds = %bb21, %bb35, %bb31, %entry
  ret void
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q4k_gemm_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(none) %v7, i64 %v8) #6 {
entry:
  %v21 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v22 = zext nneg i32 %v21 to i64
  %v23 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v24 = zext nneg i32 %v23 to i64
  %v25 = zext i32 %v4 to i64
  %v26 = zext i32 %v6 to i64
  %v27 = mul nuw i64 %v26, %v25
  %v28.not = icmp ugt i64 %v27, %v24
  br i1 %v28.not, label %bb4, label %bb24

bb4:                                              ; preds = %entry
  %v30.not = icmp eq i32 %v4, 0
  br i1 %v30.not, label %bb25, label %bb5

bb5:                                              ; preds = %bb4
  %v34 = zext i32 %v5 to i64
  %v38.not4 = icmp ult i32 %v21, %v5
  br i1 %v38.not4, label %bb7.lr.ph, label %bb9

bb7.lr.ph:                                        ; preds = %bb5
  %v4.frozen = freeze i32 %v4
  %v333 = udiv i32 %v23, %v4.frozen
  %v33.zext = zext nneg i32 %v333 to i64
  %0 = mul i32 %v333, %v4.frozen
  %v322.decomposed = sub i32 %v23, %0
  %v32.zext = zext nneg i32 %v322.decomposed to i64
  %v40 = mul nuw nsw i64 %v32.zext, %v34
  %v43 = mul nuw nsw i64 %v33.zext, %v34
  br label %bb7

bb7:                                              ; preds = %bb7.lr.ph, %bb7
  %v376 = phi i64 [ %v22, %bb7.lr.ph ], [ %v52, %bb7 ]
  %v365 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v51, %bb7 ]
  %reass.add = add nuw nsw i64 %v376, %v40
  %reass.mul = mul i64 %reass.add, 144
  %v44 = add nuw nsw i64 %v376, %v43
  %v45 = shl i64 %v44, 8
  %v50 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr %v0, i64 %v1, i64 %reass.mul, ptr %v2, i64 %v3, i64 %v45, i32 1) #19
  %v51 = fadd contract float %v365, %v50
  %v52 = add nuw nsw i64 %v376, 32
  %v38.not = icmp samesign ult i64 %v52, %v34
  br i1 %v38.not, label %bb7, label %bb9

bb9:                                              ; preds = %bb7, %bb5
  %v36.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v51, %bb7 ]
  %v53 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_1, i64 %v22
  store float %v36.lcssa, ptr addrspace(3) %v53, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v58.not = icmp samesign ult i32 %v21, 16
  br i1 %v58.not, label %bb14, label %bb18

bb14:                                             ; preds = %bb9
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v53, i64 64
  %v63 = load float, ptr addrspace(3) %gep, align 4
  %v65 = load float, ptr addrspace(3) %v53, align 4
  %v66 = fadd contract float %v63, %v65
  store float %v66, ptr addrspace(3) %v53, align 4
  br label %bb18

bb18:                                             ; preds = %bb9, %bb14
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v58.not.1 = icmp samesign ult i32 %v21, 8
  br i1 %v58.not.1, label %bb14.1, label %bb18.1

bb14.1:                                           ; preds = %bb18
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v53, i64 32
  %v63.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v65.1 = load float, ptr addrspace(3) %v53, align 4
  %v66.1 = fadd contract float %v63.1, %v65.1
  store float %v66.1, ptr addrspace(3) %v53, align 4
  br label %bb18.1

bb18.1:                                           ; preds = %bb14.1, %bb18
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v58.not.2 = icmp samesign ult i32 %v21, 4
  br i1 %v58.not.2, label %bb14.2, label %bb18.2

bb14.2:                                           ; preds = %bb18.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v53, i64 16
  %v63.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v65.2 = load float, ptr addrspace(3) %v53, align 4
  %v66.2 = fadd contract float %v63.2, %v65.2
  store float %v66.2, ptr addrspace(3) %v53, align 4
  br label %bb18.2

bb18.2:                                           ; preds = %bb14.2, %bb18.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v58.not.3 = icmp samesign ult i32 %v21, 2
  br i1 %v58.not.3, label %bb14.3, label %bb18.3

bb14.3:                                           ; preds = %bb18.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v53, i64 8
  %v63.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v65.3 = load float, ptr addrspace(3) %v53, align 4
  %v66.3 = fadd contract float %v63.3, %v65.3
  store float %v66.3, ptr addrspace(3) %v53, align 4
  br label %bb18.3

bb18.3:                                           ; preds = %bb14.3, %bb18.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v58.not.4 = icmp eq i32 %v21, 0
  br i1 %v58.not.4, label %bb14.4, label %bb18.4

bb14.4:                                           ; preds = %bb18.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v53, i64 4
  %v63.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v65.4 = load float, ptr addrspace(3) %v53, align 4
  %v66.4 = fadd contract float %v63.4, %v65.4
  store float %v66.4, ptr addrspace(3) %v53, align 4
  br label %bb18.4

bb18.4:                                           ; preds = %bb14.4, %bb18.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v69 = icmp eq i32 %v21, 0
  br i1 %v69, label %bb21, label %bb24

bb21:                                             ; preds = %bb18.4
  %v74 = getelementptr inbounds nuw float, ptr %v7, i64 %v24
  %v72 = load float, ptr addrspace(3) @__shared_mem_1, align 4
  store float %v72, ptr %v74, align 4
  br label %bb24

bb24:                                             ; preds = %bb18.4, %bb21, %entry
  ret void

bb25:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @q4k_gemv_row(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(address_is_null) %v7, i64 %v8) #0 {
entry:
  %v24 = alloca [8 x i8], align 4
  %v25 = alloca [8 x i8], align 4
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i7 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i8 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i9 = icmp eq i32 %v4.i7, 1
  %v7.i10 = icmp eq i32 %v6.i8, 1
  %v8.not.not.i = and i1 %v5.i9, %v7.i10
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i11 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i11
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v28 = zext i32 %v4 to i64
  %v29 = zext i32 %v6 to i64
  %v30 = mul nuw i64 %v29, %v28
  %v31.not = icmp ult i64 %v22.i, %v30
  br i1 %v31.not, label %bb3, label %bb53

bb3:                                              ; preds = %entry
  %v33.not = icmp eq i32 %v4, 0
  br i1 %v33.not, label %bb61, label %bb4

bb4:                                              ; preds = %bb3
  %v28.frozen = freeze i64 %v28
  %v36 = udiv i64 %v22.i, %v28.frozen
  %0 = mul i64 %v36, %v28.frozen
  %v35.decomposed = sub i64 %v22.i, %0
  %v37 = mul i32 %v5, 144
  %v38 = zext i32 %v37 to i64
  %v39 = mul nuw i64 %v35.decomposed, %v38
  %v42.not98.not = icmp eq i32 %v5, 0
  br i1 %v42.not98.not, label %bb49, label %bb6.lr.ph

bb6.lr.ph:                                        ; preds = %bb4
  %v140.fca.4.gep = getelementptr inbounds nuw i8, ptr %v24, i64 4
  %v141.fca.4.gep = getelementptr inbounds nuw i8, ptr %v25, i64 4
  %v143 = zext i32 %v5 to i64
  %v144 = mul i64 %v36, %v143
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb48
  %v41100 = phi i32 [ 0, %bb6.lr.ph ], [ %v247, %bb48 ]
  %v4099 = phi float [ 0.000000e+00, %bb6.lr.ph ], [ %v244, %bb48 ]
  %v44 = zext i32 %v41100 to i64
  %v45 = mul nuw nsw i64 %v44, 144
  %v46 = add i64 %v45, %v39
  %v48 = icmp ult i64 %v46, %v1
  br i1 %v48, label %bb7, label %bb62

bb7:                                              ; preds = %bb6
  %v52 = or disjoint i64 %v46, 1
  %v53 = icmp ult i64 %v52, %v1
  br i1 %v53, label %bb8, label %bb63

bb8:                                              ; preds = %bb7
  %v50 = getelementptr inbounds i8, ptr %v0, i64 %v46
  %v51 = load i8, ptr %v50, align 1
  %v55 = getelementptr inbounds i8, ptr %v0, i64 %v52
  %v56 = load i8, ptr %v55, align 1
  %v60 = alloca [2 x i8], align 2
  store i8 %v51, ptr %v60, align 2
  %v60.repack1 = getelementptr inbounds nuw i8, ptr %v60, i64 1
  store i8 %v56, ptr %v60.repack1, align 1
  %v61 = load i16, ptr %v60, align 2
  %v62 = or disjoint i64 %v46, 2
  %v63 = icmp ult i64 %v62, %v1
  br i1 %v63, label %bb9, label %bb64

bb9:                                              ; preds = %bb8
  %v67 = or disjoint i64 %v46, 3
  %v68 = icmp ult i64 %v67, %v1
  br i1 %v68, label %bb10, label %bb65

bb10:                                             ; preds = %bb9
  %v65 = getelementptr inbounds i8, ptr %v0, i64 %v62
  %v66 = load i8, ptr %v65, align 1
  %v70 = getelementptr inbounds i8, ptr %v0, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v75 = alloca [2 x i8], align 2
  store i8 %v66, ptr %v75, align 2
  %v75.repack3 = getelementptr inbounds nuw i8, ptr %v75, i64 1
  store i8 %v71, ptr %v75.repack3, align 1
  %v76 = load i16, ptr %v75, align 2
  %v4.i12 = lshr i16 %v61, 15
  %v6.i13 = zext nneg i16 %v4.i12 to i32
  %v9.i = lshr i16 %v61, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v61, 1023
  %v13.i14 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb10
  %v15.i15 = icmp eq i16 %v12.i, 0
  br i1 %v15.i15, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i16 = shl nuw i32 %v6.i13, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i14, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i14, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i13, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb10
  %v38.i = shl nuw i32 %v6.i13, 31
  %v41.i = shl nuw nsw i32 %v13.i14, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb10
  %v44.i = shl nuw i32 %v6.i13, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i14, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i16, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v4.i17 = lshr i16 %v76, 15
  %v6.i18 = zext nneg i16 %v4.i17 to i32
  %v9.i19 = lshr i16 %v76, 10
  %v10.i20 = and i16 %v9.i19, 31
  %v12.i21 = and i16 %v76, 1023
  %v13.i22 = zext nneg i16 %v12.i21 to i32
  switch i16 %v10.i20, label %bb10.i45 [
    i16 0, label %bb1.i30
    i16 31, label %bb9.i23
  ]

bb1.i30:                                          ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v15.i31 = icmp eq i16 %v12.i21, 0
  br i1 %v15.i31, label %bb2.i43, label %bb6.i32

bb2.i43:                                          ; preds = %bb1.i30
  %v17.i44 = shl nuw i32 %v6.i18, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit52

bb6.i32:                                          ; preds = %bb1.i30
  %v13.masked.numleadingzeros.i33 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i22, i1 true)
  %v13.masked.leadingonepos.i34 = xor i32 %v13.masked.numleadingzeros.i33, 31
  %bb5.tripcount.i35 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i34
  %v23.i36 = shl nuw nsw i32 %v13.i22, %bb5.tripcount.i35
  %v27.i37 = shl nuw i32 %v6.i18, 31
  %3 = shl nuw nsw i32 %v13.masked.numleadingzeros.i33, 23
  %reass.sub101 = sub i32 %v27.i37, %3
  %v31.i39 = add i32 %reass.sub101, 1124073472
  %v25.i40 = shl i32 %v23.i36, 13
  %v33.i41 = and i32 %v25.i40, 8380416
  %v34.i42 = or disjoint i32 %v33.i41, %v31.i39
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit52

bb9.i23:                                          ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v38.i24 = shl nuw i32 %v6.i18, 31
  %v41.i25 = shl nuw nsw i32 %v13.i22, 13
  %v39.i26 = or disjoint i32 %v38.i24, %v41.i25
  %v42.i27 = or disjoint i32 %v39.i26, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit52

bb10.i45:                                         ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v44.i46 = shl nuw i32 %v6.i18, 31
  %4 = add nuw nsw i16 %v10.i20, 112
  %v46.i47 = zext nneg i16 %4 to i32
  %v48.i48 = shl nuw nsw i32 %v46.i47, 23
  %v49.i49 = or disjoint i32 %v48.i48, %v44.i46
  %v51.i50 = shl nuw nsw i32 %v13.i22, 13
  %v52.i51 = or disjoint i32 %v49.i49, %v51.i50
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit52

cuda_kernels__oxide_kernels__f16_to_f32.exit52:   ; preds = %bb2.i43, %bb6.i32, %bb9.i23, %bb10.i45
  %v54.i28 = phi i32 [ %v34.i42, %bb6.i32 ], [ %v17.i44, %bb2.i43 ], [ %v42.i27, %bb9.i23 ], [ %v52.i51, %bb10.i45 ]
  %v55.i29 = bitcast i32 %v54.i28 to float
  %v79 = or disjoint i64 %v46, 4
  %v80 = icmp ult i64 %v79, %v1
  br i1 %v80, label %bb13, label %bb66

bb13:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit52
  %v82 = getelementptr inbounds i8, ptr %v0, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = or disjoint i64 %v46, 5
  %v85 = icmp ult i64 %v84, %v1
  br i1 %v85, label %bb14, label %bb67

bb14:                                             ; preds = %bb13
  %v87 = getelementptr inbounds i8, ptr %v0, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = or disjoint i64 %v46, 6
  %v90 = icmp ult i64 %v89, %v1
  br i1 %v90, label %bb15, label %bb68

bb15:                                             ; preds = %bb14
  %v92 = getelementptr inbounds i8, ptr %v0, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = or disjoint i64 %v46, 7
  %v95 = icmp ult i64 %v94, %v1
  br i1 %v95, label %bb16, label %bb69

bb16:                                             ; preds = %bb15
  %v97 = getelementptr inbounds i8, ptr %v0, i64 %v94
  %v98 = load i8, ptr %v97, align 1
  %v99 = or disjoint i64 %v46, 8
  %v100 = icmp ult i64 %v99, %v1
  br i1 %v100, label %bb17, label %bb70

bb17:                                             ; preds = %bb16
  %v102 = getelementptr inbounds i8, ptr %v0, i64 %v99
  %v103 = load i8, ptr %v102, align 1
  %v104 = or disjoint i64 %v46, 9
  %v105 = icmp ult i64 %v104, %v1
  br i1 %v105, label %bb18, label %bb71

bb18:                                             ; preds = %bb17
  %v107 = getelementptr inbounds i8, ptr %v0, i64 %v104
  %v108 = load i8, ptr %v107, align 1
  %v109 = or disjoint i64 %v46, 10
  %v110 = icmp ult i64 %v109, %v1
  br i1 %v110, label %bb19, label %bb72

bb19:                                             ; preds = %bb18
  %v112 = getelementptr inbounds i8, ptr %v0, i64 %v109
  %v113 = load i8, ptr %v112, align 1
  %v114 = or disjoint i64 %v46, 11
  %v115 = icmp ult i64 %v114, %v1
  br i1 %v115, label %bb20, label %bb73

bb20:                                             ; preds = %bb19
  %v117 = getelementptr inbounds i8, ptr %v0, i64 %v114
  %v118 = load i8, ptr %v117, align 1
  %v119 = or disjoint i64 %v46, 12
  %v120 = icmp ult i64 %v119, %v1
  br i1 %v120, label %bb21, label %bb74

bb21:                                             ; preds = %bb20
  %v122 = getelementptr inbounds i8, ptr %v0, i64 %v119
  %v123 = load i8, ptr %v122, align 1
  %v124 = or disjoint i64 %v46, 13
  %v125 = icmp ult i64 %v124, %v1
  br i1 %v125, label %bb22, label %bb75

bb22:                                             ; preds = %bb21
  %v127 = getelementptr inbounds i8, ptr %v0, i64 %v124
  %v128 = load i8, ptr %v127, align 1
  %v129 = or disjoint i64 %v46, 14
  %v130 = icmp ult i64 %v129, %v1
  br i1 %v130, label %bb23, label %bb76

bb23:                                             ; preds = %bb22
  %v134 = or disjoint i64 %v46, 15
  %v135 = icmp ult i64 %v134, %v1
  br i1 %v135, label %bb24, label %bb77

bb24:                                             ; preds = %bb23
  %v132 = getelementptr inbounds i8, ptr %v0, i64 %v129
  %v133 = load i8, ptr %v132, align 1
  %v137 = getelementptr inbounds i8, ptr %v0, i64 %v134
  %v138 = load i8, ptr %v137, align 1
  %v43.sroa.4.0.insert.ext.i = zext i8 %v98 to i32
  %v43.sroa.4.0.insert.shift.i = shl nuw i32 %v43.sroa.4.0.insert.ext.i, 24
  %v43.sroa.3.0.insert.ext.i = zext i8 %v93 to i32
  %v43.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.3.0.insert.ext.i, 16
  %v43.sroa.2.0.insert.ext.i = zext i8 %v88 to i32
  %v43.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.2.0.insert.ext.i, 8
  %v43.sroa.0.0.insert.ext.i = zext i8 %v83 to i32
  %v43.sroa.3.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.shift.i, %v43.sroa.0.0.insert.ext.i
  %v43.sroa.2.0.insert.insert.i = or disjoint i32 %v43.sroa.3.0.insert.insert.i, %v43.sroa.3.0.insert.shift.i
  %v43.sroa.0.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.insert.i, %v43.sroa.4.0.insert.shift.i
  %v51.sroa.4.0.insert.ext.i = zext i8 %v118 to i32
  %v51.sroa.4.0.insert.shift.i = shl nuw i32 %v51.sroa.4.0.insert.ext.i, 24
  %v51.sroa.3.0.insert.ext.i = zext i8 %v113 to i32
  %v51.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.3.0.insert.ext.i, 16
  %v51.sroa.2.0.insert.ext.i = zext i8 %v108 to i32
  %v51.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.2.0.insert.ext.i, 8
  %v51.sroa.0.0.insert.ext.i = zext i8 %v103 to i32
  %v51.sroa.3.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.shift.i, %v51.sroa.0.0.insert.ext.i
  %v51.sroa.2.0.insert.insert.i = or disjoint i32 %v51.sroa.3.0.insert.insert.i, %v51.sroa.3.0.insert.shift.i
  %v51.sroa.0.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.insert.i, %v51.sroa.4.0.insert.shift.i
  %v59.sroa.4.0.insert.ext.i = zext i8 %v138 to i32
  %v59.sroa.4.0.insert.shift.i = shl nuw i32 %v59.sroa.4.0.insert.ext.i, 24
  %v59.sroa.3.0.insert.ext.i = zext i8 %v133 to i32
  %v59.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.3.0.insert.ext.i, 16
  %v59.sroa.2.0.insert.ext.i = zext i8 %v128 to i32
  %v59.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.2.0.insert.ext.i, 8
  %v59.sroa.0.0.insert.ext.i = zext i8 %v123 to i32
  %v59.sroa.3.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.shift.i, %v59.sroa.0.0.insert.ext.i
  %v59.sroa.2.0.insert.insert.i = or disjoint i32 %v59.sroa.3.0.insert.insert.i, %v59.sroa.3.0.insert.shift.i
  %v59.sroa.0.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.insert.i, %v59.sroa.4.0.insert.shift.i
  %v65.i = lshr i32 %v59.sroa.0.0.insert.insert.i, 4
  %v66.i = and i32 %v65.i, 252645135
  %5 = lshr i32 %v51.sroa.0.0.insert.insert.i, 2
  %v73.i = and i32 %5, 808464432
  %v81.i = and i32 %v59.sroa.0.0.insert.insert.i, 252645135
  %6 = lshr i32 %v43.sroa.0.0.insert.insert.i, 2
  %v88.i = and i32 %6, 808464432
  %v94.i = and i32 %v43.sroa.0.0.insert.insert.i, 1061109567
  %v98.sroa.2.0.extract.shift.i = lshr i32 %v94.i, 8
  %v98.sroa.4.0.extract.shift.i = lshr i32 %v94.i, 24
  %v98.sroa.3.0.extract.shift.i = lshr i32 %v94.i, 16
  %v98.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v98.sroa.4.0.extract.shift.i to i8
  %v98.sroa.3.0.extract.trunc.i = trunc i32 %v98.sroa.3.0.extract.shift.i to i8
  %7 = insertelement <4 x i32> poison, i32 %v94.i, i64 0
  %8 = insertelement <4 x i32> %7, i32 %v98.sroa.2.0.extract.shift.i, i64 1
  %9 = trunc <4 x i32> %8 to <4 x i8>
  %10 = insertelement <4 x i8> %9, i8 %v98.sroa.3.0.extract.trunc.i, i64 2
  %11 = insertelement <4 x i8> %10, i8 %v98.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %11, ptr %v24, align 4
  %v89.i = or disjoint i32 %v81.i, %v88.i
  %v102.sroa.2.0.extract.shift.i = lshr i32 %v89.i, 8
  %v102.sroa.4.0.extract.shift.i = lshr i32 %v89.i, 24
  %v102.sroa.3.0.extract.shift.i = lshr i32 %v89.i, 16
  %v102.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v102.sroa.4.0.extract.shift.i to i8
  %v102.sroa.3.0.extract.trunc.i = trunc i32 %v102.sroa.3.0.extract.shift.i to i8
  %12 = insertelement <4 x i32> poison, i32 %v89.i, i64 0
  %13 = insertelement <4 x i32> %12, i32 %v102.sroa.2.0.extract.shift.i, i64 1
  %14 = trunc <4 x i32> %13 to <4 x i8>
  %15 = insertelement <4 x i8> %14, i8 %v102.sroa.3.0.extract.trunc.i, i64 2
  %16 = insertelement <4 x i8> %15, i8 %v102.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %16, ptr %v140.fca.4.gep, align 4
  %v78.i = and i32 %v51.sroa.0.0.insert.insert.i, 1061109567
  %v106.sroa.2.0.extract.shift.i = lshr i32 %v78.i, 8
  %v106.sroa.4.0.extract.shift.i = lshr i32 %v78.i, 24
  %v106.sroa.3.0.extract.shift.i = lshr i32 %v78.i, 16
  %v106.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v106.sroa.4.0.extract.shift.i to i8
  %v106.sroa.3.0.extract.trunc.i = trunc i32 %v106.sroa.3.0.extract.shift.i to i8
  %17 = insertelement <4 x i32> poison, i32 %v78.i, i64 0
  %18 = insertelement <4 x i32> %17, i32 %v106.sroa.2.0.extract.shift.i, i64 1
  %19 = trunc <4 x i32> %18 to <4 x i8>
  %20 = insertelement <4 x i8> %19, i8 %v106.sroa.3.0.extract.trunc.i, i64 2
  %21 = insertelement <4 x i8> %20, i8 %v106.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %21, ptr %v25, align 4
  %v74.i = or disjoint i32 %v66.i, %v73.i
  %v110.sroa.2.0.extract.shift.i = lshr i32 %v74.i, 8
  %v110.sroa.4.0.extract.shift.i = lshr i32 %v74.i, 24
  %v110.sroa.3.0.extract.shift.i = lshr i32 %v74.i, 16
  %v110.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v110.sroa.4.0.extract.shift.i to i8
  %v110.sroa.3.0.extract.trunc.i = trunc i32 %v110.sroa.3.0.extract.shift.i to i8
  %22 = insertelement <4 x i32> poison, i32 %v74.i, i64 0
  %23 = insertelement <4 x i32> %22, i32 %v110.sroa.2.0.extract.shift.i, i64 1
  %24 = trunc <4 x i32> %23 to <4 x i8>
  %25 = insertelement <4 x i8> %24, i8 %v110.sroa.3.0.extract.trunc.i, i64 2
  %26 = insertelement <4 x i8> %25, i8 %v110.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %26, ptr %v141.fca.4.gep, align 4
  %v142 = add nuw i64 %v46, 16
  %v1455 = add i64 %v144, %v44
  %v148 = shl i64 %v1455, 8
  br label %bb28.preheader

bb28.preheader:                                   ; preds = %bb24, %bb32
  %v15089 = phi i64 [ 0, %bb24 ], [ %v173, %bb32 ]
  %v14988 = phi float [ 0.000000e+00, %bb24 ], [ %v172, %bb32 ]
  %v157 = shl nuw nsw i64 %v15089, 5
  %v158 = add nuw nsw i64 %v157, %v148
  br label %bb29

bb29:                                             ; preds = %bb30.1, %bb28.preheader
  %v15487 = phi i64 [ 0, %bb28.preheader ], [ %v166.1, %bb30.1 ]
  %v15386 = phi float [ 0.000000e+00, %bb28.preheader ], [ %v165.1, %bb30.1 ]
  %v159 = add nuw nsw i64 %v15487, %v158
  %v161 = icmp ult i64 %v159, %v3
  br i1 %v161, label %bb30, label %bb78

bb30:                                             ; preds = %bb29
  %v166 = or disjoint i64 %v15487, 1
  %v159.1 = add nuw nsw i64 %v166, %v158
  %v161.1 = icmp ult i64 %v159.1, %v3
  br i1 %v161.1, label %bb30.1, label %bb78

bb30.1:                                           ; preds = %bb30
  %v163 = getelementptr inbounds float, ptr %v2, i64 %v159
  %v164 = load float, ptr %v163, align 4
  %v165 = fadd contract float %v15386, %v164
  %v163.1 = getelementptr inbounds float, ptr %v2, i64 %v159.1
  %v164.1 = load float, ptr %v163.1, align 4
  %v165.1 = fadd contract float %v165, %v164.1
  %v166.1 = add nuw nsw i64 %v15487, 2
  %exitcond.1 = icmp eq i64 %v166.1, 32
  br i1 %exitcond.1, label %bb32, label %bb29

bb32:                                             ; preds = %bb30.1
  %v168 = getelementptr inbounds nuw i8, ptr %v25, i64 %v15089
  %v169 = load i8, ptr %v168, align 1
  %v170 = uitofp i8 %v169 to float
  %v171 = fmul contract float %v165.1, %v170
  %v172 = fadd contract float %v14988, %v171
  %v173 = add nuw nsw i64 %v15089, 1
  %exitcond102 = icmp eq i64 %v173, 8
  br i1 %exitcond102, label %bb33, label %bb28.preheader

bb33:                                             ; preds = %bb32
  %v174 = fmul contract float %v172, %v55.i29
  %v175 = fsub contract float %v4099, %v174
  %v213 = or disjoint i64 %v148, 32
  br label %bb35

bb35:                                             ; preds = %bb33, %bb47
  %v17997 = phi i64 [ 0, %bb33 ], [ %v246, %bb47 ]
  %v17896 = phi i64 [ 0, %bb33 ], [ %v245, %bb47 ]
  %v17795 = phi i64 [ 0, %bb33 ], [ %v218, %bb47 ]
  %v17694 = phi float [ %v175, %bb33 ], [ %v244, %bb47 ]
  %v182 = shl nuw nsw i64 %v17997, 5
  %v183 = add i64 %v142, %v182
  %v185 = getelementptr inbounds nuw i8, ptr %v24, i64 %v17795
  %v186 = load i8, ptr %v185, align 2
  %v187 = uitofp i8 %v186 to float
  %v200 = add nuw nsw i64 %v17896, %v148
  br label %bb38

bb38:                                             ; preds = %bb35, %bb40
  %v19091 = phi i64 [ 0, %bb35 ], [ %v209, %bb40 ]
  %v18990 = phi float [ 0.000000e+00, %bb35 ], [ %v208, %bb40 ]
  %v193 = add nuw i64 %v19091, %v183
  %v194 = icmp ult i64 %v193, %v1
  br i1 %v194, label %bb39, label %bb81

bb39:                                             ; preds = %bb38
  %v201 = add nuw nsw i64 %v19091, %v200
  %v203 = icmp ult i64 %v201, %v3
  br i1 %v203, label %bb40, label %bb82

bb40:                                             ; preds = %bb39
  %v196 = getelementptr inbounds i8, ptr %v0, i64 %v193
  %v197 = load i8, ptr %v196, align 1
  %v198 = and i8 %v197, 15
  %v199 = uitofp nneg i8 %v198 to float
  %v205 = getelementptr inbounds float, ptr %v2, i64 %v201
  %v206 = load float, ptr %v205, align 4
  %v207 = fmul contract float %v206, %v199
  %v208 = fadd contract float %v18990, %v207
  %v209 = add nuw nsw i64 %v19091, 1
  %exitcond103 = icmp eq i64 %v209, 32
  br i1 %exitcond103, label %bb41, label %bb38

bb41:                                             ; preds = %bb40
  %v210 = fmul contract float %v55.i, %v187
  %v211 = fmul contract float %v210, %v208
  %v212 = fadd contract float %v17694, %v211
  %v215 = getelementptr inbounds nuw i8, ptr %v185, i64 1
  %v216 = load i8, ptr %v215, align 1
  %v217 = uitofp i8 %v216 to float
  %v218 = add nuw nsw i64 %v17795, 2
  %v232 = add nuw nsw i64 %v213, %v17896
  br label %bb44

bb44:                                             ; preds = %bb41, %bb46
  %v22093 = phi i64 [ 0, %bb41 ], [ %v241, %bb46 ]
  %v21992 = phi float [ 0.000000e+00, %bb41 ], [ %v240, %bb46 ]
  %v223 = add nuw i64 %v22093, %v183
  %v224 = icmp ult i64 %v223, %v1
  br i1 %v224, label %bb45, label %bb84

bb45:                                             ; preds = %bb44
  %v233 = add nuw nsw i64 %v22093, %v232
  %v235 = icmp ult i64 %v233, %v3
  br i1 %v235, label %bb46, label %bb85

bb46:                                             ; preds = %bb45
  %v226 = getelementptr inbounds i8, ptr %v0, i64 %v223
  %v227 = load i8, ptr %v226, align 1
  %v230 = lshr i8 %v227, 4
  %v231 = uitofp nneg i8 %v230 to float
  %v237 = getelementptr inbounds float, ptr %v2, i64 %v233
  %v238 = load float, ptr %v237, align 4
  %v239 = fmul contract float %v238, %v231
  %v240 = fadd contract float %v21992, %v239
  %v241 = add nuw nsw i64 %v22093, 1
  %exitcond104 = icmp eq i64 %v241, 32
  br i1 %exitcond104, label %bb47, label %bb44

bb47:                                             ; preds = %bb46
  %v242 = fmul contract float %v55.i, %v217
  %v243 = fmul contract float %v242, %v240
  %v244 = fadd contract float %v212, %v243
  %v245 = add nuw nsw i64 %v17896, 64
  %v246 = add nuw nsw i64 %v17997, 1
  %exitcond105 = icmp eq i64 %v246, 4
  br i1 %exitcond105, label %bb48, label %bb35

bb48:                                             ; preds = %bb47
  %v247 = add nuw i32 %v41100, 1
  %exitcond106.not = icmp eq i32 %v247, %v5
  br i1 %exitcond106.not, label %bb49, label %bb6

bb49:                                             ; preds = %bb48, %bb4
  %v40.lcssa = phi float [ 0.000000e+00, %bb4 ], [ %v244, %bb48 ]
  %v251 = icmp ult i64 %v22.i, %v8
  %or.cond.not = select i1 %.v18.i, i1 %v251, i1 false
  %v2656 = icmp ne ptr %v7, null
  %v265 = select i1 %or.cond.not, i1 %v2656, i1 false
  br i1 %v265, label %bb50, label %bb53

bb50:                                             ; preds = %bb49
  %v254 = getelementptr inbounds nuw float, ptr %v7, i64 %v22.i
  store float %v40.lcssa, ptr %v254, align 4
  br label %bb53

bb53:                                             ; preds = %bb49, %bb50, %entry
  ret void

bb61:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb62:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb63:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb64:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb65:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb66:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit52
  tail call void @llvm.trap() #19
  unreachable

bb67:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb68:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb69:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb70:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb71:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb72:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb73:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb74:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb75:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable

bb76:                                             ; preds = %bb22
  tail call void @llvm.trap() #19
  unreachable

bb77:                                             ; preds = %bb23
  tail call void @llvm.trap() #19
  unreachable

bb78:                                             ; preds = %bb30, %bb29
  tail call void @llvm.trap() #19
  unreachable

bb81:                                             ; preds = %bb38
  tail call void @llvm.trap() #19
  unreachable

bb82:                                             ; preds = %bb39
  tail call void @llvm.trap() #19
  unreachable

bb84:                                             ; preds = %bb44
  tail call void @llvm.trap() #19
  unreachable

bb85:                                             ; preds = %bb45
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q4k_gemv_row_tiled(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, ptr writeonly captures(none) %v6, i64 %v7) #6 {
entry:
  %v21 = alloca [8 x i8], align 4
  %v22 = alloca [8 x i8], align 4
  %v23 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v24 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v25.not = icmp ult i32 %v24, %v4
  br i1 %v25.not, label %bb4, label %bb64

bb4:                                              ; preds = %entry
  %v27 = zext nneg i32 %v24 to i64
  %v28 = mul i32 %v5, 144
  %v29 = zext i32 %v28 to i64
  %v30 = mul nuw nsw i64 %v29, %v27
  %v33.not87 = icmp ult i32 %v23, %v5
  br i1 %v33.not87, label %bb6.lr.ph, label %bb49

bb6.lr.ph:                                        ; preds = %bb4
  %v131.fca.4.gep = getelementptr inbounds nuw i8, ptr %v21, i64 4
  %v132.fca.4.gep = getelementptr inbounds nuw i8, ptr %v22, i64 4
  %0 = add nuw nsw i64 %v30, 16
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb48
  %v3289 = phi i32 [ %v23, %bb6.lr.ph ], [ %v234, %bb48 ]
  %v3188 = phi float [ 0.000000e+00, %bb6.lr.ph ], [ %v231, %bb48 ]
  %1 = zext i32 %v3289 to i64
  %2 = shl nuw nsw i64 %1, 8
  %3 = or disjoint i64 %2, 32
  %4 = sub nuw nsw i64 -32, %2
  %5 = mul nuw nsw i64 %1, 144
  %6 = add nuw i64 %0, %5
  %7 = add nuw i64 %v30, %5
  %8 = sub nuw nsw i64 -16, %7
  %9 = mul nsw i64 %1, -256
  %v37 = add nuw i64 %5, %v30
  %v39 = icmp ult i64 %v37, %v1
  br i1 %v39, label %bb7, label %bb65

bb7:                                              ; preds = %bb6
  %v43 = or disjoint i64 %v37, 1
  %v44 = icmp ult i64 %v43, %v1
  br i1 %v44, label %bb8, label %bb66

bb8:                                              ; preds = %bb7
  %v41 = getelementptr inbounds i8, ptr %v0, i64 %v37
  %v42 = load i8, ptr %v41, align 1
  %v46 = getelementptr inbounds i8, ptr %v0, i64 %v43
  %v47 = load i8, ptr %v46, align 1
  %v51 = alloca [2 x i8], align 2
  store i8 %v42, ptr %v51, align 2
  %v51.repack1 = getelementptr inbounds nuw i8, ptr %v51, i64 1
  store i8 %v47, ptr %v51.repack1, align 1
  %v52 = load i16, ptr %v51, align 2
  %v53 = or disjoint i64 %v37, 2
  %v54 = icmp ult i64 %v53, %v1
  br i1 %v54, label %bb9, label %bb67

bb9:                                              ; preds = %bb8
  %v58 = or disjoint i64 %v37, 3
  %v59 = icmp ult i64 %v58, %v1
  br i1 %v59, label %bb10, label %bb68

bb10:                                             ; preds = %bb9
  %v56 = getelementptr inbounds i8, ptr %v0, i64 %v53
  %v57 = load i8, ptr %v56, align 1
  %v61 = getelementptr inbounds i8, ptr %v0, i64 %v58
  %v62 = load i8, ptr %v61, align 1
  %v66 = alloca [2 x i8], align 2
  store i8 %v57, ptr %v66, align 2
  %v66.repack3 = getelementptr inbounds nuw i8, ptr %v66, i64 1
  store i8 %v62, ptr %v66.repack3, align 1
  %v67 = load i16, ptr %v66, align 2
  %v4.i = lshr i16 %v52, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v52, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v52, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb10
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %10 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %10
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb10
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb10
  %v44.i = shl nuw i32 %v6.i, 31
  %11 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %11 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v4.i6 = lshr i16 %v67, 15
  %v6.i7 = zext nneg i16 %v4.i6 to i32
  %v9.i8 = lshr i16 %v67, 10
  %v10.i9 = and i16 %v9.i8, 31
  %v12.i10 = and i16 %v67, 1023
  %v13.i11 = zext nneg i16 %v12.i10 to i32
  switch i16 %v10.i9, label %bb10.i34 [
    i16 0, label %bb1.i19
    i16 31, label %bb9.i12
  ]

bb1.i19:                                          ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v15.i20 = icmp eq i16 %v12.i10, 0
  br i1 %v15.i20, label %bb2.i32, label %bb6.i21

bb2.i32:                                          ; preds = %bb1.i19
  %v17.i33 = shl nuw i32 %v6.i7, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit41

bb6.i21:                                          ; preds = %bb1.i19
  %v13.masked.numleadingzeros.i22 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i11, i1 true)
  %v13.masked.leadingonepos.i23 = xor i32 %v13.masked.numleadingzeros.i22, 31
  %bb5.tripcount.i24 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i23
  %v23.i25 = shl nuw nsw i32 %v13.i11, %bb5.tripcount.i24
  %v27.i26 = shl nuw i32 %v6.i7, 31
  %12 = shl nuw nsw i32 %v13.masked.numleadingzeros.i22, 23
  %reass.sub91 = sub i32 %v27.i26, %12
  %v31.i28 = add i32 %reass.sub91, 1124073472
  %v25.i29 = shl i32 %v23.i25, 13
  %v33.i30 = and i32 %v25.i29, 8380416
  %v34.i31 = or disjoint i32 %v33.i30, %v31.i28
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit41

bb9.i12:                                          ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v38.i13 = shl nuw i32 %v6.i7, 31
  %v41.i14 = shl nuw nsw i32 %v13.i11, 13
  %v39.i15 = or disjoint i32 %v38.i13, %v41.i14
  %v42.i16 = or disjoint i32 %v39.i15, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit41

bb10.i34:                                         ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v44.i35 = shl nuw i32 %v6.i7, 31
  %13 = add nuw nsw i16 %v10.i9, 112
  %v46.i36 = zext nneg i16 %13 to i32
  %v48.i37 = shl nuw nsw i32 %v46.i36, 23
  %v49.i38 = or disjoint i32 %v48.i37, %v44.i35
  %v51.i39 = shl nuw nsw i32 %v13.i11, 13
  %v52.i40 = or disjoint i32 %v49.i38, %v51.i39
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit41

cuda_kernels__oxide_kernels__f16_to_f32.exit41:   ; preds = %bb2.i32, %bb6.i21, %bb9.i12, %bb10.i34
  %v54.i17 = phi i32 [ %v34.i31, %bb6.i21 ], [ %v17.i33, %bb2.i32 ], [ %v42.i16, %bb9.i12 ], [ %v52.i40, %bb10.i34 ]
  %v55.i18 = bitcast i32 %v54.i17 to float
  %v70 = or disjoint i64 %v37, 4
  %v71 = icmp ult i64 %v70, %v1
  br i1 %v71, label %bb13, label %bb69

bb13:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit41
  %v73 = getelementptr inbounds i8, ptr %v0, i64 %v70
  %v74 = load i8, ptr %v73, align 1
  %v75 = or disjoint i64 %v37, 5
  %v76 = icmp ult i64 %v75, %v1
  br i1 %v76, label %bb14, label %bb70

bb14:                                             ; preds = %bb13
  %v78 = getelementptr inbounds i8, ptr %v0, i64 %v75
  %v79 = load i8, ptr %v78, align 1
  %v80 = or disjoint i64 %v37, 6
  %v81 = icmp ult i64 %v80, %v1
  br i1 %v81, label %bb15, label %bb71

bb15:                                             ; preds = %bb14
  %v83 = getelementptr inbounds i8, ptr %v0, i64 %v80
  %v84 = load i8, ptr %v83, align 1
  %v85 = or disjoint i64 %v37, 7
  %v86 = icmp ult i64 %v85, %v1
  br i1 %v86, label %bb16, label %bb72

bb16:                                             ; preds = %bb15
  %v88 = getelementptr inbounds i8, ptr %v0, i64 %v85
  %v89 = load i8, ptr %v88, align 1
  %v90 = or disjoint i64 %v37, 8
  %v91 = icmp ult i64 %v90, %v1
  br i1 %v91, label %bb17, label %bb73

bb17:                                             ; preds = %bb16
  %v93 = getelementptr inbounds i8, ptr %v0, i64 %v90
  %v94 = load i8, ptr %v93, align 1
  %v95 = or disjoint i64 %v37, 9
  %v96 = icmp ult i64 %v95, %v1
  br i1 %v96, label %bb18, label %bb74

bb18:                                             ; preds = %bb17
  %v98 = getelementptr inbounds i8, ptr %v0, i64 %v95
  %v99 = load i8, ptr %v98, align 1
  %v100 = or disjoint i64 %v37, 10
  %v101 = icmp ult i64 %v100, %v1
  br i1 %v101, label %bb19, label %bb75

bb19:                                             ; preds = %bb18
  %v103 = getelementptr inbounds i8, ptr %v0, i64 %v100
  %v104 = load i8, ptr %v103, align 1
  %v105 = or disjoint i64 %v37, 11
  %v106 = icmp ult i64 %v105, %v1
  br i1 %v106, label %bb20, label %bb76

bb20:                                             ; preds = %bb19
  %v108 = getelementptr inbounds i8, ptr %v0, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v110 = or disjoint i64 %v37, 12
  %v111 = icmp ult i64 %v110, %v1
  br i1 %v111, label %bb21, label %bb77

bb21:                                             ; preds = %bb20
  %v113 = getelementptr inbounds i8, ptr %v0, i64 %v110
  %v114 = load i8, ptr %v113, align 1
  %v115 = or disjoint i64 %v37, 13
  %v116 = icmp ult i64 %v115, %v1
  br i1 %v116, label %bb22, label %bb78

bb22:                                             ; preds = %bb21
  %v118 = getelementptr inbounds i8, ptr %v0, i64 %v115
  %v119 = load i8, ptr %v118, align 1
  %v120 = or disjoint i64 %v37, 14
  %v121 = icmp ult i64 %v120, %v1
  br i1 %v121, label %bb23, label %bb79

bb23:                                             ; preds = %bb22
  %v125 = or disjoint i64 %v37, 15
  %v126 = icmp ult i64 %v125, %v1
  br i1 %v126, label %bb24, label %bb80

bb24:                                             ; preds = %bb23
  %v123 = getelementptr inbounds i8, ptr %v0, i64 %v120
  %v124 = load i8, ptr %v123, align 1
  %v128 = getelementptr inbounds i8, ptr %v0, i64 %v125
  %v129 = load i8, ptr %v128, align 1
  %v43.sroa.4.0.insert.ext.i = zext i8 %v89 to i32
  %v43.sroa.4.0.insert.shift.i = shl nuw i32 %v43.sroa.4.0.insert.ext.i, 24
  %v43.sroa.3.0.insert.ext.i = zext i8 %v84 to i32
  %v43.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.3.0.insert.ext.i, 16
  %v43.sroa.2.0.insert.ext.i = zext i8 %v79 to i32
  %v43.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.2.0.insert.ext.i, 8
  %v43.sroa.0.0.insert.ext.i = zext i8 %v74 to i32
  %v43.sroa.3.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.shift.i, %v43.sroa.0.0.insert.ext.i
  %v43.sroa.2.0.insert.insert.i = or disjoint i32 %v43.sroa.3.0.insert.insert.i, %v43.sroa.3.0.insert.shift.i
  %v43.sroa.0.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.insert.i, %v43.sroa.4.0.insert.shift.i
  %v51.sroa.4.0.insert.ext.i = zext i8 %v109 to i32
  %v51.sroa.4.0.insert.shift.i = shl nuw i32 %v51.sroa.4.0.insert.ext.i, 24
  %v51.sroa.3.0.insert.ext.i = zext i8 %v104 to i32
  %v51.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.3.0.insert.ext.i, 16
  %v51.sroa.2.0.insert.ext.i = zext i8 %v99 to i32
  %v51.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.2.0.insert.ext.i, 8
  %v51.sroa.0.0.insert.ext.i = zext i8 %v94 to i32
  %v51.sroa.3.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.shift.i, %v51.sroa.0.0.insert.ext.i
  %v51.sroa.2.0.insert.insert.i = or disjoint i32 %v51.sroa.3.0.insert.insert.i, %v51.sroa.3.0.insert.shift.i
  %v51.sroa.0.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.insert.i, %v51.sroa.4.0.insert.shift.i
  %v59.sroa.4.0.insert.ext.i = zext i8 %v129 to i32
  %v59.sroa.4.0.insert.shift.i = shl nuw i32 %v59.sroa.4.0.insert.ext.i, 24
  %v59.sroa.3.0.insert.ext.i = zext i8 %v124 to i32
  %v59.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.3.0.insert.ext.i, 16
  %v59.sroa.2.0.insert.ext.i = zext i8 %v119 to i32
  %v59.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.2.0.insert.ext.i, 8
  %v59.sroa.0.0.insert.ext.i = zext i8 %v114 to i32
  %v59.sroa.3.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.shift.i, %v59.sroa.0.0.insert.ext.i
  %v59.sroa.2.0.insert.insert.i = or disjoint i32 %v59.sroa.3.0.insert.insert.i, %v59.sroa.3.0.insert.shift.i
  %v59.sroa.0.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.insert.i, %v59.sroa.4.0.insert.shift.i
  %v65.i = lshr i32 %v59.sroa.0.0.insert.insert.i, 4
  %v66.i = and i32 %v65.i, 252645135
  %14 = lshr i32 %v51.sroa.0.0.insert.insert.i, 2
  %v73.i = and i32 %14, 808464432
  %v81.i = and i32 %v59.sroa.0.0.insert.insert.i, 252645135
  %15 = lshr i32 %v43.sroa.0.0.insert.insert.i, 2
  %v88.i = and i32 %15, 808464432
  %v94.i = and i32 %v43.sroa.0.0.insert.insert.i, 1061109567
  %v98.sroa.2.0.extract.shift.i = lshr i32 %v94.i, 8
  %v98.sroa.4.0.extract.shift.i = lshr i32 %v94.i, 24
  %v98.sroa.3.0.extract.shift.i = lshr i32 %v94.i, 16
  %v98.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v98.sroa.4.0.extract.shift.i to i8
  %v98.sroa.3.0.extract.trunc.i = trunc i32 %v98.sroa.3.0.extract.shift.i to i8
  %16 = insertelement <4 x i32> poison, i32 %v94.i, i64 0
  %17 = insertelement <4 x i32> %16, i32 %v98.sroa.2.0.extract.shift.i, i64 1
  %18 = trunc <4 x i32> %17 to <4 x i8>
  %19 = insertelement <4 x i8> %18, i8 %v98.sroa.3.0.extract.trunc.i, i64 2
  %20 = insertelement <4 x i8> %19, i8 %v98.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %20, ptr %v21, align 4
  %v89.i = or disjoint i32 %v81.i, %v88.i
  %v102.sroa.2.0.extract.shift.i = lshr i32 %v89.i, 8
  %v102.sroa.4.0.extract.shift.i = lshr i32 %v89.i, 24
  %v102.sroa.3.0.extract.shift.i = lshr i32 %v89.i, 16
  %v102.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v102.sroa.4.0.extract.shift.i to i8
  %v102.sroa.3.0.extract.trunc.i = trunc i32 %v102.sroa.3.0.extract.shift.i to i8
  %21 = insertelement <4 x i32> poison, i32 %v89.i, i64 0
  %22 = insertelement <4 x i32> %21, i32 %v102.sroa.2.0.extract.shift.i, i64 1
  %23 = trunc <4 x i32> %22 to <4 x i8>
  %24 = insertelement <4 x i8> %23, i8 %v102.sroa.3.0.extract.trunc.i, i64 2
  %25 = insertelement <4 x i8> %24, i8 %v102.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %25, ptr %v131.fca.4.gep, align 4
  %v78.i = and i32 %v51.sroa.0.0.insert.insert.i, 1061109567
  %v106.sroa.2.0.extract.shift.i = lshr i32 %v78.i, 8
  %v106.sroa.4.0.extract.shift.i = lshr i32 %v78.i, 24
  %v106.sroa.3.0.extract.shift.i = lshr i32 %v78.i, 16
  %v106.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v106.sroa.4.0.extract.shift.i to i8
  %v106.sroa.3.0.extract.trunc.i = trunc i32 %v106.sroa.3.0.extract.shift.i to i8
  %26 = insertelement <4 x i32> poison, i32 %v78.i, i64 0
  %27 = insertelement <4 x i32> %26, i32 %v106.sroa.2.0.extract.shift.i, i64 1
  %28 = trunc <4 x i32> %27 to <4 x i8>
  %29 = insertelement <4 x i8> %28, i8 %v106.sroa.3.0.extract.trunc.i, i64 2
  %30 = insertelement <4 x i8> %29, i8 %v106.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %30, ptr %v22, align 4
  %v74.i = or disjoint i32 %v66.i, %v73.i
  %v110.sroa.2.0.extract.shift.i = lshr i32 %v74.i, 8
  %v110.sroa.4.0.extract.shift.i = lshr i32 %v74.i, 24
  %v110.sroa.3.0.extract.shift.i = lshr i32 %v74.i, 16
  %v110.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v110.sroa.4.0.extract.shift.i to i8
  %v110.sroa.3.0.extract.trunc.i = trunc i32 %v110.sroa.3.0.extract.shift.i to i8
  %31 = insertelement <4 x i32> poison, i32 %v74.i, i64 0
  %32 = insertelement <4 x i32> %31, i32 %v110.sroa.2.0.extract.shift.i, i64 1
  %33 = trunc <4 x i32> %32 to <4 x i8>
  %34 = insertelement <4 x i8> %33, i8 %v110.sroa.3.0.extract.trunc.i, i64 2
  %35 = insertelement <4 x i8> %34, i8 %v110.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %35, ptr %v132.fca.4.gep, align 4
  %v133 = add nuw i64 %v37, 16
  %invariant.gep125 = getelementptr inbounds nuw float, ptr %v2, i64 %2
  br label %bb28.preheader

bb28.preheader:                                   ; preds = %bb24, %bb32
  %indvars.iv92 = phi i64 [ %9, %bb24 ], [ %indvars.iv.next93, %bb32 ]
  %indvars.iv = phi i64 [ %2, %bb24 ], [ %indvars.iv.next, %bb32 ]
  %v13778 = phi i64 [ 0, %bb24 ], [ %v160, %bb32 ]
  %v13677 = phi float [ 0.000000e+00, %bb24 ], [ %v159, %bb32 ]
  %umax = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv)
  %36 = add i64 %umax, %indvars.iv92
  %.not = icmp ult i64 %36, 32
  br i1 %.not, label %bb81, label %bb28.preheader.split

bb28.preheader.split:                             ; preds = %bb28.preheader
  %.idx = shl nuw nsw i64 %v13778, 7
  %gep = getelementptr inbounds nuw i8, ptr %invariant.gep125, i64 %.idx
  br label %bb29

bb29:                                             ; preds = %bb29, %bb28.preheader.split
  %v14176 = phi i64 [ 0, %bb28.preheader.split ], [ %v153.3, %bb29 ]
  %v14075 = phi float [ 0.000000e+00, %bb28.preheader.split ], [ %v152.3, %bb29 ]
  %gep124 = getelementptr inbounds nuw float, ptr %gep, i64 %v14176
  %v151 = load float, ptr %gep124, align 4
  %v152 = fadd contract float %v14075, %v151
  %37 = getelementptr inbounds nuw float, ptr %gep, i64 %v14176
  %gep124.1 = getelementptr inbounds nuw i8, ptr %37, i64 4
  %v151.1 = load float, ptr %gep124.1, align 4
  %v152.1 = fadd contract float %v152, %v151.1
  %38 = getelementptr inbounds nuw float, ptr %gep, i64 %v14176
  %gep124.2 = getelementptr inbounds nuw i8, ptr %38, i64 8
  %v151.2 = load float, ptr %gep124.2, align 4
  %v152.2 = fadd contract float %v152.1, %v151.2
  %39 = getelementptr inbounds nuw float, ptr %gep, i64 %v14176
  %gep124.3 = getelementptr inbounds nuw i8, ptr %39, i64 12
  %v151.3 = load float, ptr %gep124.3, align 4
  %v152.3 = fadd contract float %v152.2, %v151.3
  %v153.3 = add nuw nsw i64 %v14176, 4
  %exitcond.3 = icmp eq i64 %v153.3, 32
  br i1 %exitcond.3, label %bb32, label %bb29

bb32:                                             ; preds = %bb29
  %v155 = getelementptr inbounds nuw i8, ptr %v22, i64 %v13778
  %v156 = load i8, ptr %v155, align 1
  %v157 = uitofp i8 %v156 to float
  %v158 = fmul contract float %v152.3, %v157
  %v159 = fadd contract float %v13677, %v158
  %v160 = add nuw nsw i64 %v13778, 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 32
  %indvars.iv.next93 = add nsw i64 %indvars.iv92, -32
  %exitcond94 = icmp eq i64 %v160, 8
  br i1 %exitcond94, label %bb33, label %bb28.preheader

bb33:                                             ; preds = %bb32
  %v161 = fmul contract float %v159, %v55.i18
  %v162 = fsub contract float %v3188, %v161
  %invariant.gep135 = getelementptr inbounds nuw float, ptr %v2, i64 %2
  %invariant.gep133 = getelementptr inbounds nuw float, ptr %v2, i64 %2
  br label %bb35

bb35:                                             ; preds = %bb33, %bb47
  %indvars.iv112 = phi i64 [ %4, %bb33 ], [ %indvars.iv.next113, %bb47 ]
  %indvars.iv109 = phi i64 [ %3, %bb33 ], [ %indvars.iv.next110, %bb47 ]
  %indvars.iv103 = phi i64 [ %9, %bb33 ], [ %indvars.iv.next104, %bb47 ]
  %indvars.iv100 = phi i64 [ %2, %bb33 ], [ %indvars.iv.next101, %bb47 ]
  %indvars.iv98 = phi i64 [ %8, %bb33 ], [ %indvars.iv.next99, %bb47 ]
  %indvars.iv95 = phi i64 [ %6, %bb33 ], [ %indvars.iv.next96, %bb47 ]
  %v16686 = phi i64 [ 0, %bb33 ], [ %v233, %bb47 ]
  %v16585 = phi i64 [ 0, %bb33 ], [ %v232, %bb47 ]
  %v16484 = phi i64 [ 0, %bb33 ], [ %v205, %bb47 ]
  %v16383 = phi float [ %v162, %bb33 ], [ %v231, %bb47 ]
  %umax108 = tail call i64 @llvm.umax.i64(i64 %v1, i64 %indvars.iv95)
  %40 = add i64 %umax108, %indvars.iv98
  %umax111 = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv109)
  %41 = add i64 %umax111, %indvars.iv112
  %.fr = freeze i64 %41
  %umin114 = tail call i64 @llvm.umin.i64(i64 %.fr, i64 %40)
  %umin115 = tail call i64 @llvm.umin.i64(i64 %umin114, i64 31)
  %v169 = shl nuw nsw i64 %v16686, 5
  %v170 = add nuw i64 %v133, %v169
  %umax102 = tail call i64 @llvm.umax.i64(i64 %v3, i64 %indvars.iv100)
  %42 = add i64 %umax102, %indvars.iv103
  %.fr118 = freeze i64 %42
  %umin105 = tail call i64 @llvm.umin.i64(i64 %.fr118, i64 %40)
  %umin106 = tail call i64 @llvm.umin.i64(i64 %umin105, i64 31)
  %v172 = getelementptr inbounds nuw i8, ptr %v21, i64 %v16484
  %v173 = load i8, ptr %v172, align 2
  %v174 = uitofp i8 %v173 to float
  %.not119 = icmp eq i64 %40, %umin106
  br i1 %.not119, label %bb84, label %bb36.split

bb36.split:                                       ; preds = %bb35
  %.not120 = icmp eq i64 %.fr118, %umin106
  br i1 %.not120, label %bb85, label %bb36.split.split

bb36.split.split:                                 ; preds = %bb36.split
  %invariant.gep = getelementptr i8, ptr %v0, i64 %v170
  %gep136 = getelementptr inbounds nuw float, ptr %invariant.gep135, i64 %v16585
  br label %bb38

bb38:                                             ; preds = %bb38, %bb36.split.split
  %v17780 = phi i64 [ 0, %bb36.split.split ], [ %v196.1, %bb38 ]
  %v17679 = phi float [ 0.000000e+00, %bb36.split.split ], [ %v195.1, %bb38 ]
  %gep126 = getelementptr i8, ptr %invariant.gep, i64 %v17780
  %v184 = load i8, ptr %gep126, align 1
  %v185 = and i8 %v184, 15
  %v186 = uitofp nneg i8 %v185 to float
  %gep130 = getelementptr inbounds nuw float, ptr %gep136, i64 %v17780
  %v193 = load float, ptr %gep130, align 4
  %v194 = fmul contract float %v193, %v186
  %v195 = fadd contract float %v17679, %v194
  %v196 = or disjoint i64 %v17780, 1
  %gep126.1 = getelementptr i8, ptr %invariant.gep, i64 %v196
  %v184.1 = load i8, ptr %gep126.1, align 1
  %v185.1 = and i8 %v184.1, 15
  %v186.1 = uitofp nneg i8 %v185.1 to float
  %gep130.1 = getelementptr inbounds nuw float, ptr %gep136, i64 %v196
  %v193.1 = load float, ptr %gep130.1, align 4
  %v194.1 = fmul contract float %v193.1, %v186.1
  %v195.1 = fadd contract float %v195, %v194.1
  %v196.1 = add nuw nsw i64 %v17780, 2
  %exitcond107.1 = icmp eq i64 %v196.1, 32
  br i1 %exitcond107.1, label %bb41, label %bb38

bb41:                                             ; preds = %bb38
  %v197 = fmul contract float %v55.i, %v174
  %v198 = fmul contract float %v197, %v195.1
  %v199 = fadd contract float %v16383, %v198
  %v202 = getelementptr inbounds nuw i8, ptr %v172, i64 1
  %v203 = load i8, ptr %v202, align 1
  %v204 = uitofp i8 %v203 to float
  %v205 = add nuw nsw i64 %v16484, 2
  %.not121 = icmp eq i64 %40, %umin115
  br i1 %.not121, label %bb87, label %bb41.split

bb41.split:                                       ; preds = %bb41
  %.not122 = icmp eq i64 %.fr, %umin115
  br i1 %.not122, label %bb88, label %bb41.split.split

bb41.split.split:                                 ; preds = %bb41.split
  %invariant.gep131 = getelementptr i8, ptr %v0, i64 %v170
  br label %bb44

bb44:                                             ; preds = %bb44, %bb41.split.split
  %v20782 = phi i64 [ 0, %bb41.split.split ], [ %v228.1, %bb44 ]
  %v20681 = phi float [ 0.000000e+00, %bb41.split.split ], [ %v227.1, %bb44 ]
  %gep132 = getelementptr i8, ptr %invariant.gep131, i64 %v20782
  %v214 = load i8, ptr %gep132, align 1
  %v217 = lshr i8 %v214, 4
  %v218 = uitofp nneg i8 %v217 to float
  %gep134 = getelementptr inbounds nuw float, ptr %invariant.gep133, i64 %v20782
  %43 = getelementptr inbounds nuw i8, ptr %gep134, i64 128
  %v224 = getelementptr inbounds nuw float, ptr %43, i64 %v16585
  %v225 = load float, ptr %v224, align 4
  %v226 = fmul contract float %v225, %v218
  %v227 = fadd contract float %v20681, %v226
  %v228 = or disjoint i64 %v20782, 1
  %gep132.1 = getelementptr i8, ptr %invariant.gep131, i64 %v228
  %v214.1 = load i8, ptr %gep132.1, align 1
  %v217.1 = lshr i8 %v214.1, 4
  %v218.1 = uitofp nneg i8 %v217.1 to float
  %gep134.1 = getelementptr inbounds nuw float, ptr %invariant.gep133, i64 %v228
  %44 = getelementptr inbounds nuw i8, ptr %gep134.1, i64 128
  %v224.1 = getelementptr inbounds nuw float, ptr %44, i64 %v16585
  %v225.1 = load float, ptr %v224.1, align 4
  %v226.1 = fmul contract float %v225.1, %v218.1
  %v227.1 = fadd contract float %v227, %v226.1
  %v228.1 = add nuw nsw i64 %v20782, 2
  %exitcond116.1 = icmp eq i64 %v228.1, 32
  br i1 %exitcond116.1, label %bb47, label %bb44

bb47:                                             ; preds = %bb44
  %v229 = fmul contract float %v55.i, %v204
  %v230 = fmul contract float %v229, %v227.1
  %v231 = fadd contract float %v199, %v230
  %v232 = add nuw nsw i64 %v16585, 64
  %v233 = add nuw nsw i64 %v16686, 1
  %indvars.iv.next96 = add nuw i64 %indvars.iv95, 32
  %indvars.iv.next99 = add i64 %indvars.iv98, -32
  %indvars.iv.next101 = add nuw nsw i64 %indvars.iv100, 64
  %indvars.iv.next104 = add nsw i64 %indvars.iv103, -64
  %indvars.iv.next110 = add nuw nsw i64 %indvars.iv109, 64
  %indvars.iv.next113 = add nsw i64 %indvars.iv112, -64
  %exitcond117 = icmp eq i64 %v233, 4
  br i1 %exitcond117, label %bb48, label %bb35

bb48:                                             ; preds = %bb47
  %v234 = add i32 %v3289, 32
  %v33.not = icmp ult i32 %v234, %v5
  br i1 %v33.not, label %bb6, label %bb49

bb49:                                             ; preds = %bb48, %bb4
  %v31.lcssa = phi float [ 0.000000e+00, %bb4 ], [ %v231, %bb48 ]
  %v235 = zext nneg i32 %v23 to i64
  %v236 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_13, i64 %v235
  store float %v31.lcssa, ptr addrspace(3) %v236, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v241.not = icmp samesign ult i32 %v23, 16
  br i1 %v241.not, label %bb54, label %bb58

bb54:                                             ; preds = %bb49
  %45 = zext nneg i32 %v23 to i64
  %46 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_13, i64 %45
  %v246 = getelementptr inbounds nuw i8, ptr addrspace(3) %46, i64 64
  %v247 = load float, ptr addrspace(3) %v246, align 4
  %v249 = load float, ptr addrspace(3) %v236, align 4
  %v250 = fadd contract float %v247, %v249
  store float %v250, ptr addrspace(3) %v236, align 4
  br label %bb58

bb58:                                             ; preds = %bb49, %bb54
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v241.not.1 = icmp samesign ult i32 %v23, 8
  br i1 %v241.not.1, label %bb54.1, label %bb58.1

bb54.1:                                           ; preds = %bb58
  %47 = zext nneg i32 %v23 to i64
  %48 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_13, i64 %47
  %v246.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %48, i64 32
  %v247.1 = load float, ptr addrspace(3) %v246.1, align 4
  %v249.1 = load float, ptr addrspace(3) %v236, align 4
  %v250.1 = fadd contract float %v247.1, %v249.1
  store float %v250.1, ptr addrspace(3) %v236, align 4
  br label %bb58.1

bb58.1:                                           ; preds = %bb54.1, %bb58
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v241.not.2 = icmp samesign ult i32 %v23, 4
  br i1 %v241.not.2, label %bb54.2, label %bb58.2

bb54.2:                                           ; preds = %bb58.1
  %49 = zext nneg i32 %v23 to i64
  %50 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_13, i64 %49
  %v246.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %50, i64 16
  %v247.2 = load float, ptr addrspace(3) %v246.2, align 4
  %v249.2 = load float, ptr addrspace(3) %v236, align 4
  %v250.2 = fadd contract float %v247.2, %v249.2
  store float %v250.2, ptr addrspace(3) %v236, align 4
  br label %bb58.2

bb58.2:                                           ; preds = %bb54.2, %bb58.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v241.not.3 = icmp samesign ult i32 %v23, 2
  br i1 %v241.not.3, label %bb54.3, label %bb58.3

bb54.3:                                           ; preds = %bb58.2
  %51 = zext nneg i32 %v23 to i64
  %52 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_13, i64 %51
  %v246.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %52, i64 8
  %v247.3 = load float, ptr addrspace(3) %v246.3, align 4
  %v249.3 = load float, ptr addrspace(3) %v236, align 4
  %v250.3 = fadd contract float %v247.3, %v249.3
  store float %v250.3, ptr addrspace(3) %v236, align 4
  br label %bb58.3

bb58.3:                                           ; preds = %bb54.3, %bb58.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v241.not.4 = icmp eq i32 %v23, 0
  br i1 %v241.not.4, label %bb54.4, label %bb58.4

bb54.4:                                           ; preds = %bb58.3
  %v247.4 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_13, i64 4), align 4
  %v249.4 = load float, ptr addrspace(3) %v236, align 4
  %v250.4 = fadd contract float %v247.4, %v249.4
  store float %v250.4, ptr addrspace(3) %v236, align 4
  br label %bb58.4

bb58.4:                                           ; preds = %bb54.4, %bb58.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v253 = icmp eq i32 %v23, 0
  br i1 %v253, label %bb61, label %bb64

bb61:                                             ; preds = %bb58.4
  %v258 = getelementptr inbounds nuw float, ptr %v6, i64 %v27
  %v256 = load float, ptr addrspace(3) @__shared_mem_13, align 4
  store float %v256, ptr %v258, align 4
  br label %bb64

bb64:                                             ; preds = %bb58.4, %bb61, %entry
  ret void

bb65:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb66:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb67:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb68:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb69:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit41
  tail call void @llvm.trap() #19
  unreachable

bb70:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb71:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb72:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb73:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb74:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb75:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb76:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb77:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb78:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable

bb79:                                             ; preds = %bb22
  tail call void @llvm.trap() #19
  unreachable

bb80:                                             ; preds = %bb23
  tail call void @llvm.trap() #19
  unreachable

bb81:                                             ; preds = %bb28.preheader
  tail call void @llvm.trap() #19
  unreachable

bb84:                                             ; preds = %bb35
  tail call void @llvm.trap() #19
  unreachable

bb85:                                             ; preds = %bb36.split
  tail call void @llvm.trap() #19
  unreachable

bb87:                                             ; preds = %bb41
  tail call void @llvm.trap() #19
  unreachable

bb88:                                             ; preds = %bb41.split
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q4k_q8_gemv_multiwarp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr writeonly captures(none) %v12, i64 %v13) #6 {
entry:
  %v35 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v36 = zext nneg i32 %v35 to i64
  %v37 = and i64 %v36, 31
  %v40 = lshr i64 %v36, 5
  %v41 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %0 = lshr i32 %v41, 5
  %v45 = zext nneg i32 %0 to i64
  %v46 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v47 = zext nneg i32 %v46 to i64
  %v48 = zext i32 %v9 to i64
  %v49 = zext i32 %v11 to i64
  %v50 = mul nuw i64 %v49, %v48
  %v51.not = icmp ugt i64 %v50, %v47
  %v53.not = icmp samesign ult i64 %v40, %v45
  %or.cond = select i1 %v51.not, i1 %v53.not, i1 false
  br i1 %or.cond, label %bb7, label %bb37

bb7:                                              ; preds = %entry
  %v55.not = icmp eq i32 %v9, 0
  br i1 %v55.not, label %bb40, label %bb8

bb8:                                              ; preds = %bb7
  %v9.frozen = freeze i32 %v9
  %v589 = udiv i32 %v46, %v9.frozen
  %v58.zext = zext nneg i32 %v589 to i64
  %v59 = zext i32 %v10 to i64
  %v61 = mul nuw nsw i64 %v58.zext, %v59
  %v65.not13 = icmp samesign ult i64 %v40, %v59
  br i1 %v65.not13, label %bb10.lr.ph, label %bb16.preheader

bb10.lr.ph:                                       ; preds = %bb8
  %v60 = mul nuw nsw i64 %v59, 144
  %1 = mul i32 %v589, %v9.frozen
  %v578.decomposed = sub i32 %v46, %1
  %v57.zext = zext nneg i32 %v578.decomposed to i64
  %v67 = mul i64 %v60, %v57.zext
  %v75 = shl nuw nsw i64 %v37, 2
  %v11.i.i = and i64 %v75, 28
  %2 = getelementptr i8, ptr %v0, i64 %v67
  %3 = trunc nuw nsw i64 %v75 to i32
  %4 = lshr i32 %3, 3
  %5 = and i32 %4, 4
  %v76.1 = or disjoint i64 %v75, 128
  %v551.i.1 = lshr i64 %v76.1, 5
  %v19.i.i.1 = add nsw i64 %v551.i.1, -4
  %6 = shl nuw nsw i64 %v36, 1
  %v14.i.i = and i64 %6, 32
  %v551.i = lshr i64 %v37, 3
  %7 = lshr exact i64 %v76.1, 1
  %v14.i.i.1 = and i64 %7, 96
  br label %bb10

bb16.preheader:                                   ; preds = %bb2.i46.i.1, %bb8
  %v63.lcssa = phi float [ 0.000000e+00, %bb8 ], [ %v119.1, %bb2.i46.i.1 ]
  %v126 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v63.lcssa, i32 16, i32 31) #19
  %v154 = fadd contract float %v63.lcssa, %v126
  %v126.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v154, i32 8, i32 31) #19
  %v154.1 = fadd contract float %v154, %v126.1
  %v126.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v154.1, i32 4, i32 31) #19
  %v154.2 = fadd contract float %v154.1, %v126.2
  %v126.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v154.2, i32 2, i32 31) #19
  %v154.3 = fadd contract float %v154.2, %v126.3
  %v126.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v154.3, i32 1, i32 31) #19
  %v127.not = icmp eq i64 %v37, 0
  br i1 %v127.not, label %bb19, label %bb21

bb10:                                             ; preds = %bb10.lr.ph, %bb2.i46.i.1
  %v6415 = phi i64 [ %v40, %bb10.lr.ph ], [ %v121, %bb2.i46.i.1 ]
  %v6314 = phi float [ 0.000000e+00, %bb10.lr.ph ], [ %v119.1, %bb2.i46.i.1 ]
  %v68 = mul i64 %v6415, 144
  %v621 = add i64 %v6415, %v61
  %v78 = shl i64 %v621, 8
  %8 = getelementptr i8, ptr %2, i64 %v68
  %9 = getelementptr i8, ptr %8, i64 16
  %invariant.gep = getelementptr i8, ptr %9, i64 %v11.i.i
  %v27.i = load i8, ptr %8, align 1
  %v31.i = getelementptr i8, ptr %8, i64 1
  %v32.i = load i8, ptr %v31.i, align 1
  %v36.sroa.2.0.insert.ext.i = zext i8 %v32.i to i16
  %v36.sroa.2.0.insert.shift.i = shl nuw i16 %v36.sroa.2.0.insert.ext.i, 8
  %v36.sroa.0.0.insert.ext.i = zext i8 %v27.i to i16
  %v4.i.i = lshr i16 %v36.sroa.2.0.insert.ext.i, 7
  %v6.i.i = zext nneg i16 %v4.i.i to i32
  %v9.i.i = lshr i16 %v36.sroa.2.0.insert.ext.i, 2
  %v10.i.i = and i16 %v9.i.i, 31
  %v36.sroa.2.0.insert.shift.masked.i = and i16 %v36.sroa.2.0.insert.shift.i, 768
  %v12.i.i = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i, %v36.sroa.0.0.insert.ext.i
  %v13.i.i = zext nneg i16 %v12.i.i to i32
  %v42.i = getelementptr i8, ptr %8, i64 2
  %v43.i = load i8, ptr %v42.i, align 1
  %v47.i = getelementptr i8, ptr %8, i64 3
  %v48.i = load i8, ptr %v47.i, align 1
  %v52.sroa.2.0.insert.ext.i = zext i8 %v48.i to i16
  %v52.sroa.2.0.insert.shift.i = shl nuw i16 %v52.sroa.2.0.insert.ext.i, 8
  %v52.sroa.0.0.insert.ext.i = zext i8 %v43.i to i16
  %v4.i5.i = lshr i16 %v52.sroa.2.0.insert.ext.i, 7
  %v6.i6.i = zext nneg i16 %v4.i5.i to i32
  %v9.i7.i = lshr i16 %v52.sroa.2.0.insert.ext.i, 2
  %v10.i8.i = and i16 %v9.i7.i, 31
  %v52.sroa.2.0.insert.shift.masked.i = and i16 %v52.sroa.2.0.insert.shift.i, 768
  %v12.i9.i = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i, %v52.sroa.0.0.insert.ext.i
  %v13.i10.i = zext nneg i16 %v12.i9.i to i32
  %v38.i.i = shl nuw i32 %v6.i.i, 31
  %v41.i.i = shl nuw nsw i32 %v13.i.i, 13
  %v39.i.i = or disjoint i32 %v41.i.i, %v38.i.i
  %v42.i.i = or disjoint i32 %v39.i.i, 2139095040
  %v15.i.i = icmp eq i16 %v12.i.i, 0
  %v13.masked.numleadingzeros.i.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i, i1 true)
  %v13.masked.leadingonepos.i.i = xor i32 %v13.masked.numleadingzeros.i.i, 31
  %bb5.tripcount.i.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i
  %v23.i.i = shl nuw nsw i32 %v13.i.i, %bb5.tripcount.i.i
  %reass.sub.i = or disjoint i32 %v38.i.i, 1124073472
  %10 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i, 23
  %v31.i.i = sub nuw nsw i32 %reass.sub.i, %10
  %v25.i.i = shl i32 %v23.i.i, 13
  %v33.i2.i = and i32 %v25.i.i, 8380416
  %v34.i3.i = or disjoint i32 %v31.i.i, %v33.i2.i
  %11 = add nuw nsw i16 %v10.i.i, 112
  %v46.i4.i = zext nneg i16 %11 to i32
  %v48.i.i = shl nuw nsw i32 %v46.i4.i, 23
  %v49.i.i = or disjoint i32 %v48.i.i, %v38.i.i
  %v52.i.i = or disjoint i32 %v49.i.i, %v41.i.i
  %v38.i12.i = shl nuw i32 %v6.i6.i, 31
  %v41.i13.i = shl nuw nsw i32 %v13.i10.i, 13
  %v39.i14.i = or disjoint i32 %v41.i13.i, %v38.i12.i
  %v42.i15.i = or disjoint i32 %v39.i14.i, 2139095040
  %v15.i19.i = icmp eq i16 %v12.i9.i, 0
  %v13.masked.numleadingzeros.i21.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i, i1 true)
  %v13.masked.leadingonepos.i22.i = xor i32 %v13.masked.numleadingzeros.i21.i, 31
  %bb5.tripcount.i23.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i
  %v23.i24.i = shl nuw nsw i32 %v13.i10.i, %bb5.tripcount.i23.i
  %reass.sub63.i = or disjoint i32 %v38.i12.i, 1124073472
  %12 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i, 23
  %v31.i27.i = sub nuw nsw i32 %reass.sub63.i, %12
  %v25.i28.i = shl i32 %v23.i24.i, 13
  %v33.i29.i = and i32 %v25.i28.i, 8380416
  %v34.i30.i = or disjoint i32 %v31.i27.i, %v33.i29.i
  %13 = add nuw nsw i16 %v10.i8.i, 112
  %v46.i35.i = zext nneg i16 %13 to i32
  %v48.i36.i = shl nuw nsw i32 %v46.i35.i, 23
  %v49.i37.i = or disjoint i32 %v48.i36.i, %v38.i12.i
  %v52.i39.i = or disjoint i32 %v49.i37.i, %v41.i13.i
  %14 = getelementptr i8, ptr %8, i64 8
  %15 = getelementptr i8, ptr %8, i64 4
  %16 = getelementptr i8, ptr %8, i64 12
  %v79 = or disjoint i64 %v75, %v78
  %v81 = getelementptr inbounds i8, ptr %v2, i64 %v79
  %v33.sroa.0.0.copyload = load i32, ptr %v81, align 1
  %sext = shl i32 %v33.sroa.0.0.copyload, 24
  %v103 = ashr exact i32 %sext, 24
  %17 = shl i32 %v33.sroa.0.0.copyload, 16
  %v104 = ashr i32 %17, 24
  %18 = shl i32 %v33.sroa.0.0.copyload, 8
  %v106 = ashr i32 %18, 24
  %v108 = ashr i32 %v33.sroa.0.0.copyload, 24
  %v105 = add nsw i32 %v104, %v108
  %v107 = add nsw i32 %v105, %v103
  %v109 = add nsw i32 %v107, %v106
  %v1105 = lshr i64 %v79, 5
  %v114 = getelementptr inbounds nuw float, ptr %v4, i64 %v1105
  %v115 = load float, ptr %v114, align 4
  %gep = getelementptr i8, ptr %invariant.gep, i64 %v14.i.i
  %v9.sroa.0.0.copyload.i.i = load i32, ptr %gep, align 1
  %v32.v.i.i = lshr i32 %v9.sroa.0.0.copyload.i.i, %5
  %v32.i.i = and i32 %v32.v.i.i, 252645135
  %v33.i.i = xor i32 %v32.i.i, 134744072
  %v34.i.i = and i32 %v33.i.i, 134744072
  %19 = mul nuw i32 %v34.i.i, 30
  %v46.i.i = add nuw nsw i32 %19, %v33.i.i
  %v20.i = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i, i32 %v33.sroa.0.0.copyload, i32 0) #19
  switch i16 %v10.i.i, label %bb10.i.i [
    i16 0, label %bb1.i.i
    i16 31, label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  ]

bb1.i.i:                                          ; preds = %bb10
  %v17.i.i.v34.i3.i = select i1 %v15.i.i, i32 %v38.i.i, i32 %v34.i3.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

bb10.i.i:                                         ; preds = %bb10
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

cuda_kernels__oxide_kernels__f16_to_f32.exit.i:   ; preds = %bb10, %bb1.i.i, %bb10.i.i
  %v54.i.i = phi i32 [ %v52.i.i, %bb10.i.i ], [ %v17.i.i.v34.i3.i, %bb1.i.i ], [ %v42.i.i, %bb10 ]
  switch i16 %v10.i8.i, label %bb10.i33.i [
    i16 0, label %bb1.i18.i
    i16 31, label %bb1.i42.i
  ]

bb1.i18.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  %v17.i32.i.v34.i30.i = select i1 %v15.i19.i, i32 %v38.i12.i, i32 %v34.i30.i
  br label %bb1.i42.i

bb10.i33.i:                                       ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  br label %bb1.i42.i

bb1.i42.i:                                        ; preds = %bb10.i33.i, %bb1.i18.i, %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  %v54.i16.i = phi i32 [ %v52.i39.i, %bb10.i33.i ], [ %v17.i32.i.v34.i30.i, %bb1.i18.i ], [ %v42.i15.i, %cuda_kernels__oxide_kernels__f16_to_f32.exit.i ]
  %v16.i.i = getelementptr i8, ptr %15, i64 %v551.i
  %v17.i43.i = load i8, ptr %v16.i.i, align 1
  %v18.i44.i = and i8 %v17.i43.i, 63
  %v16.i54.i = getelementptr i8, ptr %14, i64 %v551.i
  %v17.i55.i = load i8, ptr %v16.i54.i, align 1
  %v18.i56.i = and i8 %v17.i55.i, 63
  %v55.i.i = bitcast i32 %v54.i.i to float
  %v59.i = uitofp nneg i8 %v18.i44.i to float
  %v60.i = fmul contract float %v55.i.i, %v59.i
  %v21.i = shl nsw i32 %v109, 3
  %v22.i = add i32 %v20.i, %v21.i
  %v61.i = sitofp i32 %v22.i to float
  %v62.i = fmul contract float %v60.i, %v61.i
  %v55.i17.i = bitcast i32 %v54.i16.i to float
  %v66.i = uitofp nneg i8 %v18.i56.i to float
  %v67.i = fmul contract float %v55.i17.i, %v66.i
  %v68.i = sitofp i32 %v109 to float
  %v69.i = fmul contract float %v67.i, %v68.i
  %v70.i = fsub contract float %v62.i, %v69.i
  %v71.i = fmul contract float %v115, %v70.i
  %v119 = fadd contract float %v6314, %v71.i
  %v79.1 = or disjoint i64 %v76.1, %v78
  %v81.1 = getelementptr inbounds i8, ptr %v2, i64 %v79.1
  %v33.sroa.0.0.copyload.1 = load i32, ptr %v81.1, align 1
  %sext.1 = shl i32 %v33.sroa.0.0.copyload.1, 24
  %v103.1 = ashr exact i32 %sext.1, 24
  %20 = shl i32 %v33.sroa.0.0.copyload.1, 16
  %v104.1 = ashr i32 %20, 24
  %21 = shl i32 %v33.sroa.0.0.copyload.1, 8
  %v106.1 = ashr i32 %21, 24
  %v108.1 = ashr i32 %v33.sroa.0.0.copyload.1, 24
  %v105.1 = add nsw i32 %v104.1, %v108.1
  %v107.1 = add nsw i32 %v105.1, %v103.1
  %v109.1 = add nsw i32 %v107.1, %v106.1
  %v1105.1 = lshr i64 %v79.1, 5
  %v114.1 = getelementptr inbounds nuw float, ptr %v4, i64 %v1105.1
  %v115.1 = load float, ptr %v114.1, align 4
  %gep.1 = getelementptr i8, ptr %invariant.gep, i64 %v14.i.i.1
  %v9.sroa.0.0.copyload.i.i.1 = load i32, ptr %gep.1, align 1
  %v32.v.i.i.1 = lshr i32 %v9.sroa.0.0.copyload.i.i.1, %5
  %v32.i.i.1 = and i32 %v32.v.i.i.1, 252645135
  %v33.i.i.1 = xor i32 %v32.i.i.1, 134744072
  %v34.i.i.1 = and i32 %v33.i.i.1, 134744072
  %22 = mul nuw i32 %v34.i.i.1, 30
  %v46.i.i.1 = add nuw nsw i32 %22, %v33.i.i.1
  %v20.i.1 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i.1, i32 %v33.sroa.0.0.copyload.1, i32 0) #19
  switch i16 %v10.i.i, label %bb10.i.i.1 [
    i16 0, label %bb1.i.i.1
    i16 31, label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1
  ]

bb1.i.i.1:                                        ; preds = %bb1.i42.i
  %v17.i.i.v34.i3.i.1 = select i1 %v15.i.i, i32 %v38.i.i, i32 %v34.i3.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1

bb10.i.i.1:                                       ; preds = %bb1.i42.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1

cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1: ; preds = %bb10.i.i.1, %bb1.i.i.1, %bb1.i42.i
  %v54.i.i.1 = phi i32 [ %v52.i.i, %bb10.i.i.1 ], [ %v17.i.i.v34.i3.i.1, %bb1.i.i.1 ], [ %v42.i.i, %bb1.i42.i ]
  switch i16 %v10.i8.i, label %bb10.i33.i.1 [
    i16 0, label %bb1.i18.i.1
    i16 31, label %bb2.i46.i.1
  ]

bb1.i18.i.1:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1
  %v17.i32.i.v34.i30.i.1 = select i1 %v15.i19.i, i32 %v38.i12.i, i32 %v34.i30.i
  br label %bb2.i46.i.1

bb10.i33.i.1:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1
  br label %bb2.i46.i.1

bb2.i46.i.1:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1, %bb1.i18.i.1, %bb10.i33.i.1
  %v54.i16.i.1 = phi i32 [ %v52.i39.i, %bb10.i33.i.1 ], [ %v17.i32.i.v34.i30.i.1, %bb1.i18.i.1 ], [ %v42.i15.i, %cuda_kernels__oxide_kernels__f16_to_f32.exit.i.1 ]
  %v25.i47.i.1 = getelementptr i8, ptr %14, i64 %v551.i.1
  %v26.i.i.1 = load i8, ptr %v25.i47.i.1, align 1
  %v27.i48.i.1 = and i8 %v26.i.i.1, 15
  %v32.i49.i.1 = getelementptr i8, ptr %8, i64 %v551.i.1
  %v33.i50.i.1 = load i8, ptr %v32.i49.i.1, align 1
  %23 = lshr i8 %v33.i50.i.1, 2
  %v39.i51.i.1 = and i8 %23, 48
  %v40.i.i.1 = or disjoint i8 %v39.i51.i.1, %v27.i48.i.1
  %v34.i60.i.1 = getelementptr i8, ptr %14, i64 %v19.i.i.1
  %v35.i.i.1 = load i8, ptr %v34.i60.i.1, align 1
  %24 = lshr i8 %v35.i.i.1, 2
  %v41.i61.i.1 = and i8 %24, 48
  %v25.i58.i.1 = getelementptr i8, ptr %16, i64 %v19.i.i.1
  %v26.i59.i.1 = load i8, ptr %v25.i58.i.1, align 1
  %v29.i.i.1 = lshr i8 %v26.i59.i.1, 4
  %v42.i62.i.1 = or disjoint i8 %v41.i61.i.1, %v29.i.i.1
  %v55.i.i.1 = bitcast i32 %v54.i.i.1 to float
  %v59.i.1 = uitofp nneg i8 %v40.i.i.1 to float
  %v60.i.1 = fmul contract float %v55.i.i.1, %v59.i.1
  %v21.i.1 = shl nsw i32 %v109.1, 3
  %v22.i.1 = add i32 %v20.i.1, %v21.i.1
  %v61.i.1 = sitofp i32 %v22.i.1 to float
  %v62.i.1 = fmul contract float %v60.i.1, %v61.i.1
  %v55.i17.i.1 = bitcast i32 %v54.i16.i.1 to float
  %v66.i.1 = uitofp nneg i8 %v42.i62.i.1 to float
  %v67.i.1 = fmul contract float %v55.i17.i.1, %v66.i.1
  %v68.i.1 = sitofp i32 %v109.1 to float
  %v69.i.1 = fmul contract float %v67.i.1, %v68.i.1
  %v70.i.1 = fsub contract float %v62.i.1, %v69.i.1
  %v71.i.1 = fmul contract float %v115.1, %v70.i.1
  %v119.1 = fadd contract float %v119, %v71.i.1
  %v121 = add i64 %v6415, %v45
  %v65.not = icmp ult i64 %v121, %v59
  br i1 %v65.not, label %bb10, label %bb16.preheader

bb19:                                             ; preds = %bb16.preheader
  %v154.4 = fadd contract float %v154.3, %v126.4
  %v129 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_5, i64 %v40
  store float %v154.4, ptr addrspace(3) %v129, align 4
  br label %bb21

bb21:                                             ; preds = %bb19, %bb16.preheader
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v131 = icmp eq i64 %v40, 0
  br i1 %v131, label %bb23, label %bb37

bb23:                                             ; preds = %bb21
  %v132.not = icmp samesign ult i64 %v37, %v45
  br i1 %v132.not, label %bb24, label %bb27

bb24:                                             ; preds = %bb23
  %v135 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_5, i64 %v37
  %v136 = load float, ptr addrspace(3) %v135, align 4
  br label %bb27

bb27:                                             ; preds = %bb23, %bb24
  %v137 = phi float [ %v136, %bb24 ], [ 0.000000e+00, %bb23 ]
  %v142 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v137, i32 16, i32 31) #19
  %v156 = fadd contract float %v137, %v142
  %v142.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v156, i32 8, i32 31) #19
  %v156.1 = fadd contract float %v156, %v142.1
  %v142.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v156.1, i32 4, i32 31) #19
  %v156.2 = fadd contract float %v156.1, %v142.2
  %v142.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v156.2, i32 2, i32 31) #19
  %v156.3 = fadd contract float %v156.2, %v142.3
  %v142.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v156.3, i32 1, i32 31) #19
  %v156.4 = fadd contract float %v156.3, %v142.4
  br i1 %v127.not, label %bb31, label %bb37

bb31:                                             ; preds = %bb27
  %v144 = icmp eq i32 %v8, 0
  br i1 %v144, label %bb34, label %bb32

bb32:                                             ; preds = %bb31
  %v148 = getelementptr inbounds nuw float, ptr %v6, i64 %v47
  %v149 = load float, ptr %v148, align 4
  br label %bb34

bb34:                                             ; preds = %bb31, %bb32
  %v150 = phi float [ %v149, %bb32 ], [ 0.000000e+00, %bb31 ]
  %v152 = getelementptr inbounds nuw float, ptr %v12, i64 %v47
  %v153 = fadd contract float %v156.4, %v150
  store float %v153, ptr %v152, align 4
  br label %bb37

bb37:                                             ; preds = %bb21, %bb34, %bb27, %entry
  ret void

bb40:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @q4k_q8_gemv_warp4(ptr readonly %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, ptr writeonly captures(none) %v9, i64 %v10) #3 {
entry:
  %v28 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v30 = zext i32 %v6 to i64
  %v31 = add nuw nsw i64 %v30, 3
  %v321 = lshr i64 %v31, 2
  %v33 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v34 = zext nneg i32 %v33 to i64
  %v35 = zext i32 %v8 to i64
  %v36 = mul nuw nsw i64 %v321, %v35
  %v37.not = icmp samesign ugt i64 %v36, %v34
  br i1 %v37.not, label %bb4, label %bb40

bb4:                                              ; preds = %entry
  %v39.not = icmp eq i64 %v321, 0
  br i1 %v39.not, label %bb45, label %bb5

bb5:                                              ; preds = %bb4
  %v41.rhs.trunc = trunc nuw nsw i64 %v321 to i32
  %v41.rhs.trunc.frozen = freeze i32 %v41.rhs.trunc
  %v41428 = udiv i32 %v33, %v41.rhs.trunc.frozen
  %v41.zext = zext nneg i32 %v41428 to i64
  %0 = mul i32 %v41428, %v41.rhs.trunc.frozen
  %v42429.decomposed = sub i32 %v33, %0
  %v42.zext = zext nneg i32 %v42429.decomposed to i64
  %v43 = shl nuw nsw i64 %v42.zext, 2
  %v44 = zext i32 %v7 to i64
  %v46 = mul nuw nsw i64 %v41.zext, %v44
  %v53.not448.not = icmp eq i32 %v7, 0
  br i1 %v53.not448.not, label %bb24.preheader, label %bb8.preheader.lr.ph

bb8.preheader.lr.ph:                              ; preds = %bb5
  %v45 = mul nuw nsw i64 %v44, 144
  %1 = shl nuw nsw i32 %v28, 2
  %v63 = zext nneg i32 %1 to i64
  %v104.not = icmp samesign ult i64 %v43, %v30
  %v114 = or disjoint i64 %v43, 1
  %v115.not = icmp samesign ult i64 %v114, %v30
  %v125 = or disjoint i64 %v43, 2
  %v126.not = icmp samesign ult i64 %v125, %v30
  %v136 = or disjoint i64 %v43, 3
  %v137.not = icmp samesign ult i64 %v136, %v30
  %v106 = mul i64 %v43, %v45
  %v11.i.i = and i64 %v63, 28
  %2 = getelementptr i8, ptr %v0, i64 %v106
  %3 = lshr i32 %v28, 1
  %4 = and i32 %3, 4
  %v117 = mul i64 %v114, %v45
  %5 = getelementptr i8, ptr %v0, i64 %v117
  %v128 = mul i64 %v125, %v45
  %6 = getelementptr i8, ptr %v0, i64 %v128
  %v139 = mul i64 %v136, %v45
  %7 = getelementptr i8, ptr %v0, i64 %v139
  br label %bb8.preheader

bb8.preheader:                                    ; preds = %bb8.preheader.lr.ph, %bb22
  %v52453 = phi i64 [ 0, %bb8.preheader.lr.ph ], [ %v148, %bb22 ]
  %v51452 = phi float [ 0.000000e+00, %bb8.preheader.lr.ph ], [ %v146, %bb22 ]
  %v50451 = phi float [ 0.000000e+00, %bb8.preheader.lr.ph ], [ %v135, %bb22 ]
  %v49450 = phi float [ 0.000000e+00, %bb8.preheader.lr.ph ], [ %v124, %bb22 ]
  %v48449 = phi float [ 0.000000e+00, %bb8.preheader.lr.ph ], [ %v113, %bb22 ]
  %v472 = add nuw nsw i64 %v52453, %v46
  %v66 = shl i64 %v472, 8
  %v107 = mul nuw nsw i64 %v52453, 144
  %8 = getelementptr i8, ptr %2, i64 %v107
  %9 = getelementptr i8, ptr %8, i64 16
  %invariant.gep = getelementptr i8, ptr %9, i64 %v11.i.i
  %v31.i = getelementptr i8, ptr %8, i64 1
  %v42.i = getelementptr i8, ptr %8, i64 2
  %v47.i = getelementptr i8, ptr %8, i64 3
  %10 = getelementptr i8, ptr %8, i64 4
  %11 = getelementptr i8, ptr %8, i64 8
  %12 = getelementptr i8, ptr %8, i64 12
  %13 = getelementptr i8, ptr %5, i64 %v107
  %14 = getelementptr i8, ptr %13, i64 16
  %invariant.gep442 = getelementptr i8, ptr %14, i64 %v11.i.i
  %v31.i19 = getelementptr i8, ptr %13, i64 1
  %v42.i38 = getelementptr i8, ptr %13, i64 2
  %v47.i40 = getelementptr i8, ptr %13, i64 3
  %15 = getelementptr i8, ptr %13, i64 4
  %16 = getelementptr i8, ptr %13, i64 8
  %17 = getelementptr i8, ptr %13, i64 12
  %18 = getelementptr i8, ptr %6, i64 %v107
  %19 = getelementptr i8, ptr %18, i64 16
  %invariant.gep444 = getelementptr i8, ptr %19, i64 %v11.i.i
  %v31.i159 = getelementptr i8, ptr %18, i64 1
  %v42.i178 = getelementptr i8, ptr %18, i64 2
  %v47.i180 = getelementptr i8, ptr %18, i64 3
  %20 = getelementptr i8, ptr %18, i64 4
  %21 = getelementptr i8, ptr %18, i64 8
  %22 = getelementptr i8, ptr %18, i64 12
  %23 = getelementptr i8, ptr %7, i64 %v107
  %24 = getelementptr i8, ptr %23, i64 16
  %invariant.gep446 = getelementptr i8, ptr %24, i64 %v11.i.i
  %v31.i299 = getelementptr i8, ptr %23, i64 1
  %v42.i318 = getelementptr i8, ptr %23, i64 2
  %v47.i320 = getelementptr i8, ptr %23, i64 3
  %25 = getelementptr i8, ptr %23, i64 4
  %26 = getelementptr i8, ptr %23, i64 8
  %27 = getelementptr i8, ptr %23, i64 12
  br label %bb9

bb24.preheader:                                   ; preds = %bb22, %bb5
  %v48.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v113, %bb22 ]
  %v49.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v124, %bb22 ]
  %v50.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v135, %bb22 ]
  %v51.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v146, %bb22 ]
  %v156 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v48.lcssa, i32 16, i32 31) #19
  %v182 = fadd contract float %v48.lcssa, %v156
  %v183 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v49.lcssa, i32 16, i32 31) #19
  %v184 = fadd contract float %v49.lcssa, %v183
  %v185 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v50.lcssa, i32 16, i32 31) #19
  %v186 = fadd contract float %v50.lcssa, %v185
  %v187 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v51.lcssa, i32 16, i32 31) #19
  %v188 = fadd contract float %v51.lcssa, %v187
  %v156.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182, i32 8, i32 31) #19
  %v182.1 = fadd contract float %v182, %v156.1
  %v183.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184, i32 8, i32 31) #19
  %v184.1 = fadd contract float %v184, %v183.1
  %v185.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186, i32 8, i32 31) #19
  %v186.1 = fadd contract float %v186, %v185.1
  %v187.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188, i32 8, i32 31) #19
  %v188.1 = fadd contract float %v188, %v187.1
  %v156.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182.1, i32 4, i32 31) #19
  %v182.2 = fadd contract float %v182.1, %v156.2
  %v183.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184.1, i32 4, i32 31) #19
  %v184.2 = fadd contract float %v184.1, %v183.2
  %v185.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186.1, i32 4, i32 31) #19
  %v186.2 = fadd contract float %v186.1, %v185.2
  %v187.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188.1, i32 4, i32 31) #19
  %v188.2 = fadd contract float %v188.1, %v187.2
  %v156.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182.2, i32 2, i32 31) #19
  %v182.3 = fadd contract float %v182.2, %v156.3
  %v183.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184.2, i32 2, i32 31) #19
  %v184.3 = fadd contract float %v184.2, %v183.3
  %v185.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186.2, i32 2, i32 31) #19
  %v186.3 = fadd contract float %v186.2, %v185.3
  %v187.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188.2, i32 2, i32 31) #19
  %v188.3 = fadd contract float %v188.2, %v187.3
  %v156.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182.3, i32 1, i32 31) #19
  %v182.4 = fadd contract float %v182.3, %v156.4
  %v183.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184.3, i32 1, i32 31) #19
  %v184.4 = fadd contract float %v184.3, %v183.4
  %v185.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186.3, i32 1, i32 31) #19
  %v186.4 = fadd contract float %v186.3, %v185.4
  %v187.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188.3, i32 1, i32 31) #19
  %v188.4 = fadd contract float %v188.3, %v187.4
  %v157 = icmp eq i32 %v28, 0
  br i1 %v157, label %bb27, label %bb40

bb9:                                              ; preds = %bb8.preheader, %bb21
  %v60.not = phi i1 [ true, %bb8.preheader ], [ false, %bb21 ]
  %v59441 = phi i64 [ 0, %bb8.preheader ], [ 128, %bb21 ]
  %v58440 = phi float [ %v51452, %bb8.preheader ], [ %v146, %bb21 ]
  %v57439 = phi float [ %v50451, %bb8.preheader ], [ %v135, %bb21 ]
  %v56438 = phi float [ %v49450, %bb8.preheader ], [ %v124, %bb21 ]
  %v55437 = phi float [ %v48449, %bb8.preheader ], [ %v113, %bb21 ]
  %v64 = add nuw nsw i64 %v59441, %v63
  %v67 = add i64 %v64, %v66
  %v69 = getelementptr inbounds i8, ptr %v2, i64 %v67
  %v26.sroa.0.0.copyload = load i32, ptr %v69, align 1
  %sext = shl i32 %v26.sroa.0.0.copyload, 24
  %v91 = ashr exact i32 %sext, 24
  %28 = shl i32 %v26.sroa.0.0.copyload, 16
  %v92 = ashr i32 %28, 24
  %29 = shl i32 %v26.sroa.0.0.copyload, 8
  %v94 = ashr i32 %29, 24
  %v96 = ashr i32 %v26.sroa.0.0.copyload, 24
  %v93 = add nsw i32 %v92, %v96
  %v95 = add nsw i32 %v93, %v91
  %v97 = add nsw i32 %v95, %v94
  %v986 = lshr i64 %v67, 5
  %v102 = getelementptr inbounds nuw float, ptr %v4, i64 %v986
  %v103 = load float, ptr %v102, align 4
  br i1 %v104.not, label %bb10, label %bb12

bb10:                                             ; preds = %bb9
  %30 = lshr exact i64 %v64, 1
  %v14.i.i = and i64 %30, 4064
  %gep = getelementptr i8, ptr %invariant.gep, i64 %v14.i.i
  %v9.sroa.0.0.copyload.i.i = load i32, ptr %gep, align 1
  %v32.v.i.i = lshr i32 %v9.sroa.0.0.copyload.i.i, %4
  %v32.i.i = and i32 %v32.v.i.i, 252645135
  %v33.i.i = xor i32 %v32.i.i, 134744072
  %v34.i.i = and i32 %v33.i.i, 134744072
  %31 = mul nuw i32 %v34.i.i, 30
  %v46.i.i = add nuw nsw i32 %31, %v33.i.i
  %v20.i = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i, i32 %v26.sroa.0.0.copyload, i32 0) #19
  %v27.i = load i8, ptr %8, align 1
  %v32.i = load i8, ptr %v31.i, align 1
  %v36.sroa.2.0.insert.ext.i = zext i8 %v32.i to i16
  %v36.sroa.2.0.insert.shift.i = shl nuw i16 %v36.sroa.2.0.insert.ext.i, 8
  %v36.sroa.0.0.insert.ext.i = zext i8 %v27.i to i16
  %v4.i.i = lshr i16 %v36.sroa.2.0.insert.ext.i, 7
  %v6.i.i = zext nneg i16 %v4.i.i to i32
  %v9.i.i = lshr i16 %v36.sroa.2.0.insert.ext.i, 2
  %v10.i.i = and i16 %v9.i.i, 31
  %v36.sroa.2.0.insert.shift.masked.i = and i16 %v36.sroa.2.0.insert.shift.i, 768
  %v12.i.i = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i, %v36.sroa.0.0.insert.ext.i
  %v13.i.i = zext nneg i16 %v12.i.i to i32
  switch i16 %v10.i.i, label %bb10.i.i [
    i16 0, label %bb1.i.i
    i16 31, label %bb9.i.i
  ]

bb1.i.i:                                          ; preds = %bb10
  %v15.i.i = icmp eq i16 %v12.i.i, 0
  br i1 %v15.i.i, label %bb2.i.i, label %bb6.i.i

bb2.i.i:                                          ; preds = %bb1.i.i
  %v17.i.i = shl nuw i32 %v6.i.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

bb6.i.i:                                          ; preds = %bb1.i.i
  %v13.masked.numleadingzeros.i.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i, i1 true)
  %v13.masked.leadingonepos.i.i = xor i32 %v13.masked.numleadingzeros.i.i, 31
  %bb5.tripcount.i.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i
  %v23.i.i = shl nuw nsw i32 %v13.i.i, %bb5.tripcount.i.i
  %v27.i.i = shl nuw i32 %v6.i.i, 31
  %reass.sub.i = or disjoint i32 %v27.i.i, 1124073472
  %32 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i, 23
  %v31.i.i = sub nuw nsw i32 %reass.sub.i, %32
  %v25.i.i = shl i32 %v23.i.i, 13
  %v33.i2.i = and i32 %v25.i.i, 8380416
  %v34.i3.i = or disjoint i32 %v31.i.i, %v33.i2.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

bb9.i.i:                                          ; preds = %bb10
  %v38.i.i = shl nuw i32 %v6.i.i, 31
  %v41.i.i = shl nuw nsw i32 %v13.i.i, 13
  %v39.i.i = or disjoint i32 %v41.i.i, %v38.i.i
  %v42.i.i = or disjoint i32 %v39.i.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

bb10.i.i:                                         ; preds = %bb10
  %v44.i.i = shl nuw i32 %v6.i.i, 31
  %33 = add nuw nsw i16 %v10.i.i, 112
  %v46.i4.i = zext nneg i16 %33 to i32
  %v48.i.i = shl nuw nsw i32 %v46.i4.i, 23
  %v49.i.i = or disjoint i32 %v48.i.i, %v44.i.i
  %v51.i.i = shl nuw nsw i32 %v13.i.i, 13
  %v52.i.i = or disjoint i32 %v49.i.i, %v51.i.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i

cuda_kernels__oxide_kernels__f16_to_f32.exit.i:   ; preds = %bb10.i.i, %bb9.i.i, %bb6.i.i, %bb2.i.i
  %v54.i.i = phi i32 [ %v34.i3.i, %bb6.i.i ], [ %v17.i.i, %bb2.i.i ], [ %v42.i.i, %bb9.i.i ], [ %v52.i.i, %bb10.i.i ]
  %v43.i = load i8, ptr %v42.i, align 1
  %v48.i = load i8, ptr %v47.i, align 1
  %v52.sroa.2.0.insert.ext.i = zext i8 %v48.i to i16
  %v52.sroa.2.0.insert.shift.i = shl nuw i16 %v52.sroa.2.0.insert.ext.i, 8
  %v52.sroa.0.0.insert.ext.i = zext i8 %v43.i to i16
  %v4.i5.i = lshr i16 %v52.sroa.2.0.insert.ext.i, 7
  %v6.i6.i = zext nneg i16 %v4.i5.i to i32
  %v9.i7.i = lshr i16 %v52.sroa.2.0.insert.ext.i, 2
  %v10.i8.i = and i16 %v9.i7.i, 31
  %v52.sroa.2.0.insert.shift.masked.i = and i16 %v52.sroa.2.0.insert.shift.i, 768
  %v12.i9.i = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i, %v52.sroa.0.0.insert.ext.i
  %v13.i10.i = zext nneg i16 %v12.i9.i to i32
  switch i16 %v10.i8.i, label %bb10.i33.i [
    i16 0, label %bb1.i18.i
    i16 31, label %bb9.i11.i
  ]

bb1.i18.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  %v15.i19.i = icmp eq i16 %v12.i9.i, 0
  br i1 %v15.i19.i, label %bb2.i31.i, label %bb6.i20.i

bb2.i31.i:                                        ; preds = %bb1.i18.i
  %v17.i32.i = shl nuw i32 %v6.i6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i

bb6.i20.i:                                        ; preds = %bb1.i18.i
  %v13.masked.numleadingzeros.i21.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i, i1 true)
  %v13.masked.leadingonepos.i22.i = xor i32 %v13.masked.numleadingzeros.i21.i, 31
  %bb5.tripcount.i23.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i
  %v23.i24.i = shl nuw nsw i32 %v13.i10.i, %bb5.tripcount.i23.i
  %v27.i25.i = shl nuw i32 %v6.i6.i, 31
  %reass.sub63.i = or disjoint i32 %v27.i25.i, 1124073472
  %34 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i, 23
  %v31.i27.i = sub nuw nsw i32 %reass.sub63.i, %34
  %v25.i28.i = shl i32 %v23.i24.i, 13
  %v33.i29.i = and i32 %v25.i28.i, 8380416
  %v34.i30.i = or disjoint i32 %v31.i27.i, %v33.i29.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i

bb9.i11.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  %v38.i12.i = shl nuw i32 %v6.i6.i, 31
  %v41.i13.i = shl nuw nsw i32 %v13.i10.i, 13
  %v39.i14.i = or disjoint i32 %v41.i13.i, %v38.i12.i
  %v42.i15.i = or disjoint i32 %v39.i14.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i

bb10.i33.i:                                       ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i
  %v44.i34.i = shl nuw i32 %v6.i6.i, 31
  %35 = add nuw nsw i16 %v10.i8.i, 112
  %v46.i35.i = zext nneg i16 %35 to i32
  %v48.i36.i = shl nuw nsw i32 %v46.i35.i, 23
  %v49.i37.i = or disjoint i32 %v48.i36.i, %v44.i34.i
  %v51.i38.i = shl nuw nsw i32 %v13.i10.i, 13
  %v52.i39.i = or disjoint i32 %v49.i37.i, %v51.i38.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i

cuda_kernels__oxide_kernels__f16_to_f32.exit40.i: ; preds = %bb10.i33.i, %bb9.i11.i, %bb6.i20.i, %bb2.i31.i
  %v54.i16.i = phi i32 [ %v34.i30.i, %bb6.i20.i ], [ %v17.i32.i, %bb2.i31.i ], [ %v42.i15.i, %bb9.i11.i ], [ %v52.i39.i, %bb10.i33.i ]
  %v551.i = lshr i64 %v64, 5
  %v9.i41.i = icmp samesign ugt i64 %v64, 127
  br i1 %v9.i41.i, label %bb2.i46.i, label %bb1.i42.i

bb1.i42.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i
  %v16.i.i = getelementptr i8, ptr %10, i64 %v551.i
  %v17.i43.i = load i8, ptr %v16.i.i, align 1
  %v18.i44.i = and i8 %v17.i43.i, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i

bb2.i46.i:                                        ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i
  %v25.i47.i = getelementptr i8, ptr %11, i64 %v551.i
  %v26.i.i = load i8, ptr %v25.i47.i, align 1
  %v27.i48.i = and i8 %v26.i.i, 15
  %v32.i49.i = getelementptr i8, ptr %8, i64 %v551.i
  %v33.i50.i = load i8, ptr %v32.i49.i, align 1
  %36 = lshr i8 %v33.i50.i, 2
  %v39.i51.i = and i8 %36, 48
  %v40.i.i = or disjoint i8 %v39.i51.i, %v27.i48.i
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i

cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i: ; preds = %bb2.i46.i, %bb1.i42.i
  %v41.i45.i = phi i8 [ %v18.i44.i, %bb1.i42.i ], [ %v40.i.i, %bb2.i46.i ]
  br i1 %v9.i41.i, label %bb2.i57.i, label %bb1.i53.i

bb1.i53.i:                                        ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i
  %v16.i54.i = getelementptr i8, ptr %11, i64 %v551.i
  %v17.i55.i = load i8, ptr %v16.i54.i, align 1
  %v18.i56.i = and i8 %v17.i55.i, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit

bb2.i57.i:                                        ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i
  %v19.i.i = add nsw i64 %v551.i, -4
  %v25.i58.i = getelementptr i8, ptr %12, i64 %v19.i.i
  %v26.i59.i = load i8, ptr %v25.i58.i, align 1
  %v29.i.i = lshr i8 %v26.i59.i, 4
  %v34.i60.i = getelementptr i8, ptr %11, i64 %v19.i.i
  %v35.i.i = load i8, ptr %v34.i60.i, align 1
  %37 = lshr i8 %v35.i.i, 2
  %v41.i61.i = and i8 %37, 48
  %v42.i62.i = or disjoint i8 %v41.i61.i, %v29.i.i
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit

cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit: ; preds = %bb1.i53.i, %bb2.i57.i
  %v43.i.i = phi i8 [ %v18.i56.i, %bb1.i53.i ], [ %v42.i62.i, %bb2.i57.i ]
  %v55.i.i = bitcast i32 %v54.i.i to float
  %v59.i = uitofp nneg i8 %v41.i45.i to float
  %v60.i = fmul contract float %v55.i.i, %v59.i
  %v21.i = shl nsw i32 %v97, 3
  %v22.i = add i32 %v20.i, %v21.i
  %v61.i = sitofp i32 %v22.i to float
  %v62.i = fmul contract float %v60.i, %v61.i
  %v55.i17.i = bitcast i32 %v54.i16.i to float
  %v66.i = uitofp nneg i8 %v43.i.i to float
  %v67.i = fmul contract float %v55.i17.i, %v66.i
  %v68.i = sitofp i32 %v97 to float
  %v69.i = fmul contract float %v67.i, %v68.i
  %v70.i = fsub contract float %v62.i, %v69.i
  %v71.i = fmul contract float %v103, %v70.i
  %v112 = fadd contract float %v55437, %v71.i
  br label %bb12

bb12:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit, %bb9
  %v113 = phi float [ %v55437, %bb9 ], [ %v112, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit ]
  br i1 %v115.not, label %bb13, label %bb15

bb13:                                             ; preds = %bb12
  %38 = lshr exact i64 %v64, 1
  %v14.i.i9 = and i64 %38, 4064
  %gep443 = getelementptr i8, ptr %invariant.gep442, i64 %v14.i.i9
  %v9.sroa.0.0.copyload.i.i11 = load i32, ptr %gep443, align 1
  %v32.v.i.i12 = lshr i32 %v9.sroa.0.0.copyload.i.i11, %4
  %v32.i.i13 = and i32 %v32.v.i.i12, 252645135
  %v33.i.i14 = xor i32 %v32.i.i13, 134744072
  %v34.i.i15 = and i32 %v33.i.i14, 134744072
  %39 = mul nuw i32 %v34.i.i15, 30
  %v46.i.i16 = add nuw nsw i32 %39, %v33.i.i14
  %v20.i17 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i16, i32 %v26.sroa.0.0.copyload, i32 0) #19
  %v27.i18 = load i8, ptr %13, align 1
  %v32.i20 = load i8, ptr %v31.i19, align 1
  %v36.sroa.2.0.insert.ext.i21 = zext i8 %v32.i20 to i16
  %v36.sroa.2.0.insert.shift.i22 = shl nuw i16 %v36.sroa.2.0.insert.ext.i21, 8
  %v36.sroa.0.0.insert.ext.i23 = zext i8 %v27.i18 to i16
  %v4.i.i24 = lshr i16 %v36.sroa.2.0.insert.ext.i21, 7
  %v6.i.i25 = zext nneg i16 %v4.i.i24 to i32
  %v9.i.i26 = lshr i16 %v36.sroa.2.0.insert.ext.i21, 2
  %v10.i.i27 = and i16 %v9.i.i26, 31
  %v36.sroa.2.0.insert.shift.masked.i28 = and i16 %v36.sroa.2.0.insert.shift.i22, 768
  %v12.i.i29 = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i28, %v36.sroa.0.0.insert.ext.i23
  %v13.i.i30 = zext nneg i16 %v12.i.i29 to i32
  switch i16 %v10.i.i27, label %bb10.i.i140 [
    i16 0, label %bb1.i.i125
    i16 31, label %bb9.i.i31
  ]

bb1.i.i125:                                       ; preds = %bb13
  %v15.i.i126 = icmp eq i16 %v12.i.i29, 0
  br i1 %v15.i.i126, label %bb2.i.i138, label %bb6.i.i127

bb2.i.i138:                                       ; preds = %bb1.i.i125
  %v17.i.i139 = shl nuw i32 %v6.i.i25, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36

bb6.i.i127:                                       ; preds = %bb1.i.i125
  %v13.masked.numleadingzeros.i.i128 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i30, i1 true)
  %v13.masked.leadingonepos.i.i129 = xor i32 %v13.masked.numleadingzeros.i.i128, 31
  %bb5.tripcount.i.i130 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i129
  %v23.i.i131 = shl nuw nsw i32 %v13.i.i30, %bb5.tripcount.i.i130
  %v27.i.i132 = shl nuw i32 %v6.i.i25, 31
  %reass.sub.i133 = or disjoint i32 %v27.i.i132, 1124073472
  %40 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i128, 23
  %v31.i.i134 = sub nuw nsw i32 %reass.sub.i133, %40
  %v25.i.i135 = shl i32 %v23.i.i131, 13
  %v33.i2.i136 = and i32 %v25.i.i135, 8380416
  %v34.i3.i137 = or disjoint i32 %v31.i.i134, %v33.i2.i136
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36

bb9.i.i31:                                        ; preds = %bb13
  %v38.i.i32 = shl nuw i32 %v6.i.i25, 31
  %v41.i.i33 = shl nuw nsw i32 %v13.i.i30, 13
  %v39.i.i34 = or disjoint i32 %v41.i.i33, %v38.i.i32
  %v42.i.i35 = or disjoint i32 %v39.i.i34, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36

bb10.i.i140:                                      ; preds = %bb13
  %v44.i.i141 = shl nuw i32 %v6.i.i25, 31
  %41 = add nuw nsw i16 %v10.i.i27, 112
  %v46.i4.i142 = zext nneg i16 %41 to i32
  %v48.i.i143 = shl nuw nsw i32 %v46.i4.i142, 23
  %v49.i.i144 = or disjoint i32 %v48.i.i143, %v44.i.i141
  %v51.i.i145 = shl nuw nsw i32 %v13.i.i30, 13
  %v52.i.i146 = or disjoint i32 %v49.i.i144, %v51.i.i145
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36

cuda_kernels__oxide_kernels__f16_to_f32.exit.i36: ; preds = %bb10.i.i140, %bb9.i.i31, %bb6.i.i127, %bb2.i.i138
  %v54.i.i37 = phi i32 [ %v34.i3.i137, %bb6.i.i127 ], [ %v17.i.i139, %bb2.i.i138 ], [ %v42.i.i35, %bb9.i.i31 ], [ %v52.i.i146, %bb10.i.i140 ]
  %v43.i39 = load i8, ptr %v42.i38, align 1
  %v48.i41 = load i8, ptr %v47.i40, align 1
  %v52.sroa.2.0.insert.ext.i42 = zext i8 %v48.i41 to i16
  %v52.sroa.2.0.insert.shift.i43 = shl nuw i16 %v52.sroa.2.0.insert.ext.i42, 8
  %v52.sroa.0.0.insert.ext.i44 = zext i8 %v43.i39 to i16
  %v4.i5.i45 = lshr i16 %v52.sroa.2.0.insert.ext.i42, 7
  %v6.i6.i46 = zext nneg i16 %v4.i5.i45 to i32
  %v9.i7.i47 = lshr i16 %v52.sroa.2.0.insert.ext.i42, 2
  %v10.i8.i48 = and i16 %v9.i7.i47, 31
  %v52.sroa.2.0.insert.shift.masked.i49 = and i16 %v52.sroa.2.0.insert.shift.i43, 768
  %v12.i9.i50 = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i49, %v52.sroa.0.0.insert.ext.i44
  %v13.i10.i51 = zext nneg i16 %v12.i9.i50 to i32
  switch i16 %v10.i8.i48, label %bb10.i33.i118 [
    i16 0, label %bb1.i18.i103
    i16 31, label %bb9.i11.i52
  ]

bb1.i18.i103:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36
  %v15.i19.i104 = icmp eq i16 %v12.i9.i50, 0
  br i1 %v15.i19.i104, label %bb2.i31.i116, label %bb6.i20.i105

bb2.i31.i116:                                     ; preds = %bb1.i18.i103
  %v17.i32.i117 = shl nuw i32 %v6.i6.i46, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57

bb6.i20.i105:                                     ; preds = %bb1.i18.i103
  %v13.masked.numleadingzeros.i21.i106 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i51, i1 true)
  %v13.masked.leadingonepos.i22.i107 = xor i32 %v13.masked.numleadingzeros.i21.i106, 31
  %bb5.tripcount.i23.i108 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i107
  %v23.i24.i109 = shl nuw nsw i32 %v13.i10.i51, %bb5.tripcount.i23.i108
  %v27.i25.i110 = shl nuw i32 %v6.i6.i46, 31
  %reass.sub63.i111 = or disjoint i32 %v27.i25.i110, 1124073472
  %42 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i106, 23
  %v31.i27.i112 = sub nuw nsw i32 %reass.sub63.i111, %42
  %v25.i28.i113 = shl i32 %v23.i24.i109, 13
  %v33.i29.i114 = and i32 %v25.i28.i113, 8380416
  %v34.i30.i115 = or disjoint i32 %v31.i27.i112, %v33.i29.i114
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57

bb9.i11.i52:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36
  %v38.i12.i53 = shl nuw i32 %v6.i6.i46, 31
  %v41.i13.i54 = shl nuw nsw i32 %v13.i10.i51, 13
  %v39.i14.i55 = or disjoint i32 %v41.i13.i54, %v38.i12.i53
  %v42.i15.i56 = or disjoint i32 %v39.i14.i55, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57

bb10.i33.i118:                                    ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i36
  %v44.i34.i119 = shl nuw i32 %v6.i6.i46, 31
  %43 = add nuw nsw i16 %v10.i8.i48, 112
  %v46.i35.i120 = zext nneg i16 %43 to i32
  %v48.i36.i121 = shl nuw nsw i32 %v46.i35.i120, 23
  %v49.i37.i122 = or disjoint i32 %v48.i36.i121, %v44.i34.i119
  %v51.i38.i123 = shl nuw nsw i32 %v13.i10.i51, 13
  %v52.i39.i124 = or disjoint i32 %v49.i37.i122, %v51.i38.i123
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57

cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57: ; preds = %bb10.i33.i118, %bb9.i11.i52, %bb6.i20.i105, %bb2.i31.i116
  %v54.i16.i58 = phi i32 [ %v34.i30.i115, %bb6.i20.i105 ], [ %v17.i32.i117, %bb2.i31.i116 ], [ %v42.i15.i56, %bb9.i11.i52 ], [ %v52.i39.i124, %bb10.i33.i118 ]
  %v551.i59 = lshr i64 %v64, 5
  %v9.i41.i60 = icmp samesign ugt i64 %v64, 127
  br i1 %v9.i41.i60, label %bb2.i46.i95, label %bb1.i42.i61

bb1.i42.i61:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57
  %v16.i.i62 = getelementptr i8, ptr %15, i64 %v551.i59
  %v17.i43.i63 = load i8, ptr %v16.i.i62, align 1
  %v18.i44.i64 = and i8 %v17.i43.i63, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i65

bb2.i46.i95:                                      ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i57
  %v25.i47.i96 = getelementptr i8, ptr %16, i64 %v551.i59
  %v26.i.i97 = load i8, ptr %v25.i47.i96, align 1
  %v27.i48.i98 = and i8 %v26.i.i97, 15
  %v32.i49.i99 = getelementptr i8, ptr %13, i64 %v551.i59
  %v33.i50.i100 = load i8, ptr %v32.i49.i99, align 1
  %44 = lshr i8 %v33.i50.i100, 2
  %v39.i51.i101 = and i8 %44, 48
  %v40.i.i102 = or disjoint i8 %v39.i51.i101, %v27.i48.i98
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i65

cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i65: ; preds = %bb2.i46.i95, %bb1.i42.i61
  %v41.i45.i66 = phi i8 [ %v18.i44.i64, %bb1.i42.i61 ], [ %v40.i.i102, %bb2.i46.i95 ]
  br i1 %v9.i41.i60, label %bb2.i57.i86, label %bb1.i53.i67

bb1.i53.i67:                                      ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i65
  %v16.i54.i68 = getelementptr i8, ptr %16, i64 %v551.i59
  %v17.i55.i69 = load i8, ptr %v16.i54.i68, align 1
  %v18.i56.i70 = and i8 %v17.i55.i69, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit147

bb2.i57.i86:                                      ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i65
  %v19.i.i87 = add nsw i64 %v551.i59, -4
  %v25.i58.i88 = getelementptr i8, ptr %17, i64 %v19.i.i87
  %v26.i59.i89 = load i8, ptr %v25.i58.i88, align 1
  %v29.i.i90 = lshr i8 %v26.i59.i89, 4
  %v34.i60.i91 = getelementptr i8, ptr %16, i64 %v19.i.i87
  %v35.i.i92 = load i8, ptr %v34.i60.i91, align 1
  %45 = lshr i8 %v35.i.i92, 2
  %v41.i61.i93 = and i8 %45, 48
  %v42.i62.i94 = or disjoint i8 %v41.i61.i93, %v29.i.i90
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit147

cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit147: ; preds = %bb1.i53.i67, %bb2.i57.i86
  %v43.i.i71 = phi i8 [ %v18.i56.i70, %bb1.i53.i67 ], [ %v42.i62.i94, %bb2.i57.i86 ]
  %v55.i.i72 = bitcast i32 %v54.i.i37 to float
  %v59.i73 = uitofp nneg i8 %v41.i45.i66 to float
  %v60.i74 = fmul contract float %v55.i.i72, %v59.i73
  %v21.i75 = shl nsw i32 %v97, 3
  %v22.i76 = add i32 %v20.i17, %v21.i75
  %v61.i77 = sitofp i32 %v22.i76 to float
  %v62.i78 = fmul contract float %v60.i74, %v61.i77
  %v55.i17.i79 = bitcast i32 %v54.i16.i58 to float
  %v66.i80 = uitofp nneg i8 %v43.i.i71 to float
  %v67.i81 = fmul contract float %v55.i17.i79, %v66.i80
  %v68.i82 = sitofp i32 %v97 to float
  %v69.i83 = fmul contract float %v67.i81, %v68.i82
  %v70.i84 = fsub contract float %v62.i78, %v69.i83
  %v71.i85 = fmul contract float %v103, %v70.i84
  %v123 = fadd contract float %v56438, %v71.i85
  br label %bb15

bb15:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit147, %bb12
  %v124 = phi float [ %v56438, %bb12 ], [ %v123, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit147 ]
  br i1 %v126.not, label %bb16, label %bb18

bb16:                                             ; preds = %bb15
  %46 = lshr exact i64 %v64, 1
  %v14.i.i149 = and i64 %46, 4064
  %gep445 = getelementptr i8, ptr %invariant.gep444, i64 %v14.i.i149
  %v9.sroa.0.0.copyload.i.i151 = load i32, ptr %gep445, align 1
  %v32.v.i.i152 = lshr i32 %v9.sroa.0.0.copyload.i.i151, %4
  %v32.i.i153 = and i32 %v32.v.i.i152, 252645135
  %v33.i.i154 = xor i32 %v32.i.i153, 134744072
  %v34.i.i155 = and i32 %v33.i.i154, 134744072
  %47 = mul nuw i32 %v34.i.i155, 30
  %v46.i.i156 = add nuw nsw i32 %47, %v33.i.i154
  %v20.i157 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i156, i32 %v26.sroa.0.0.copyload, i32 0) #19
  %v27.i158 = load i8, ptr %18, align 1
  %v32.i160 = load i8, ptr %v31.i159, align 1
  %v36.sroa.2.0.insert.ext.i161 = zext i8 %v32.i160 to i16
  %v36.sroa.2.0.insert.shift.i162 = shl nuw i16 %v36.sroa.2.0.insert.ext.i161, 8
  %v36.sroa.0.0.insert.ext.i163 = zext i8 %v27.i158 to i16
  %v4.i.i164 = lshr i16 %v36.sroa.2.0.insert.ext.i161, 7
  %v6.i.i165 = zext nneg i16 %v4.i.i164 to i32
  %v9.i.i166 = lshr i16 %v36.sroa.2.0.insert.ext.i161, 2
  %v10.i.i167 = and i16 %v9.i.i166, 31
  %v36.sroa.2.0.insert.shift.masked.i168 = and i16 %v36.sroa.2.0.insert.shift.i162, 768
  %v12.i.i169 = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i168, %v36.sroa.0.0.insert.ext.i163
  %v13.i.i170 = zext nneg i16 %v12.i.i169 to i32
  switch i16 %v10.i.i167, label %bb10.i.i280 [
    i16 0, label %bb1.i.i265
    i16 31, label %bb9.i.i171
  ]

bb1.i.i265:                                       ; preds = %bb16
  %v15.i.i266 = icmp eq i16 %v12.i.i169, 0
  br i1 %v15.i.i266, label %bb2.i.i278, label %bb6.i.i267

bb2.i.i278:                                       ; preds = %bb1.i.i265
  %v17.i.i279 = shl nuw i32 %v6.i.i165, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176

bb6.i.i267:                                       ; preds = %bb1.i.i265
  %v13.masked.numleadingzeros.i.i268 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i170, i1 true)
  %v13.masked.leadingonepos.i.i269 = xor i32 %v13.masked.numleadingzeros.i.i268, 31
  %bb5.tripcount.i.i270 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i269
  %v23.i.i271 = shl nuw nsw i32 %v13.i.i170, %bb5.tripcount.i.i270
  %v27.i.i272 = shl nuw i32 %v6.i.i165, 31
  %reass.sub.i273 = or disjoint i32 %v27.i.i272, 1124073472
  %48 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i268, 23
  %v31.i.i274 = sub nuw nsw i32 %reass.sub.i273, %48
  %v25.i.i275 = shl i32 %v23.i.i271, 13
  %v33.i2.i276 = and i32 %v25.i.i275, 8380416
  %v34.i3.i277 = or disjoint i32 %v31.i.i274, %v33.i2.i276
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176

bb9.i.i171:                                       ; preds = %bb16
  %v38.i.i172 = shl nuw i32 %v6.i.i165, 31
  %v41.i.i173 = shl nuw nsw i32 %v13.i.i170, 13
  %v39.i.i174 = or disjoint i32 %v41.i.i173, %v38.i.i172
  %v42.i.i175 = or disjoint i32 %v39.i.i174, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176

bb10.i.i280:                                      ; preds = %bb16
  %v44.i.i281 = shl nuw i32 %v6.i.i165, 31
  %49 = add nuw nsw i16 %v10.i.i167, 112
  %v46.i4.i282 = zext nneg i16 %49 to i32
  %v48.i.i283 = shl nuw nsw i32 %v46.i4.i282, 23
  %v49.i.i284 = or disjoint i32 %v48.i.i283, %v44.i.i281
  %v51.i.i285 = shl nuw nsw i32 %v13.i.i170, 13
  %v52.i.i286 = or disjoint i32 %v49.i.i284, %v51.i.i285
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176

cuda_kernels__oxide_kernels__f16_to_f32.exit.i176: ; preds = %bb10.i.i280, %bb9.i.i171, %bb6.i.i267, %bb2.i.i278
  %v54.i.i177 = phi i32 [ %v34.i3.i277, %bb6.i.i267 ], [ %v17.i.i279, %bb2.i.i278 ], [ %v42.i.i175, %bb9.i.i171 ], [ %v52.i.i286, %bb10.i.i280 ]
  %v43.i179 = load i8, ptr %v42.i178, align 1
  %v48.i181 = load i8, ptr %v47.i180, align 1
  %v52.sroa.2.0.insert.ext.i182 = zext i8 %v48.i181 to i16
  %v52.sroa.2.0.insert.shift.i183 = shl nuw i16 %v52.sroa.2.0.insert.ext.i182, 8
  %v52.sroa.0.0.insert.ext.i184 = zext i8 %v43.i179 to i16
  %v4.i5.i185 = lshr i16 %v52.sroa.2.0.insert.ext.i182, 7
  %v6.i6.i186 = zext nneg i16 %v4.i5.i185 to i32
  %v9.i7.i187 = lshr i16 %v52.sroa.2.0.insert.ext.i182, 2
  %v10.i8.i188 = and i16 %v9.i7.i187, 31
  %v52.sroa.2.0.insert.shift.masked.i189 = and i16 %v52.sroa.2.0.insert.shift.i183, 768
  %v12.i9.i190 = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i189, %v52.sroa.0.0.insert.ext.i184
  %v13.i10.i191 = zext nneg i16 %v12.i9.i190 to i32
  switch i16 %v10.i8.i188, label %bb10.i33.i258 [
    i16 0, label %bb1.i18.i243
    i16 31, label %bb9.i11.i192
  ]

bb1.i18.i243:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176
  %v15.i19.i244 = icmp eq i16 %v12.i9.i190, 0
  br i1 %v15.i19.i244, label %bb2.i31.i256, label %bb6.i20.i245

bb2.i31.i256:                                     ; preds = %bb1.i18.i243
  %v17.i32.i257 = shl nuw i32 %v6.i6.i186, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197

bb6.i20.i245:                                     ; preds = %bb1.i18.i243
  %v13.masked.numleadingzeros.i21.i246 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i191, i1 true)
  %v13.masked.leadingonepos.i22.i247 = xor i32 %v13.masked.numleadingzeros.i21.i246, 31
  %bb5.tripcount.i23.i248 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i247
  %v23.i24.i249 = shl nuw nsw i32 %v13.i10.i191, %bb5.tripcount.i23.i248
  %v27.i25.i250 = shl nuw i32 %v6.i6.i186, 31
  %reass.sub63.i251 = or disjoint i32 %v27.i25.i250, 1124073472
  %50 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i246, 23
  %v31.i27.i252 = sub nuw nsw i32 %reass.sub63.i251, %50
  %v25.i28.i253 = shl i32 %v23.i24.i249, 13
  %v33.i29.i254 = and i32 %v25.i28.i253, 8380416
  %v34.i30.i255 = or disjoint i32 %v31.i27.i252, %v33.i29.i254
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197

bb9.i11.i192:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176
  %v38.i12.i193 = shl nuw i32 %v6.i6.i186, 31
  %v41.i13.i194 = shl nuw nsw i32 %v13.i10.i191, 13
  %v39.i14.i195 = or disjoint i32 %v41.i13.i194, %v38.i12.i193
  %v42.i15.i196 = or disjoint i32 %v39.i14.i195, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197

bb10.i33.i258:                                    ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i176
  %v44.i34.i259 = shl nuw i32 %v6.i6.i186, 31
  %51 = add nuw nsw i16 %v10.i8.i188, 112
  %v46.i35.i260 = zext nneg i16 %51 to i32
  %v48.i36.i261 = shl nuw nsw i32 %v46.i35.i260, 23
  %v49.i37.i262 = or disjoint i32 %v48.i36.i261, %v44.i34.i259
  %v51.i38.i263 = shl nuw nsw i32 %v13.i10.i191, 13
  %v52.i39.i264 = or disjoint i32 %v49.i37.i262, %v51.i38.i263
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197

cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197: ; preds = %bb10.i33.i258, %bb9.i11.i192, %bb6.i20.i245, %bb2.i31.i256
  %v54.i16.i198 = phi i32 [ %v34.i30.i255, %bb6.i20.i245 ], [ %v17.i32.i257, %bb2.i31.i256 ], [ %v42.i15.i196, %bb9.i11.i192 ], [ %v52.i39.i264, %bb10.i33.i258 ]
  %v551.i199 = lshr i64 %v64, 5
  %v9.i41.i200 = icmp samesign ugt i64 %v64, 127
  br i1 %v9.i41.i200, label %bb2.i46.i235, label %bb1.i42.i201

bb1.i42.i201:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197
  %v16.i.i202 = getelementptr i8, ptr %20, i64 %v551.i199
  %v17.i43.i203 = load i8, ptr %v16.i.i202, align 1
  %v18.i44.i204 = and i8 %v17.i43.i203, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i205

bb2.i46.i235:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i197
  %v25.i47.i236 = getelementptr i8, ptr %21, i64 %v551.i199
  %v26.i.i237 = load i8, ptr %v25.i47.i236, align 1
  %v27.i48.i238 = and i8 %v26.i.i237, 15
  %v32.i49.i239 = getelementptr i8, ptr %18, i64 %v551.i199
  %v33.i50.i240 = load i8, ptr %v32.i49.i239, align 1
  %52 = lshr i8 %v33.i50.i240, 2
  %v39.i51.i241 = and i8 %52, 48
  %v40.i.i242 = or disjoint i8 %v39.i51.i241, %v27.i48.i238
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i205

cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i205: ; preds = %bb2.i46.i235, %bb1.i42.i201
  %v41.i45.i206 = phi i8 [ %v18.i44.i204, %bb1.i42.i201 ], [ %v40.i.i242, %bb2.i46.i235 ]
  br i1 %v9.i41.i200, label %bb2.i57.i226, label %bb1.i53.i207

bb1.i53.i207:                                     ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i205
  %v16.i54.i208 = getelementptr i8, ptr %21, i64 %v551.i199
  %v17.i55.i209 = load i8, ptr %v16.i54.i208, align 1
  %v18.i56.i210 = and i8 %v17.i55.i209, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit287

bb2.i57.i226:                                     ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i205
  %v19.i.i227 = add nsw i64 %v551.i199, -4
  %v25.i58.i228 = getelementptr i8, ptr %22, i64 %v19.i.i227
  %v26.i59.i229 = load i8, ptr %v25.i58.i228, align 1
  %v29.i.i230 = lshr i8 %v26.i59.i229, 4
  %v34.i60.i231 = getelementptr i8, ptr %21, i64 %v19.i.i227
  %v35.i.i232 = load i8, ptr %v34.i60.i231, align 1
  %53 = lshr i8 %v35.i.i232, 2
  %v41.i61.i233 = and i8 %53, 48
  %v42.i62.i234 = or disjoint i8 %v41.i61.i233, %v29.i.i230
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit287

cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit287: ; preds = %bb1.i53.i207, %bb2.i57.i226
  %v43.i.i211 = phi i8 [ %v18.i56.i210, %bb1.i53.i207 ], [ %v42.i62.i234, %bb2.i57.i226 ]
  %v55.i.i212 = bitcast i32 %v54.i.i177 to float
  %v59.i213 = uitofp nneg i8 %v41.i45.i206 to float
  %v60.i214 = fmul contract float %v55.i.i212, %v59.i213
  %v21.i215 = shl nsw i32 %v97, 3
  %v22.i216 = add i32 %v20.i157, %v21.i215
  %v61.i217 = sitofp i32 %v22.i216 to float
  %v62.i218 = fmul contract float %v60.i214, %v61.i217
  %v55.i17.i219 = bitcast i32 %v54.i16.i198 to float
  %v66.i220 = uitofp nneg i8 %v43.i.i211 to float
  %v67.i221 = fmul contract float %v55.i17.i219, %v66.i220
  %v68.i222 = sitofp i32 %v97 to float
  %v69.i223 = fmul contract float %v67.i221, %v68.i222
  %v70.i224 = fsub contract float %v62.i218, %v69.i223
  %v71.i225 = fmul contract float %v103, %v70.i224
  %v134 = fadd contract float %v57439, %v71.i225
  br label %bb18

bb18:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit287, %bb15
  %v135 = phi float [ %v57439, %bb15 ], [ %v134, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit287 ]
  br i1 %v137.not, label %bb19, label %bb21

bb19:                                             ; preds = %bb18
  %54 = lshr exact i64 %v64, 1
  %v14.i.i289 = and i64 %54, 4064
  %gep447 = getelementptr i8, ptr %invariant.gep446, i64 %v14.i.i289
  %v9.sroa.0.0.copyload.i.i291 = load i32, ptr %gep447, align 1
  %v32.v.i.i292 = lshr i32 %v9.sroa.0.0.copyload.i.i291, %4
  %v32.i.i293 = and i32 %v32.v.i.i292, 252645135
  %v33.i.i294 = xor i32 %v32.i.i293, 134744072
  %v34.i.i295 = and i32 %v33.i.i294, 134744072
  %55 = mul nuw i32 %v34.i.i295, 30
  %v46.i.i296 = add nuw nsw i32 %55, %v33.i.i294
  %v20.i297 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v46.i.i296, i32 %v26.sroa.0.0.copyload, i32 0) #19
  %v27.i298 = load i8, ptr %23, align 1
  %v32.i300 = load i8, ptr %v31.i299, align 1
  %v36.sroa.2.0.insert.ext.i301 = zext i8 %v32.i300 to i16
  %v36.sroa.2.0.insert.shift.i302 = shl nuw i16 %v36.sroa.2.0.insert.ext.i301, 8
  %v36.sroa.0.0.insert.ext.i303 = zext i8 %v27.i298 to i16
  %v4.i.i304 = lshr i16 %v36.sroa.2.0.insert.ext.i301, 7
  %v6.i.i305 = zext nneg i16 %v4.i.i304 to i32
  %v9.i.i306 = lshr i16 %v36.sroa.2.0.insert.ext.i301, 2
  %v10.i.i307 = and i16 %v9.i.i306, 31
  %v36.sroa.2.0.insert.shift.masked.i308 = and i16 %v36.sroa.2.0.insert.shift.i302, 768
  %v12.i.i309 = or disjoint i16 %v36.sroa.2.0.insert.shift.masked.i308, %v36.sroa.0.0.insert.ext.i303
  %v13.i.i310 = zext nneg i16 %v12.i.i309 to i32
  switch i16 %v10.i.i307, label %bb10.i.i420 [
    i16 0, label %bb1.i.i405
    i16 31, label %bb9.i.i311
  ]

bb1.i.i405:                                       ; preds = %bb19
  %v15.i.i406 = icmp eq i16 %v12.i.i309, 0
  br i1 %v15.i.i406, label %bb2.i.i418, label %bb6.i.i407

bb2.i.i418:                                       ; preds = %bb1.i.i405
  %v17.i.i419 = shl nuw i32 %v6.i.i305, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316

bb6.i.i407:                                       ; preds = %bb1.i.i405
  %v13.masked.numleadingzeros.i.i408 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i.i310, i1 true)
  %v13.masked.leadingonepos.i.i409 = xor i32 %v13.masked.numleadingzeros.i.i408, 31
  %bb5.tripcount.i.i410 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i409
  %v23.i.i411 = shl nuw nsw i32 %v13.i.i310, %bb5.tripcount.i.i410
  %v27.i.i412 = shl nuw i32 %v6.i.i305, 31
  %reass.sub.i413 = or disjoint i32 %v27.i.i412, 1124073472
  %56 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i408, 23
  %v31.i.i414 = sub nuw nsw i32 %reass.sub.i413, %56
  %v25.i.i415 = shl i32 %v23.i.i411, 13
  %v33.i2.i416 = and i32 %v25.i.i415, 8380416
  %v34.i3.i417 = or disjoint i32 %v31.i.i414, %v33.i2.i416
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316

bb9.i.i311:                                       ; preds = %bb19
  %v38.i.i312 = shl nuw i32 %v6.i.i305, 31
  %v41.i.i313 = shl nuw nsw i32 %v13.i.i310, 13
  %v39.i.i314 = or disjoint i32 %v41.i.i313, %v38.i.i312
  %v42.i.i315 = or disjoint i32 %v39.i.i314, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316

bb10.i.i420:                                      ; preds = %bb19
  %v44.i.i421 = shl nuw i32 %v6.i.i305, 31
  %57 = add nuw nsw i16 %v10.i.i307, 112
  %v46.i4.i422 = zext nneg i16 %57 to i32
  %v48.i.i423 = shl nuw nsw i32 %v46.i4.i422, 23
  %v49.i.i424 = or disjoint i32 %v48.i.i423, %v44.i.i421
  %v51.i.i425 = shl nuw nsw i32 %v13.i.i310, 13
  %v52.i.i426 = or disjoint i32 %v49.i.i424, %v51.i.i425
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316

cuda_kernels__oxide_kernels__f16_to_f32.exit.i316: ; preds = %bb10.i.i420, %bb9.i.i311, %bb6.i.i407, %bb2.i.i418
  %v54.i.i317 = phi i32 [ %v34.i3.i417, %bb6.i.i407 ], [ %v17.i.i419, %bb2.i.i418 ], [ %v42.i.i315, %bb9.i.i311 ], [ %v52.i.i426, %bb10.i.i420 ]
  %v43.i319 = load i8, ptr %v42.i318, align 1
  %v48.i321 = load i8, ptr %v47.i320, align 1
  %v52.sroa.2.0.insert.ext.i322 = zext i8 %v48.i321 to i16
  %v52.sroa.2.0.insert.shift.i323 = shl nuw i16 %v52.sroa.2.0.insert.ext.i322, 8
  %v52.sroa.0.0.insert.ext.i324 = zext i8 %v43.i319 to i16
  %v4.i5.i325 = lshr i16 %v52.sroa.2.0.insert.ext.i322, 7
  %v6.i6.i326 = zext nneg i16 %v4.i5.i325 to i32
  %v9.i7.i327 = lshr i16 %v52.sroa.2.0.insert.ext.i322, 2
  %v10.i8.i328 = and i16 %v9.i7.i327, 31
  %v52.sroa.2.0.insert.shift.masked.i329 = and i16 %v52.sroa.2.0.insert.shift.i323, 768
  %v12.i9.i330 = or disjoint i16 %v52.sroa.2.0.insert.shift.masked.i329, %v52.sroa.0.0.insert.ext.i324
  %v13.i10.i331 = zext nneg i16 %v12.i9.i330 to i32
  switch i16 %v10.i8.i328, label %bb10.i33.i398 [
    i16 0, label %bb1.i18.i383
    i16 31, label %bb9.i11.i332
  ]

bb1.i18.i383:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316
  %v15.i19.i384 = icmp eq i16 %v12.i9.i330, 0
  br i1 %v15.i19.i384, label %bb2.i31.i396, label %bb6.i20.i385

bb2.i31.i396:                                     ; preds = %bb1.i18.i383
  %v17.i32.i397 = shl nuw i32 %v6.i6.i326, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337

bb6.i20.i385:                                     ; preds = %bb1.i18.i383
  %v13.masked.numleadingzeros.i21.i386 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10.i331, i1 true)
  %v13.masked.leadingonepos.i22.i387 = xor i32 %v13.masked.numleadingzeros.i21.i386, 31
  %bb5.tripcount.i23.i388 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22.i387
  %v23.i24.i389 = shl nuw nsw i32 %v13.i10.i331, %bb5.tripcount.i23.i388
  %v27.i25.i390 = shl nuw i32 %v6.i6.i326, 31
  %reass.sub63.i391 = or disjoint i32 %v27.i25.i390, 1124073472
  %58 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21.i386, 23
  %v31.i27.i392 = sub nuw nsw i32 %reass.sub63.i391, %58
  %v25.i28.i393 = shl i32 %v23.i24.i389, 13
  %v33.i29.i394 = and i32 %v25.i28.i393, 8380416
  %v34.i30.i395 = or disjoint i32 %v31.i27.i392, %v33.i29.i394
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337

bb9.i11.i332:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316
  %v38.i12.i333 = shl nuw i32 %v6.i6.i326, 31
  %v41.i13.i334 = shl nuw nsw i32 %v13.i10.i331, 13
  %v39.i14.i335 = or disjoint i32 %v41.i13.i334, %v38.i12.i333
  %v42.i15.i336 = or disjoint i32 %v39.i14.i335, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337

bb10.i33.i398:                                    ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit.i316
  %v44.i34.i399 = shl nuw i32 %v6.i6.i326, 31
  %59 = add nuw nsw i16 %v10.i8.i328, 112
  %v46.i35.i400 = zext nneg i16 %59 to i32
  %v48.i36.i401 = shl nuw nsw i32 %v46.i35.i400, 23
  %v49.i37.i402 = or disjoint i32 %v48.i36.i401, %v44.i34.i399
  %v51.i38.i403 = shl nuw nsw i32 %v13.i10.i331, 13
  %v52.i39.i404 = or disjoint i32 %v49.i37.i402, %v51.i38.i403
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337

cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337: ; preds = %bb10.i33.i398, %bb9.i11.i332, %bb6.i20.i385, %bb2.i31.i396
  %v54.i16.i338 = phi i32 [ %v34.i30.i395, %bb6.i20.i385 ], [ %v17.i32.i397, %bb2.i31.i396 ], [ %v42.i15.i336, %bb9.i11.i332 ], [ %v52.i39.i404, %bb10.i33.i398 ]
  %v551.i339 = lshr i64 %v64, 5
  %v9.i41.i340 = icmp samesign ugt i64 %v64, 127
  br i1 %v9.i41.i340, label %bb2.i46.i375, label %bb1.i42.i341

bb1.i42.i341:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337
  %v16.i.i342 = getelementptr i8, ptr %25, i64 %v551.i339
  %v17.i43.i343 = load i8, ptr %v16.i.i342, align 1
  %v18.i44.i344 = and i8 %v17.i43.i343, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i345

bb2.i46.i375:                                     ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40.i337
  %v25.i47.i376 = getelementptr i8, ptr %26, i64 %v551.i339
  %v26.i.i377 = load i8, ptr %v25.i47.i376, align 1
  %v27.i48.i378 = and i8 %v26.i.i377, 15
  %v32.i49.i379 = getelementptr i8, ptr %23, i64 %v551.i339
  %v33.i50.i380 = load i8, ptr %v32.i49.i379, align 1
  %60 = lshr i8 %v33.i50.i380, 2
  %v39.i51.i381 = and i8 %60, 48
  %v40.i.i382 = or disjoint i8 %v39.i51.i381, %v27.i48.i378
  br label %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i345

cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i345: ; preds = %bb2.i46.i375, %bb1.i42.i341
  %v41.i45.i346 = phi i8 [ %v18.i44.i344, %bb1.i42.i341 ], [ %v40.i.i382, %bb2.i46.i375 ]
  br i1 %v9.i41.i340, label %bb2.i57.i366, label %bb1.i53.i347

bb1.i53.i347:                                     ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i345
  %v16.i54.i348 = getelementptr i8, ptr %26, i64 %v551.i339
  %v17.i55.i349 = load i8, ptr %v16.i54.i348, align 1
  %v18.i56.i350 = and i8 %v17.i55.i349, 63
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit427

bb2.i57.i366:                                     ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_scale.exit.i345
  %v19.i.i367 = add nsw i64 %v551.i339, -4
  %v25.i58.i368 = getelementptr i8, ptr %27, i64 %v19.i.i367
  %v26.i59.i369 = load i8, ptr %v25.i58.i368, align 1
  %v29.i.i370 = lshr i8 %v26.i59.i369, 4
  %v34.i60.i371 = getelementptr i8, ptr %26, i64 %v19.i.i367
  %v35.i.i372 = load i8, ptr %v34.i60.i371, align 1
  %61 = lshr i8 %v35.i.i372, 2
  %v41.i61.i373 = and i8 %61, 48
  %v42.i62.i374 = or disjoint i8 %v41.i61.i373, %v29.i.i370
  br label %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit427

cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit427: ; preds = %bb1.i53.i347, %bb2.i57.i366
  %v43.i.i351 = phi i8 [ %v18.i56.i350, %bb1.i53.i347 ], [ %v42.i62.i374, %bb2.i57.i366 ]
  %v55.i.i352 = bitcast i32 %v54.i.i317 to float
  %v59.i353 = uitofp nneg i8 %v41.i45.i346 to float
  %v60.i354 = fmul contract float %v55.i.i352, %v59.i353
  %v21.i355 = shl nsw i32 %v97, 3
  %v22.i356 = add i32 %v20.i297, %v21.i355
  %v61.i357 = sitofp i32 %v22.i356 to float
  %v62.i358 = fmul contract float %v60.i354, %v61.i357
  %v55.i17.i359 = bitcast i32 %v54.i16.i338 to float
  %v66.i360 = uitofp nneg i8 %v43.i.i351 to float
  %v67.i361 = fmul contract float %v55.i17.i359, %v66.i360
  %v68.i362 = sitofp i32 %v97 to float
  %v69.i363 = fmul contract float %v67.i361, %v68.i362
  %v70.i364 = fsub contract float %v62.i358, %v69.i363
  %v71.i365 = fmul contract float %v103, %v70.i364
  %v145 = fadd contract float %v58440, %v71.i365
  br label %bb21

bb21:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit427, %bb18
  %v146 = phi float [ %v58440, %bb18 ], [ %v145, %cuda_kernels__oxide_kernels__kernels__q4k_q8_chunk.exit427 ]
  br i1 %v60.not, label %bb9, label %bb22

bb22:                                             ; preds = %bb21
  %v148 = add nuw nsw i64 %v52453, 1
  %exitcond.not = icmp eq i64 %v148, %v44
  br i1 %exitcond.not, label %bb24.preheader, label %bb8.preheader

bb27:                                             ; preds = %bb24.preheader
  %v158 = mul nuw nsw i64 %v41.zext, %v30
  %v159 = add nuw nsw i64 %v43, %v158
  %v160.not = icmp samesign ult i64 %v43, %v30
  br i1 %v160.not, label %bb28, label %bb29

bb28:                                             ; preds = %bb27
  %v163 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  store float %v182.4, ptr %v163, align 4
  br label %bb29

bb29:                                             ; preds = %bb28, %bb27
  %v164 = or disjoint i64 %v43, 1
  %v165.not = icmp samesign ult i64 %v164, %v30
  br i1 %v165.not, label %bb30, label %bb32

bb30:                                             ; preds = %bb29
  %62 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  %v169 = getelementptr inbounds nuw i8, ptr %62, i64 4
  store float %v184.4, ptr %v169, align 4
  br label %bb32

bb32:                                             ; preds = %bb29, %bb30
  %v170 = or disjoint i64 %v43, 2
  %v171.not = icmp samesign ult i64 %v170, %v30
  br i1 %v171.not, label %bb33, label %bb35

bb33:                                             ; preds = %bb32
  %63 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  %v175 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store float %v186.4, ptr %v175, align 4
  br label %bb35

bb35:                                             ; preds = %bb32, %bb33
  %v176 = or disjoint i64 %v43, 3
  %v177.not = icmp samesign ult i64 %v176, %v30
  br i1 %v177.not, label %bb36, label %bb40

bb36:                                             ; preds = %bb35
  %64 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  %v181 = getelementptr inbounds nuw i8, ptr %64, i64 12
  store float %v188.4, ptr %v181, align 4
  br label %bb40

bb40:                                             ; preds = %bb24.preheader, %bb35, %bb36, %entry
  ret void

bb45:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @q5_0_gemm_element(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(address_is_null) %v7, i64 %v8) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i11 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i12 = icmp eq i32 %v4.i10, 1
  %v7.i13 = icmp eq i32 %v6.i11, 1
  %v8.not.not.i = and i1 %v5.i12, %v7.i13
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i14 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i14
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v26 = zext i32 %v4 to i64
  %v27 = zext i32 %v6 to i64
  %v28 = mul nuw i64 %v27, %v26
  %v29.not = icmp ult i64 %v22.i, %v28
  br i1 %v29.not, label %bb3, label %bb24

bb3:                                              ; preds = %entry
  %v31.not = icmp eq i32 %v4, 0
  br i1 %v31.not, label %bb32, label %bb4

bb4:                                              ; preds = %bb3
  %v35 = zext i32 %v5 to i64
  %v39.not34.not = icmp eq i32 %v5, 0
  br i1 %v39.not34.not, label %bb20, label %bb6.lr.ph

bb6.lr.ph:                                        ; preds = %bb4
  %v26.frozen = freeze i64 %v26
  %v34 = udiv i64 %v22.i, %v26.frozen
  %0 = mul i64 %v34, %v26.frozen
  %v33.decomposed = sub i64 %v22.i, %0
  %v41 = mul nuw i64 %v33.decomposed, %v35
  %v87 = mul i64 %v34, %v35
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb19
  %v3836 = phi i64 [ 0, %bb6.lr.ph ], [ %v146, %bb19 ]
  %v3735 = phi float [ 0.000000e+00, %bb6.lr.ph ], [ %v144, %bb19 ]
  %reass.add = add nuw i64 %v3836, %v41
  %reass.mul = mul i64 %reass.add, 22
  %v45 = icmp ult i64 %reass.mul, %v1
  br i1 %v45, label %bb7, label %bb33

bb7:                                              ; preds = %bb6
  %v49 = or disjoint i64 %reass.mul, 1
  %v50 = icmp ult i64 %v49, %v1
  br i1 %v50, label %bb8, label %bb34

bb8:                                              ; preds = %bb7
  %v47 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v48 = load i8, ptr %v47, align 1
  %v52 = getelementptr inbounds i8, ptr %v0, i64 %v49
  %v53 = load i8, ptr %v52, align 1
  %v57 = alloca [2 x i8], align 2
  store i8 %v48, ptr %v57, align 2
  %v57.repack1 = getelementptr inbounds nuw i8, ptr %v57, i64 1
  store i8 %v53, ptr %v57.repack1, align 1
  %v58 = load i16, ptr %v57, align 2
  %v4.i15 = lshr i16 %v58, 15
  %v6.i16 = zext nneg i16 %v4.i15 to i32
  %v9.i = lshr i16 %v58, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v58, 1023
  %v13.i17 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb8
  %v15.i18 = icmp eq i16 %v12.i, 0
  br i1 %v15.i18, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i19 = shl nuw i32 %v6.i16, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i17, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i17, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i16, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb8
  %v38.i = shl nuw i32 %v6.i16, 31
  %v41.i = shl nuw nsw i32 %v13.i17, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb8
  %v44.i = shl nuw i32 %v6.i16, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i17, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i19, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v60 = add nuw i64 %reass.mul, 2
  %v61 = icmp ult i64 %v60, %v1
  br i1 %v61, label %bb10, label %bb35

bb10:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v63 = getelementptr inbounds i8, ptr %v0, i64 %v60
  %v64 = load i8, ptr %v63, align 1
  %v65 = add nuw i64 %reass.mul, 3
  %v66 = icmp ult i64 %v65, %v1
  br i1 %v66, label %bb11, label %bb36

bb11:                                             ; preds = %bb10
  %v68 = getelementptr inbounds i8, ptr %v0, i64 %v65
  %v69 = load i8, ptr %v68, align 1
  %v70 = add nuw i64 %reass.mul, 4
  %v71 = icmp ult i64 %v70, %v1
  br i1 %v71, label %bb12, label %bb37

bb12:                                             ; preds = %bb11
  %v75 = add nuw i64 %reass.mul, 5
  %v76 = icmp ult i64 %v75, %v1
  br i1 %v76, label %bb13, label %bb38

bb13:                                             ; preds = %bb12
  %v73 = getelementptr inbounds i8, ptr %v0, i64 %v70
  %v74 = load i8, ptr %v73, align 1
  %v78 = getelementptr inbounds i8, ptr %v0, i64 %v75
  %v79 = load i8, ptr %v78, align 1
  %v85 = alloca [4 x i8], align 4
  store i8 %v64, ptr %v85, align 4
  %v85.repack3 = getelementptr inbounds nuw i8, ptr %v85, i64 1
  store i8 %v69, ptr %v85.repack3, align 1
  %v85.repack5 = getelementptr inbounds nuw i8, ptr %v85, i64 2
  store i8 %v74, ptr %v85.repack5, align 2
  %v85.repack7 = getelementptr inbounds nuw i8, ptr %v85, i64 3
  store i8 %v79, ptr %v85.repack7, align 1
  %v86 = load i32, ptr %v85, align 4
  %v88 = add i64 %v3836, %v87
  %v89 = shl i64 %v88, 5
  %v94 = add nuw i64 %reass.mul, 6
  br label %bb15

bb15:                                             ; preds = %bb13, %bb18
  %v9133 = phi i64 [ 0, %bb13 ], [ %v145, %bb18 ]
  %v9032 = phi float [ %v3735, %bb13 ], [ %v144, %bb18 ]
  %v95 = add nuw i64 %v94, %v9133
  %v96 = icmp ult i64 %v95, %v1
  br i1 %v96, label %bb16, label %bb39

bb16:                                             ; preds = %bb15
  %v127 = add nuw nsw i64 %v9133, %v89
  %v129 = icmp ult i64 %v127, %v3
  br i1 %v129, label %bb17, label %bb40

bb17:                                             ; preds = %bb16
  %v138 = or disjoint i64 %v127, 16
  %v139 = icmp ult i64 %v138, %v3
  br i1 %v139, label %bb18, label %bb41

bb18:                                             ; preds = %bb17
  %3 = trunc nuw nsw i64 %v9133 to i32
  %v112 = or disjoint i32 %3, 16
  %v114 = lshr i32 %v86, %v112
  %v115 = shl nuw nsw i32 %v114, 4
  %v118 = and i32 %v115, 16
  %v98 = getelementptr inbounds i8, ptr %v0, i64 %v95
  %v99 = load i8, ptr %v98, align 1
  %v121 = lshr i8 %v99, 4
  %v122 = zext nneg i8 %v121 to i32
  %v123 = add nsw i32 %v118, -16
  %v124 = or disjoint i32 %v123, %v122
  %v135 = sitofp i32 %v124 to float
  %v136 = fmul contract float %v55.i, %v135
  %v102 = lshr i32 %v86, %3
  %v103 = shl i32 %v102, 4
  %v106 = and i32 %v103, 16
  %v107 = and i8 %v99, 15
  %v108 = zext nneg i8 %v107 to i32
  %v109 = add nsw i32 %v106, -16
  %v110 = or disjoint i32 %v109, %v108
  %v125 = sitofp i32 %v110 to float
  %v126 = fmul contract float %v55.i, %v125
  %v131 = getelementptr inbounds float, ptr %v2, i64 %v127
  %v132 = load float, ptr %v131, align 4
  %v133 = fmul contract float %v132, %v126
  %v134 = fadd contract float %v9032, %v133
  %v141 = getelementptr inbounds float, ptr %v2, i64 %v138
  %v142 = load float, ptr %v141, align 4
  %v143 = fmul contract float %v142, %v136
  %v144 = fadd contract float %v143, %v134
  %v145 = add nuw nsw i64 %v9133, 1
  %exitcond = icmp eq i64 %v145, 16
  br i1 %exitcond, label %bb19, label %bb15

bb19:                                             ; preds = %bb18
  %v146 = add nuw nsw i64 %v3836, 1
  %exitcond37.not = icmp eq i64 %v146, %v35
  br i1 %exitcond37.not, label %bb20, label %bb6

bb20:                                             ; preds = %bb19, %bb4
  %v37.lcssa = phi float [ 0.000000e+00, %bb4 ], [ %v144, %bb19 ]
  %v150 = icmp ult i64 %v22.i, %v8
  %or.cond.not = select i1 %.v18.i, i1 %v150, i1 false
  %v1649 = icmp ne ptr %v7, null
  %v164 = select i1 %or.cond.not, i1 %v1649, i1 false
  br i1 %v164, label %bb21, label %bb24

bb21:                                             ; preds = %bb20
  %v153 = getelementptr inbounds nuw float, ptr %v7, i64 %v22.i
  store float %v37.lcssa, ptr %v153, align 4
  br label %bb24

bb24:                                             ; preds = %bb20, %bb21, %entry
  ret void

bb32:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb34:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb35:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  tail call void @llvm.trap() #19
  unreachable

bb36:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb37:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb38:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb39:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q5_0_gemm_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(none) %v7, i64 %v8) #6 {
entry:
  %v23 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v24 = zext nneg i32 %v23 to i64
  %v25 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v26 = zext nneg i32 %v25 to i64
  %v27 = zext i32 %v6 to i64
  %v28 = zext i32 %v4 to i64
  %v29 = mul nuw i64 %v27, %v28
  %v30.not = icmp ugt i64 %v29, %v26
  br i1 %v30.not, label %bb4, label %bb36

bb4:                                              ; preds = %entry
  %v32.not = icmp eq i32 %v4, 0
  br i1 %v32.not, label %bb37, label %bb5

bb5:                                              ; preds = %bb4
  %v36 = zext i32 %v5 to i64
  %v40.not26 = icmp ult i32 %v23, %v5
  br i1 %v40.not26, label %bb7.lr.ph, label %bb21

bb7.lr.ph:                                        ; preds = %bb5
  %v4.frozen = freeze i32 %v4
  %v3511 = udiv i32 %v25, %v4.frozen
  %v35.zext = zext nneg i32 %v3511 to i64
  %0 = mul i32 %v3511, %v4.frozen
  %v3410.decomposed = sub i32 %v25, %0
  %v34.zext = zext nneg i32 %v3410.decomposed to i64
  %v42 = mul nuw nsw i64 %v34.zext, %v36
  %v88 = mul nuw nsw i64 %v35.zext, %v36
  br label %bb7

bb7:                                              ; preds = %bb7.lr.ph, %bb20
  %v3928 = phi i64 [ %v24, %bb7.lr.ph ], [ %v147, %bb20 ]
  %v3827 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v145, %bb20 ]
  %reass.add = add nuw i64 %v3928, %v42
  %reass.mul = mul i64 %reass.add, 22
  %v46 = icmp ult i64 %reass.mul, %v1
  br i1 %v46, label %bb8, label %bb38

bb8:                                              ; preds = %bb7
  %v50 = or disjoint i64 %reass.mul, 1
  %v51 = icmp ult i64 %v50, %v1
  br i1 %v51, label %bb9, label %bb39

bb9:                                              ; preds = %bb8
  %v48 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v49 = load i8, ptr %v48, align 1
  %v53 = getelementptr inbounds i8, ptr %v0, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v58 = alloca [2 x i8], align 2
  store i8 %v49, ptr %v58, align 2
  %v58.repack1 = getelementptr inbounds nuw i8, ptr %v58, i64 1
  store i8 %v54, ptr %v58.repack1, align 1
  %v59 = load i16, ptr %v58, align 2
  %v4.i = lshr i16 %v59, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v59, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v59, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb9
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb9
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb9
  %v44.i = shl nuw i32 %v6.i, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v61 = add nuw i64 %reass.mul, 2
  %v62 = icmp ult i64 %v61, %v1
  br i1 %v62, label %bb11, label %bb40

bb11:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v64 = getelementptr inbounds i8, ptr %v0, i64 %v61
  %v65 = load i8, ptr %v64, align 1
  %v66 = add nuw i64 %reass.mul, 3
  %v67 = icmp ult i64 %v66, %v1
  br i1 %v67, label %bb12, label %bb41

bb12:                                             ; preds = %bb11
  %v69 = getelementptr inbounds i8, ptr %v0, i64 %v66
  %v70 = load i8, ptr %v69, align 1
  %v71 = add nuw i64 %reass.mul, 4
  %v72 = icmp ult i64 %v71, %v1
  br i1 %v72, label %bb13, label %bb42

bb13:                                             ; preds = %bb12
  %v76 = add nuw i64 %reass.mul, 5
  %v77 = icmp ult i64 %v76, %v1
  br i1 %v77, label %bb14, label %bb43

bb14:                                             ; preds = %bb13
  %v74 = getelementptr inbounds i8, ptr %v0, i64 %v71
  %v75 = load i8, ptr %v74, align 1
  %v79 = getelementptr inbounds i8, ptr %v0, i64 %v76
  %v80 = load i8, ptr %v79, align 1
  %v86 = alloca [4 x i8], align 4
  store i8 %v65, ptr %v86, align 4
  %v86.repack3 = getelementptr inbounds nuw i8, ptr %v86, i64 1
  store i8 %v70, ptr %v86.repack3, align 1
  %v86.repack5 = getelementptr inbounds nuw i8, ptr %v86, i64 2
  store i8 %v75, ptr %v86.repack5, align 2
  %v86.repack7 = getelementptr inbounds nuw i8, ptr %v86, i64 3
  store i8 %v80, ptr %v86.repack7, align 1
  %v87 = load i32, ptr %v86, align 4
  %v89 = add nuw i64 %v3928, %v88
  %v90 = shl i64 %v89, 5
  %v95 = add nuw i64 %reass.mul, 6
  br label %bb16

bb16:                                             ; preds = %bb14, %bb19
  %v9225 = phi i64 [ 0, %bb14 ], [ %v146, %bb19 ]
  %v9124 = phi float [ %v3827, %bb14 ], [ %v145, %bb19 ]
  %v96 = add nuw i64 %v95, %v9225
  %v97 = icmp ult i64 %v96, %v1
  br i1 %v97, label %bb17, label %bb44

bb17:                                             ; preds = %bb16
  %v128 = add nuw nsw i64 %v9225, %v90
  %v130 = icmp ult i64 %v128, %v3
  br i1 %v130, label %bb18, label %bb45

bb18:                                             ; preds = %bb17
  %v138 = or disjoint i64 %v128, 16
  %v139 = icmp ult i64 %v138, %v3
  br i1 %v139, label %bb19, label %bb46

bb19:                                             ; preds = %bb18
  %3 = trunc nuw nsw i64 %v9225 to i32
  %v113 = or disjoint i32 %3, 16
  %v115 = lshr i32 %v87, %v113
  %v116 = shl nuw nsw i32 %v115, 4
  %v119 = and i32 %v116, 16
  %v99 = getelementptr inbounds i8, ptr %v0, i64 %v96
  %v100 = load i8, ptr %v99, align 1
  %v122 = lshr i8 %v100, 4
  %v123 = zext nneg i8 %v122 to i32
  %v124 = add nsw i32 %v119, -16
  %v125 = or disjoint i32 %v124, %v123
  %v135 = sitofp i32 %v125 to float
  %v136 = fmul contract float %v55.i, %v135
  %v103 = lshr i32 %v87, %3
  %v104 = shl i32 %v103, 4
  %v107 = and i32 %v104, 16
  %v108 = and i8 %v100, 15
  %v109 = zext nneg i8 %v108 to i32
  %v110 = add nsw i32 %v107, -16
  %v111 = or disjoint i32 %v110, %v109
  %v126 = sitofp i32 %v111 to float
  %v127 = fmul contract float %v55.i, %v126
  %v132 = getelementptr inbounds float, ptr %v2, i64 %v128
  %v133 = load float, ptr %v132, align 4
  %v134 = fmul contract float %v133, %v127
  %v141 = getelementptr inbounds float, ptr %v2, i64 %v138
  %v142 = load float, ptr %v141, align 4
  %v143 = fmul contract float %v142, %v136
  %v144 = fadd contract float %v134, %v143
  %v145 = fadd contract float %v9124, %v144
  %v146 = add nuw nsw i64 %v9225, 1
  %exitcond = icmp eq i64 %v146, 16
  br i1 %exitcond, label %bb20, label %bb16

bb20:                                             ; preds = %bb19
  %v147 = add nuw nsw i64 %v3928, 32
  %v40.not = icmp samesign ult i64 %v147, %v36
  br i1 %v40.not, label %bb7, label %bb21

bb21:                                             ; preds = %bb20, %bb5
  %v38.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v145, %bb20 ]
  %v148 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_11, i64 %v24
  store float %v38.lcssa, ptr addrspace(3) %v148, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v153.not = icmp samesign ult i32 %v23, 16
  br i1 %v153.not, label %bb26, label %bb30

bb26:                                             ; preds = %bb21
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v148, i64 64
  %v158 = load float, ptr addrspace(3) %gep, align 4
  %v160 = load float, ptr addrspace(3) %v148, align 4
  %v161 = fadd contract float %v158, %v160
  store float %v161, ptr addrspace(3) %v148, align 4
  br label %bb30

bb30:                                             ; preds = %bb21, %bb26
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v153.not.1 = icmp samesign ult i32 %v23, 8
  br i1 %v153.not.1, label %bb26.1, label %bb30.1

bb26.1:                                           ; preds = %bb30
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v148, i64 32
  %v158.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v160.1 = load float, ptr addrspace(3) %v148, align 4
  %v161.1 = fadd contract float %v158.1, %v160.1
  store float %v161.1, ptr addrspace(3) %v148, align 4
  br label %bb30.1

bb30.1:                                           ; preds = %bb26.1, %bb30
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v153.not.2 = icmp samesign ult i32 %v23, 4
  br i1 %v153.not.2, label %bb26.2, label %bb30.2

bb26.2:                                           ; preds = %bb30.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v148, i64 16
  %v158.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v160.2 = load float, ptr addrspace(3) %v148, align 4
  %v161.2 = fadd contract float %v158.2, %v160.2
  store float %v161.2, ptr addrspace(3) %v148, align 4
  br label %bb30.2

bb30.2:                                           ; preds = %bb26.2, %bb30.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v153.not.3 = icmp samesign ult i32 %v23, 2
  br i1 %v153.not.3, label %bb26.3, label %bb30.3

bb26.3:                                           ; preds = %bb30.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v148, i64 8
  %v158.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v160.3 = load float, ptr addrspace(3) %v148, align 4
  %v161.3 = fadd contract float %v158.3, %v160.3
  store float %v161.3, ptr addrspace(3) %v148, align 4
  br label %bb30.3

bb30.3:                                           ; preds = %bb26.3, %bb30.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v153.not.4 = icmp eq i32 %v23, 0
  br i1 %v153.not.4, label %bb26.4, label %bb30.4

bb26.4:                                           ; preds = %bb30.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v148, i64 4
  %v158.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v160.4 = load float, ptr addrspace(3) %v148, align 4
  %v161.4 = fadd contract float %v158.4, %v160.4
  store float %v161.4, ptr addrspace(3) %v148, align 4
  br label %bb30.4

bb30.4:                                           ; preds = %bb26.4, %bb30.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v164 = icmp eq i32 %v23, 0
  br i1 %v164, label %bb33, label %bb36

bb33:                                             ; preds = %bb30.4
  %v169 = getelementptr inbounds nuw float, ptr %v7, i64 %v26
  %v167 = load float, ptr addrspace(3) @__shared_mem_11, align 4
  store float %v167, ptr %v169, align 4
  br label %bb36

bb36:                                             ; preds = %bb30.4, %bb33, %entry
  ret void

bb37:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb38:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb39:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb44:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q6k_gemm_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(none) %v7, i64 %v8) #6 {
entry:
  %v22 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v23 = zext nneg i32 %v22 to i64
  %v24 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v25 = zext nneg i32 %v24 to i64
  %v26 = zext i32 %v4 to i64
  %v27 = zext i32 %v6 to i64
  %v28 = mul nuw i64 %v27, %v26
  %v29.not = icmp ugt i64 %v28, %v25
  br i1 %v29.not, label %bb4, label %bb40

bb4:                                              ; preds = %entry
  %v31.not = icmp eq i32 %v4, 0
  br i1 %v31.not, label %bb41, label %bb5

bb5:                                              ; preds = %bb4
  %v35 = zext i32 %v5 to i64
  %v39.not37.not = icmp eq i32 %v5, 0
  br i1 %v39.not37.not, label %bb25, label %bb7.lr.ph

bb7.lr.ph:                                        ; preds = %bb5
  %v4.frozen = freeze i32 %v4
  %v3410 = udiv i32 %v24, %v4.frozen
  %v34.zext = zext nneg i32 %v3410 to i64
  %0 = mul i32 %v3410, %v4.frozen
  %v339.decomposed = sub i32 %v24, %0
  %v33.zext = zext nneg i32 %v339.decomposed to i64
  %v41 = mul nuw nsw i64 %v33.zext, %v35
  %v61 = mul nuw nsw i64 %v34.zext, %v35
  %v783 = lshr i64 %v23, 4
  %v70 = add nuw nsw i64 %v23, 128
  %v73 = or disjoint i64 %v783, 192
  br label %bb7

bb7:                                              ; preds = %bb7.lr.ph, %bb24
  %v3839 = phi i64 [ 0, %bb7.lr.ph ], [ %v209, %bb24 ]
  %v3738 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v207, %bb24 ]
  %reass.add = add nuw i64 %v3839, %v41
  %reass.mul = mul i64 %reass.add, 210
  %v44 = add i64 %reass.mul, 208
  %v46 = icmp ult i64 %v44, %v1
  br i1 %v46, label %bb8, label %bb42

bb8:                                              ; preds = %bb7
  %v50 = add i64 %reass.mul, 209
  %v51 = icmp ult i64 %v50, %v1
  br i1 %v51, label %bb9, label %bb43

bb9:                                              ; preds = %bb8
  %v48 = getelementptr inbounds i8, ptr %v0, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v53 = getelementptr inbounds i8, ptr %v0, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v58 = alloca [2 x i8], align 2
  store i8 %v49, ptr %v58, align 2
  %v58.repack1 = getelementptr inbounds nuw i8, ptr %v58, i64 1
  store i8 %v54, ptr %v58.repack1, align 1
  %v59 = load i16, ptr %v58, align 2
  %v4.i = lshr i16 %v59, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v59, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v59, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb9
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb9
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb9
  %v44.i = shl nuw i32 %v6.i, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v62 = add nuw nsw i64 %v3839, %v61
  %v63 = shl i64 %v62, 8
  %v69 = add i64 %reass.mul, %v23
  %v72 = add i64 %v70, %reass.mul
  %v75 = add i64 %v73, %reass.mul
  %v77 = add i64 %v63, %v23
  br label %bb12

bb12:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb23
  %v66.not = phi i1 [ true, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ false, %bb23 ]
  %v6536 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ 1, %bb23 ]
  %v6435 = phi float [ %v3738, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v207, %bb23 ]
  %v68 = shl nuw nsw i64 %v6536, 6
  %v74 = shl nuw nsw i64 %v6536, 3
  %v76 = shl nuw nsw i64 %v6536, 7
  %v79 = add i64 %v69, %v68
  %v80 = icmp ult i64 %v79, %v1
  br i1 %v80, label %bb13, label %bb44

bb13:                                             ; preds = %bb12
  %v71 = shl nuw nsw i64 %v6536, 5
  %v82 = getelementptr inbounds i8, ptr %v0, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v86 = add i64 %v72, %v71
  %v87 = icmp ult i64 %v86, %v1
  br i1 %v87, label %bb14, label %bb45

bb14:                                             ; preds = %bb13
  %v84 = and i8 %v83, 15
  %v89 = getelementptr inbounds i8, ptr %v0, i64 %v86
  %v90 = load i8, ptr %v89, align 1
  %v91 = shl i8 %v90, 4
  %3 = and i8 %v91, 48
  %v954 = or disjoint i8 %3, %v84
  %v95 = zext nneg i8 %v954 to i32
  %v96 = add nsw i32 %v95, -32
  %v97 = add i64 %v79, 32
  %v98 = icmp ult i64 %v97, %v1
  br i1 %v98, label %bb15, label %bb46

bb15:                                             ; preds = %bb14
  %v100 = getelementptr inbounds i8, ptr %v0, i64 %v97
  %v101 = load i8, ptr %v100, align 1
  %v102 = and i8 %v101, 15
  %4 = shl i8 %v90, 2
  %5 = and i8 %4, 48
  %v1115 = or disjoint i8 %v102, %5
  %v111 = zext nneg i8 %v1115 to i32
  %v112 = add nsw i32 %v111, -32
  %v115 = lshr i8 %v83, 4
  %v120 = and i8 %v90, 48
  %v1246 = or disjoint i8 %v120, %v115
  %v124 = zext nneg i8 %v1246 to i32
  %v125 = add nsw i32 %v124, -32
  %v128 = lshr i8 %v101, 4
  %6 = lshr i8 %v90, 2
  %7 = and i8 %6, 48
  %v1377 = or disjoint i8 %v128, %7
  %v137 = zext nneg i8 %v1377 to i32
  %v138 = add nsw i32 %v137, -32
  %v139 = add i64 %v75, %v74
  %v140 = icmp ult i64 %v139, %v1
  br i1 %v140, label %bb16, label %bb47

bb16:                                             ; preds = %bb15
  %v142 = getelementptr inbounds i8, ptr %v0, i64 %v139
  %v143 = load i8, ptr %v142, align 1
  %v145 = sitofp i8 %v143 to float
  %v146 = add i64 %v139, 2
  %v147 = icmp ult i64 %v146, %v1
  br i1 %v147, label %bb17, label %bb48

bb17:                                             ; preds = %bb16
  %v149 = getelementptr inbounds i8, ptr %v0, i64 %v146
  %v150 = load i8, ptr %v149, align 1
  %v152 = sitofp i8 %v150 to float
  %v153 = add i64 %v139, 4
  %v154 = icmp ult i64 %v153, %v1
  br i1 %v154, label %bb18, label %bb49

bb18:                                             ; preds = %bb17
  %v156 = getelementptr inbounds i8, ptr %v0, i64 %v153
  %v157 = load i8, ptr %v156, align 1
  %v159 = sitofp i8 %v157 to float
  %v160 = add i64 %v139, 6
  %v161 = icmp ult i64 %v160, %v1
  br i1 %v161, label %bb19, label %bb50

bb19:                                             ; preds = %bb18
  %v163 = getelementptr inbounds i8, ptr %v0, i64 %v160
  %v164 = load i8, ptr %v163, align 1
  %v166 = sitofp i8 %v164 to float
  %v170 = add i64 %v77, %v76
  %v172 = icmp ult i64 %v170, %v3
  br i1 %v172, label %bb20, label %bb51

bb20:                                             ; preds = %bb19
  %v181 = add i64 %v170, 32
  %v182 = icmp ult i64 %v181, %v3
  br i1 %v182, label %bb21, label %bb52

bb21:                                             ; preds = %bb20
  %v191 = add i64 %v170, 64
  %v192 = icmp ult i64 %v191, %v3
  br i1 %v192, label %bb22, label %bb53

bb22:                                             ; preds = %bb21
  %v201 = add i64 %v170, 96
  %v202 = icmp ult i64 %v201, %v3
  br i1 %v202, label %bb23, label %bb54

bb23:                                             ; preds = %bb22
  %v198 = fmul contract float %v55.i, %v166
  %v199 = sitofp i32 %v138 to float
  %v200 = fmul contract float %v198, %v199
  %v167 = fmul contract float %v55.i, %v145
  %v168 = sitofp i32 %v96 to float
  %v169 = fmul contract float %v167, %v168
  %v174 = getelementptr inbounds float, ptr %v2, i64 %v170
  %v175 = load float, ptr %v174, align 4
  %v176 = fmul contract float %v169, %v175
  %v177 = fadd contract float %v6435, %v176
  %v178 = fmul contract float %v55.i, %v152
  %v179 = sitofp i32 %v112 to float
  %v180 = fmul contract float %v178, %v179
  %v184 = getelementptr inbounds float, ptr %v2, i64 %v181
  %v185 = load float, ptr %v184, align 4
  %v186 = fmul contract float %v180, %v185
  %v187 = fadd contract float %v177, %v186
  %v188 = fmul contract float %v55.i, %v159
  %v189 = sitofp i32 %v125 to float
  %v190 = fmul contract float %v188, %v189
  %v194 = getelementptr inbounds float, ptr %v2, i64 %v191
  %v195 = load float, ptr %v194, align 4
  %v196 = fmul contract float %v190, %v195
  %v197 = fadd contract float %v187, %v196
  %v204 = getelementptr inbounds float, ptr %v2, i64 %v201
  %v205 = load float, ptr %v204, align 4
  %v206 = fmul contract float %v200, %v205
  %v207 = fadd contract float %v197, %v206
  br i1 %v66.not, label %bb12, label %bb24

bb24:                                             ; preds = %bb23
  %v209 = add nuw nsw i64 %v3839, 1
  %exitcond.not = icmp eq i64 %v209, %v35
  br i1 %exitcond.not, label %bb25, label %bb7

bb25:                                             ; preds = %bb24, %bb5
  %v37.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v207, %bb24 ]
  %v210 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_3, i64 %v23
  store float %v37.lcssa, ptr addrspace(3) %v210, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v215.not = icmp samesign ult i32 %v22, 16
  br i1 %v215.not, label %bb30, label %bb34

bb30:                                             ; preds = %bb25
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v210, i64 64
  %v220 = load float, ptr addrspace(3) %gep, align 4
  %v222 = load float, ptr addrspace(3) %v210, align 4
  %v223 = fadd contract float %v220, %v222
  store float %v223, ptr addrspace(3) %v210, align 4
  br label %bb34

bb34:                                             ; preds = %bb25, %bb30
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v215.not.1 = icmp samesign ult i32 %v22, 8
  br i1 %v215.not.1, label %bb30.1, label %bb34.1

bb30.1:                                           ; preds = %bb34
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v210, i64 32
  %v220.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v222.1 = load float, ptr addrspace(3) %v210, align 4
  %v223.1 = fadd contract float %v220.1, %v222.1
  store float %v223.1, ptr addrspace(3) %v210, align 4
  br label %bb34.1

bb34.1:                                           ; preds = %bb30.1, %bb34
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v215.not.2 = icmp samesign ult i32 %v22, 4
  br i1 %v215.not.2, label %bb30.2, label %bb34.2

bb30.2:                                           ; preds = %bb34.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v210, i64 16
  %v220.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v222.2 = load float, ptr addrspace(3) %v210, align 4
  %v223.2 = fadd contract float %v220.2, %v222.2
  store float %v223.2, ptr addrspace(3) %v210, align 4
  br label %bb34.2

bb34.2:                                           ; preds = %bb30.2, %bb34.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v215.not.3 = icmp samesign ult i32 %v22, 2
  br i1 %v215.not.3, label %bb30.3, label %bb34.3

bb30.3:                                           ; preds = %bb34.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v210, i64 8
  %v220.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v222.3 = load float, ptr addrspace(3) %v210, align 4
  %v223.3 = fadd contract float %v220.3, %v222.3
  store float %v223.3, ptr addrspace(3) %v210, align 4
  br label %bb34.3

bb34.3:                                           ; preds = %bb30.3, %bb34.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v215.not.4 = icmp eq i32 %v22, 0
  br i1 %v215.not.4, label %bb30.4, label %bb34.4

bb30.4:                                           ; preds = %bb34.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v210, i64 4
  %v220.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v222.4 = load float, ptr addrspace(3) %v210, align 4
  %v223.4 = fadd contract float %v220.4, %v222.4
  store float %v223.4, ptr addrspace(3) %v210, align 4
  br label %bb34.4

bb34.4:                                           ; preds = %bb30.4, %bb34.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v226 = icmp eq i32 %v22, 0
  br i1 %v226, label %bb37, label %bb40

bb37:                                             ; preds = %bb34.4
  %v231 = getelementptr inbounds nuw float, ptr %v7, i64 %v25
  %v229 = load float, ptr addrspace(3) @__shared_mem_3, align 4
  store float %v229, ptr %v231, align 4
  br label %bb40

bb40:                                             ; preds = %bb34.4, %bb37, %entry
  ret void

bb41:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb44:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb48:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb49:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb50:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb51:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb52:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb53:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable

bb54:                                             ; preds = %bb22
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @q6k_gemv_row(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(address_is_null) %v7, i64 %v8) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i10 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i11 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i12 = icmp eq i32 %v4.i10, 1
  %v7.i13 = icmp eq i32 %v6.i11, 1
  %v8.not.not.i = and i1 %v5.i12, %v7.i13
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i14 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i14
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v25 = zext i32 %v4 to i64
  %v26 = zext i32 %v6 to i64
  %v27 = mul nuw i64 %v26, %v25
  %v28.not = icmp ult i64 %v22.i, %v27
  br i1 %v28.not, label %bb3, label %bb36

bb3:                                              ; preds = %entry
  %v30.not = icmp eq i32 %v4, 0
  br i1 %v30.not, label %bb44, label %bb4

bb4:                                              ; preds = %bb3
  %v25.frozen = freeze i64 %v25
  %v33 = udiv i64 %v22.i, %v25.frozen
  %0 = mul i64 %v33, %v25.frozen
  %v32.decomposed = sub i64 %v22.i, %0
  %v34 = mul i32 %v5, 210
  %v35 = zext i32 %v34 to i64
  %v36 = mul nuw i64 %v32.decomposed, %v35
  %v39.not59.not = icmp eq i32 %v5, 0
  br i1 %v39.not59.not, label %bb32, label %bb6.lr.ph

bb6.lr.ph:                                        ; preds = %bb4
  %v61 = zext i32 %v5 to i64
  %v62 = mul i64 %v33, %v61
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb31
  %v3861 = phi i32 [ 0, %bb6.lr.ph ], [ %v247, %bb31 ]
  %v3760 = phi float [ 0.000000e+00, %bb6.lr.ph ], [ %v244, %bb31 ]
  %v41 = zext i32 %v3861 to i64
  %v42 = mul nuw nsw i64 %v41, 210
  %v43 = add i64 %v42, %v36
  %v44 = add i64 %v43, 208
  %v46 = icmp ult i64 %v44, %v1
  br i1 %v46, label %bb7, label %bb45

bb7:                                              ; preds = %bb6
  %v50 = add i64 %v43, 209
  %v51 = icmp ult i64 %v50, %v1
  br i1 %v51, label %bb8, label %bb46

bb8:                                              ; preds = %bb7
  %v48 = getelementptr inbounds i8, ptr %v0, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v53 = getelementptr inbounds i8, ptr %v0, i64 %v50
  %v54 = load i8, ptr %v53, align 1
  %v58 = alloca [2 x i8], align 2
  store i8 %v49, ptr %v58, align 2
  %v58.repack1 = getelementptr inbounds nuw i8, ptr %v58, i64 1
  store i8 %v54, ptr %v58.repack1, align 1
  %v59 = load i16, ptr %v58, align 2
  %v4.i15 = lshr i16 %v59, 15
  %v6.i16 = zext nneg i16 %v4.i15 to i32
  %v9.i = lshr i16 %v59, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v59, 1023
  %v13.i17 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb8
  %v15.i18 = icmp eq i16 %v12.i, 0
  br i1 %v15.i18, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i19 = shl nuw i32 %v6.i16, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i17, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i17, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i16, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb8
  %v38.i = shl nuw i32 %v6.i16, 31
  %v41.i = shl nuw nsw i32 %v13.i17, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb8
  %v44.i = shl nuw i32 %v6.i16, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i17, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i19, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v633 = add i64 %v62, %v41
  %v66 = shl i64 %v633, 8
  %v73 = add i64 %v43, 128
  %v76 = add i64 %v43, 192
  br label %bb11

bb11:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb30
  %v69.not = phi i1 [ true, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ false, %bb30 ]
  %v6858 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ 1, %bb30 ]
  %v6757 = phi float [ %v3760, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v244, %bb30 ]
  %v71 = shl nuw nsw i64 %v6858, 6
  %v72 = add i64 %v71, %v43
  %v74 = shl nuw nsw i64 %v6858, 5
  %v75 = add i64 %v73, %v74
  %v77 = shl nuw nsw i64 %v6858, 3
  %v78 = add i64 %v76, %v77
  %v79 = shl nuw nsw i64 %v6858, 7
  %v80 = or disjoint i64 %v79, %v66
  br label %bb13

bb13:                                             ; preds = %bb11, %bb29
  %v8256 = phi i64 [ 0, %bb11 ], [ %v245, %bb29 ]
  %v8155 = phi float [ %v6757, %bb11 ], [ %v244, %bb29 ]
  %v854 = lshr i64 %v8256, 4
  %v86 = add nuw i64 %v72, %v8256
  %v87 = icmp ult i64 %v86, %v1
  br i1 %v87, label %bb14, label %bb47

bb14:                                             ; preds = %bb13
  %v89 = getelementptr inbounds i8, ptr %v0, i64 %v86
  %v90 = load i8, ptr %v89, align 1
  %v93 = add nuw i64 %v75, %v8256
  %v94 = icmp ult i64 %v93, %v1
  br i1 %v94, label %bb15, label %bb48

bb15:                                             ; preds = %bb14
  %v91 = and i8 %v90, 15
  %v96 = getelementptr inbounds i8, ptr %v0, i64 %v93
  %v97 = load i8, ptr %v96, align 1
  %v98 = shl i8 %v97, 4
  %3 = and i8 %v98, 48
  %v1025 = or disjoint i8 %3, %v91
  %v102 = zext nneg i8 %v1025 to i32
  %v103 = add nsw i32 %v102, -32
  %v105 = add i64 %v86, 32
  %v106 = icmp ult i64 %v105, %v1
  br i1 %v106, label %bb16, label %bb49

bb16:                                             ; preds = %bb15
  %v108 = getelementptr inbounds i8, ptr %v0, i64 %v105
  %v109 = load i8, ptr %v108, align 1
  %v110 = and i8 %v109, 15
  %4 = shl i8 %v97, 2
  %5 = and i8 %4, 48
  %v1246 = or disjoint i8 %v110, %5
  %v124 = zext nneg i8 %v1246 to i32
  %v125 = add nsw i32 %v124, -32
  %v133 = lshr i8 %v90, 4
  %v143 = and i8 %v97, 48
  %v1477 = or disjoint i8 %v143, %v133
  %v147 = zext nneg i8 %v1477 to i32
  %v148 = add nsw i32 %v147, -32
  %v157 = lshr i8 %v109, 4
  %6 = lshr i8 %v97, 2
  %7 = and i8 %6, 48
  %v1718 = or disjoint i8 %v157, %7
  %v171 = zext nneg i8 %v1718 to i32
  %v172 = add nsw i32 %v171, -32
  %v173 = add nuw nsw i64 %v854, %v78
  %v174 = icmp ult i64 %v173, %v1
  br i1 %v174, label %bb22, label %bb55

bb22:                                             ; preds = %bb16
  %v176 = getelementptr inbounds i8, ptr %v0, i64 %v173
  %v177 = load i8, ptr %v176, align 1
  %v179 = sitofp i8 %v177 to float
  %v180 = add i64 %v173, 2
  %v181 = icmp ult i64 %v180, %v1
  br i1 %v181, label %bb23, label %bb56

bb23:                                             ; preds = %bb22
  %v183 = getelementptr inbounds i8, ptr %v0, i64 %v180
  %v184 = load i8, ptr %v183, align 1
  %v186 = sitofp i8 %v184 to float
  %v187 = add i64 %v173, 4
  %v188 = icmp ult i64 %v187, %v1
  br i1 %v188, label %bb24, label %bb57

bb24:                                             ; preds = %bb23
  %v190 = getelementptr inbounds i8, ptr %v0, i64 %v187
  %v191 = load i8, ptr %v190, align 1
  %v193 = sitofp i8 %v191 to float
  %v194 = add i64 %v173, 6
  %v195 = icmp ult i64 %v194, %v1
  br i1 %v195, label %bb25, label %bb58

bb25:                                             ; preds = %bb24
  %v197 = getelementptr inbounds i8, ptr %v0, i64 %v194
  %v198 = load i8, ptr %v197, align 1
  %v200 = sitofp i8 %v198 to float
  %v204 = add nuw nsw i64 %v8256, %v80
  %v206 = icmp ult i64 %v204, %v3
  br i1 %v206, label %bb26, label %bb59

bb26:                                             ; preds = %bb25
  %v216 = or disjoint i64 %v204, 32
  %v217 = icmp ult i64 %v216, %v3
  br i1 %v217, label %bb27, label %bb60

bb27:                                             ; preds = %bb26
  %v227 = or disjoint i64 %v204, 64
  %v228 = icmp ult i64 %v227, %v3
  br i1 %v228, label %bb28, label %bb61

bb28:                                             ; preds = %bb27
  %v238 = or disjoint i64 %v204, 96
  %v239 = icmp ult i64 %v238, %v3
  br i1 %v239, label %bb29, label %bb62

bb29:                                             ; preds = %bb28
  %v234 = fmul contract float %v55.i, %v200
  %v235 = sitofp i32 %v172 to float
  %v236 = fmul contract float %v234, %v235
  %v201 = fmul contract float %v55.i, %v179
  %v202 = sitofp i32 %v103 to float
  %v203 = fmul contract float %v201, %v202
  %v208 = getelementptr inbounds float, ptr %v2, i64 %v204
  %v209 = load float, ptr %v208, align 4
  %v210 = fmul contract float %v203, %v209
  %v211 = fadd contract float %v8155, %v210
  %v212 = fmul contract float %v55.i, %v186
  %v213 = sitofp i32 %v125 to float
  %v214 = fmul contract float %v212, %v213
  %v219 = getelementptr inbounds float, ptr %v2, i64 %v216
  %v220 = load float, ptr %v219, align 4
  %v221 = fmul contract float %v214, %v220
  %v222 = fadd contract float %v211, %v221
  %v223 = fmul contract float %v55.i, %v193
  %v224 = sitofp i32 %v148 to float
  %v225 = fmul contract float %v223, %v224
  %v230 = getelementptr inbounds float, ptr %v2, i64 %v227
  %v231 = load float, ptr %v230, align 4
  %v232 = fmul contract float %v225, %v231
  %v233 = fadd contract float %v222, %v232
  %v241 = getelementptr inbounds float, ptr %v2, i64 %v238
  %v242 = load float, ptr %v241, align 4
  %v243 = fmul contract float %v236, %v242
  %v244 = fadd contract float %v233, %v243
  %v245 = add nuw nsw i64 %v8256, 1
  %exitcond = icmp eq i64 %v245, 32
  br i1 %exitcond, label %bb30, label %bb13

bb30:                                             ; preds = %bb29
  br i1 %v69.not, label %bb11, label %bb31

bb31:                                             ; preds = %bb30
  %v247 = add nuw i32 %v3861, 1
  %exitcond62.not = icmp eq i32 %v247, %v5
  br i1 %exitcond62.not, label %bb32, label %bb6

bb32:                                             ; preds = %bb31, %bb4
  %v37.lcssa = phi float [ 0.000000e+00, %bb4 ], [ %v244, %bb31 ]
  %v251 = icmp ult i64 %v22.i, %v8
  %or.cond.not = select i1 %.v18.i, i1 %v251, i1 false
  %v2659 = icmp ne ptr %v7, null
  %v265 = select i1 %or.cond.not, i1 %v2659, i1 false
  br i1 %v265, label %bb33, label %bb36

bb33:                                             ; preds = %bb32
  %v254 = getelementptr inbounds nuw float, ptr %v7, i64 %v22.i
  store float %v37.lcssa, ptr %v254, align 4
  br label %bb36

bb36:                                             ; preds = %bb32, %bb33, %entry
  ret void

bb44:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb48:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb49:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb55:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb56:                                             ; preds = %bb22
  tail call void @llvm.trap() #19
  unreachable

bb57:                                             ; preds = %bb23
  tail call void @llvm.trap() #19
  unreachable

bb58:                                             ; preds = %bb24
  tail call void @llvm.trap() #19
  unreachable

bb59:                                             ; preds = %bb25
  tail call void @llvm.trap() #19
  unreachable

bb60:                                             ; preds = %bb26
  tail call void @llvm.trap() #19
  unreachable

bb61:                                             ; preds = %bb27
  tail call void @llvm.trap() #19
  unreachable

bb62:                                             ; preds = %bb28
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q6k_gemv_warp4(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(none) %v7, i64 %v8) #6 {
entry:
  %v21 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v22 = zext nneg i32 %v21 to i64
  %v23 = zext i32 %v4 to i64
  %v24 = add nuw nsw i64 %v23, 3
  %v251 = lshr i64 %v24, 2
  %v26 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v27 = zext nneg i32 %v26 to i64
  %v28 = zext i32 %v6 to i64
  %v29 = mul nuw nsw i64 %v251, %v28
  %v30.not = icmp samesign ugt i64 %v29, %v27
  br i1 %v30.not, label %bb4, label %bb55

bb4:                                              ; preds = %entry
  %v32.not = icmp eq i64 %v251, 0
  br i1 %v32.not, label %bb56, label %bb5

bb5:                                              ; preds = %bb4
  %v34.rhs.trunc = trunc nuw nsw i64 %v251 to i32
  %v34.rhs.trunc.frozen = freeze i32 %v34.rhs.trunc
  %v343 = udiv i32 %v26, %v34.rhs.trunc.frozen
  %v34.zext = zext nneg i32 %v343 to i64
  %0 = mul i32 %v343, %v34.rhs.trunc.frozen
  %v354.decomposed = sub i32 %v26, %0
  %v35.zext = zext nneg i32 %v354.decomposed to i64
  %v36 = shl nuw nsw i64 %v35.zext, 2
  %v37 = zext i32 %v5 to i64
  %v38 = mul nuw nsw i64 %v37, 210
  %v39 = shl nuw nsw i64 %v37, 8
  %v40 = mul i64 %v39, %v34.zext
  %v41.not = icmp samesign ult i64 %v36, %v23
  br i1 %v41.not, label %bb6, label %bb9

bb6:                                              ; preds = %bb5
  %v43 = mul i64 %v36, %v38
  %v48 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v0, i64 %v1, i64 %v43, ptr %v2, i64 %v3, i64 %v40, i32 %v5, i64 %v22) #19
  br label %bb9

bb9:                                              ; preds = %bb5, %bb6
  %v49 = phi float [ %v48, %bb6 ], [ 0.000000e+00, %bb5 ]
  %v50 = or disjoint i64 %v36, 1
  %v51.not = icmp samesign ult i64 %v50, %v23
  br i1 %v51.not, label %bb10, label %bb13

bb10:                                             ; preds = %bb9
  %v53 = mul i64 %v50, %v38
  %v58 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v0, i64 %v1, i64 %v53, ptr %v2, i64 %v3, i64 %v40, i32 %v5, i64 %v22) #19
  br label %bb13

bb13:                                             ; preds = %bb9, %bb10
  %v59 = phi float [ %v58, %bb10 ], [ 0.000000e+00, %bb9 ]
  %v60 = or disjoint i64 %v36, 2
  %v61.not = icmp samesign ult i64 %v60, %v23
  br i1 %v61.not, label %bb14, label %bb17

bb14:                                             ; preds = %bb13
  %v63 = mul i64 %v60, %v38
  %v68 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v0, i64 %v1, i64 %v63, ptr %v2, i64 %v3, i64 %v40, i32 %v5, i64 %v22) #19
  br label %bb17

bb17:                                             ; preds = %bb13, %bb14
  %v69 = phi float [ %v68, %bb14 ], [ 0.000000e+00, %bb13 ]
  %v70 = or disjoint i64 %v36, 3
  %v71.not = icmp samesign ult i64 %v70, %v23
  br i1 %v71.not, label %bb18, label %bb21

bb18:                                             ; preds = %bb17
  %v73 = mul i64 %v70, %v38
  %v78 = tail call fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr %v0, i64 %v1, i64 %v73, ptr %v2, i64 %v3, i64 %v40, i32 %v5, i64 %v22) #19
  br label %bb21

bb21:                                             ; preds = %bb17, %bb18
  %v79 = phi float [ %v78, %bb18 ], [ 0.000000e+00, %bb17 ]
  %v80 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_0, i64 %v22
  store float %v49, ptr addrspace(3) %v80, align 4
  %v82 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 128
  store float %v59, ptr addrspace(3) %v82, align 4
  %v84 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 256
  store float %v69, ptr addrspace(3) %v84, align 4
  %v86 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 384
  store float %v79, ptr addrspace(3) %v86, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not = icmp samesign ult i32 %v21, 16
  br i1 %v91.not, label %bb29, label %bb39

bb29:                                             ; preds = %bb21
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 64
  %v96 = load float, ptr addrspace(3) %gep, align 4
  %v98 = load float, ptr addrspace(3) %v80, align 4
  %v99 = fadd contract float %v96, %v98
  store float %v99, ptr addrspace(3) %v80, align 4
  %v102 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 192
  %v103 = load float, ptr addrspace(3) %v102, align 4
  %v105 = load float, ptr addrspace(3) %v82, align 4
  %v106 = fadd contract float %v103, %v105
  store float %v106, ptr addrspace(3) %v82, align 4
  %v109 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 320
  %v110 = load float, ptr addrspace(3) %v109, align 4
  %v112 = load float, ptr addrspace(3) %v84, align 4
  %v113 = fadd contract float %v110, %v112
  store float %v113, ptr addrspace(3) %v84, align 4
  %v116 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 448
  %v117 = load float, ptr addrspace(3) %v116, align 4
  %v119 = load float, ptr addrspace(3) %v86, align 4
  %v120 = fadd contract float %v117, %v119
  store float %v120, ptr addrspace(3) %v86, align 4
  br label %bb39

bb39:                                             ; preds = %bb21, %bb29
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.1 = icmp samesign ult i32 %v21, 8
  br i1 %v91.not.1, label %bb29.1, label %bb39.1

bb29.1:                                           ; preds = %bb39
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 32
  %v96.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v98.1 = load float, ptr addrspace(3) %v80, align 4
  %v99.1 = fadd contract float %v96.1, %v98.1
  store float %v99.1, ptr addrspace(3) %v80, align 4
  %v102.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 160
  %v103.1 = load float, ptr addrspace(3) %v102.1, align 4
  %v105.1 = load float, ptr addrspace(3) %v82, align 4
  %v106.1 = fadd contract float %v103.1, %v105.1
  store float %v106.1, ptr addrspace(3) %v82, align 4
  %v109.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 288
  %v110.1 = load float, ptr addrspace(3) %v109.1, align 4
  %v112.1 = load float, ptr addrspace(3) %v84, align 4
  %v113.1 = fadd contract float %v110.1, %v112.1
  store float %v113.1, ptr addrspace(3) %v84, align 4
  %v116.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 416
  %v117.1 = load float, ptr addrspace(3) %v116.1, align 4
  %v119.1 = load float, ptr addrspace(3) %v86, align 4
  %v120.1 = fadd contract float %v117.1, %v119.1
  store float %v120.1, ptr addrspace(3) %v86, align 4
  br label %bb39.1

bb39.1:                                           ; preds = %bb29.1, %bb39
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.2 = icmp samesign ult i32 %v21, 4
  br i1 %v91.not.2, label %bb29.2, label %bb39.2

bb29.2:                                           ; preds = %bb39.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 16
  %v96.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v98.2 = load float, ptr addrspace(3) %v80, align 4
  %v99.2 = fadd contract float %v96.2, %v98.2
  store float %v99.2, ptr addrspace(3) %v80, align 4
  %v102.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 144
  %v103.2 = load float, ptr addrspace(3) %v102.2, align 4
  %v105.2 = load float, ptr addrspace(3) %v82, align 4
  %v106.2 = fadd contract float %v103.2, %v105.2
  store float %v106.2, ptr addrspace(3) %v82, align 4
  %v109.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 272
  %v110.2 = load float, ptr addrspace(3) %v109.2, align 4
  %v112.2 = load float, ptr addrspace(3) %v84, align 4
  %v113.2 = fadd contract float %v110.2, %v112.2
  store float %v113.2, ptr addrspace(3) %v84, align 4
  %v116.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 400
  %v117.2 = load float, ptr addrspace(3) %v116.2, align 4
  %v119.2 = load float, ptr addrspace(3) %v86, align 4
  %v120.2 = fadd contract float %v117.2, %v119.2
  store float %v120.2, ptr addrspace(3) %v86, align 4
  br label %bb39.2

bb39.2:                                           ; preds = %bb29.2, %bb39.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.3 = icmp samesign ult i32 %v21, 2
  br i1 %v91.not.3, label %bb29.3, label %bb39.3

bb29.3:                                           ; preds = %bb39.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 8
  %v96.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v98.3 = load float, ptr addrspace(3) %v80, align 4
  %v99.3 = fadd contract float %v96.3, %v98.3
  store float %v99.3, ptr addrspace(3) %v80, align 4
  %v102.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 136
  %v103.3 = load float, ptr addrspace(3) %v102.3, align 4
  %v105.3 = load float, ptr addrspace(3) %v82, align 4
  %v106.3 = fadd contract float %v103.3, %v105.3
  store float %v106.3, ptr addrspace(3) %v82, align 4
  %v109.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 264
  %v110.3 = load float, ptr addrspace(3) %v109.3, align 4
  %v112.3 = load float, ptr addrspace(3) %v84, align 4
  %v113.3 = fadd contract float %v110.3, %v112.3
  store float %v113.3, ptr addrspace(3) %v84, align 4
  %v116.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 392
  %v117.3 = load float, ptr addrspace(3) %v116.3, align 4
  %v119.3 = load float, ptr addrspace(3) %v86, align 4
  %v120.3 = fadd contract float %v117.3, %v119.3
  store float %v120.3, ptr addrspace(3) %v86, align 4
  br label %bb39.3

bb39.3:                                           ; preds = %bb29.3, %bb39.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.4 = icmp eq i32 %v21, 0
  br i1 %v91.not.4, label %bb29.4, label %bb39.4

bb29.4:                                           ; preds = %bb39.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 4
  %v96.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v98.4 = load float, ptr addrspace(3) %v80, align 4
  %v99.4 = fadd contract float %v96.4, %v98.4
  store float %v99.4, ptr addrspace(3) %v80, align 4
  %v102.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 132
  %v103.4 = load float, ptr addrspace(3) %v102.4, align 4
  %v105.4 = load float, ptr addrspace(3) %v82, align 4
  %v106.4 = fadd contract float %v103.4, %v105.4
  store float %v106.4, ptr addrspace(3) %v82, align 4
  %v109.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 260
  %v110.4 = load float, ptr addrspace(3) %v109.4, align 4
  %v112.4 = load float, ptr addrspace(3) %v84, align 4
  %v113.4 = fadd contract float %v110.4, %v112.4
  store float %v113.4, ptr addrspace(3) %v84, align 4
  %v116.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v80, i64 388
  %v117.4 = load float, ptr addrspace(3) %v116.4, align 4
  %v119.4 = load float, ptr addrspace(3) %v86, align 4
  %v120.4 = fadd contract float %v117.4, %v119.4
  store float %v120.4, ptr addrspace(3) %v86, align 4
  br label %bb39.4

bb39.4:                                           ; preds = %bb29.4, %bb39.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v123 = icmp eq i32 %v21, 0
  br i1 %v123, label %bb42, label %bb55

bb42:                                             ; preds = %bb39.4
  %v124 = mul nuw nsw i64 %v34.zext, %v23
  %v125 = add nuw nsw i64 %v36, %v124
  br i1 %v41.not, label %bb43, label %bb45

bb43:                                             ; preds = %bb42
  %v131 = getelementptr inbounds nuw float, ptr %v7, i64 %v125
  %v129 = load float, ptr addrspace(3) @__shared_mem_0, align 4
  store float %v129, ptr %v131, align 4
  br label %bb45

bb45:                                             ; preds = %bb43, %bb42
  br i1 %v51.not, label %bb46, label %bb48

bb46:                                             ; preds = %bb45
  %1 = getelementptr inbounds nuw float, ptr %v7, i64 %v125
  %v138 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %v135 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_0, i64 128), align 4
  store float %v135, ptr %v138, align 4
  br label %bb48

bb48:                                             ; preds = %bb46, %bb45
  br i1 %v61.not, label %bb49, label %bb51

bb49:                                             ; preds = %bb48
  %2 = getelementptr inbounds nuw float, ptr %v7, i64 %v125
  %v145 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %v142 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_0, i64 256), align 4
  store float %v142, ptr %v145, align 4
  br label %bb51

bb51:                                             ; preds = %bb49, %bb48
  br i1 %v71.not, label %bb52, label %bb55

bb52:                                             ; preds = %bb51
  %3 = getelementptr inbounds nuw float, ptr %v7, i64 %v125
  %v152 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %v149 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_0, i64 384), align 4
  store float %v149, ptr %v152, align 4
  br label %bb55

bb55:                                             ; preds = %bb39.4, %bb51, %bb52, %entry
  ret void

bb56:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q6k_q8_gemv_multiwarp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, ptr readonly captures(none) %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, ptr writeonly captures(none) %v12, i64 %v13) #6 {
entry:
  %v35 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v36 = zext nneg i32 %v35 to i64
  %v37 = and i64 %v36, 31
  %v40 = lshr i64 %v36, 5
  %v41 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %0 = lshr i32 %v41, 5
  %v45 = zext nneg i32 %0 to i64
  %v46 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v47 = zext nneg i32 %v46 to i64
  %v48 = zext i32 %v9 to i64
  %v49 = zext i32 %v11 to i64
  %v50 = mul nuw i64 %v49, %v48
  %v51.not = icmp ugt i64 %v50, %v47
  %v53.not = icmp samesign ult i64 %v40, %v45
  %or.cond = select i1 %v51.not, i1 %v53.not, i1 false
  br i1 %or.cond, label %bb7, label %bb37

bb7:                                              ; preds = %entry
  %v55.not = icmp eq i32 %v9, 0
  br i1 %v55.not, label %bb40, label %bb8

bb8:                                              ; preds = %bb7
  %v9.frozen = freeze i32 %v9
  %v586 = udiv i32 %v46, %v9.frozen
  %v58.zext = zext nneg i32 %v586 to i64
  %v59 = zext i32 %v10 to i64
  %v61 = mul nuw nsw i64 %v58.zext, %v59
  %v65.not9 = icmp samesign ult i64 %v40, %v59
  br i1 %v65.not9, label %bb10.lr.ph, label %bb16.preheader

bb10.lr.ph:                                       ; preds = %bb8
  %v60 = mul nuw nsw i64 %v59, 210
  %1 = mul i32 %v586, %v9.frozen
  %v575.decomposed = sub i32 %v46, %1
  %v57.zext = zext nneg i32 %v575.decomposed to i64
  %v67 = shl nuw nsw i64 %v37, 3
  %v92 = mul i64 %v60, %v57.zext
  %v91.i.i = lshr i64 %v37, 4
  %v13.i.i = shl nuw nsw i64 %v91.i.i, 6
  %v16.i.i = shl nuw nsw i64 %v91.i.i, 5
  %2 = getelementptr i8, ptr %v0, i64 %v92
  %v10.i48.i = lshr i64 %v36, 2
  %v122.i49.i = and i64 %v10.i48.i, 3
  %v122.tr.i52.i = trunc nuw nsw i64 %v122.i49.i to i8
  %v19.i53.i = shl nuw nsw i8 %v122.tr.i52.i, 1
  %v30.i57.i = icmp samesign ugt i64 %v122.i49.i, 1
  %3 = lshr i64 %v37, 1
  %v63.i = and i64 %3, 8
  %v653.i = and i64 %3, 1
  %v67.i = and i64 %3, 6
  br label %bb10

bb16.preheader:                                   ; preds = %bb14, %bb8
  %v63.lcssa = phi float [ 0.000000e+00, %bb8 ], [ %v98, %bb14 ]
  %v105 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v63.lcssa, i32 16, i32 31) #19
  %v133 = fadd contract float %v63.lcssa, %v105
  %v105.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v133, i32 8, i32 31) #19
  %v133.1 = fadd contract float %v133, %v105.1
  %v105.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v133.1, i32 4, i32 31) #19
  %v133.2 = fadd contract float %v133.1, %v105.2
  %v105.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v133.2, i32 2, i32 31) #19
  %v133.3 = fadd contract float %v133.2, %v105.3
  %v105.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v133.3, i32 1, i32 31) #19
  %v106.not = icmp eq i64 %v37, 0
  br i1 %v106.not, label %bb19, label %bb21

bb10:                                             ; preds = %bb10.lr.ph, %bb14
  %v6411 = phi i64 [ %v40, %bb10.lr.ph ], [ %v100, %bb14 ]
  %v6310 = phi float [ 0.000000e+00, %bb10.lr.ph ], [ %v98, %bb14 ]
  %v621 = add i64 %v6411, %v61
  %v75 = shl i64 %v621, 8
  %v93 = mul i64 %v6411, 210
  %4 = getelementptr i8, ptr %2, i64 %v93
  %5 = getelementptr i8, ptr %4, i64 %v13.i.i
  %6 = getelementptr i8, ptr %4, i64 128
  %7 = getelementptr i8, ptr %6, i64 %v16.i.i
  %v45.i = getelementptr i8, ptr %4, i64 208
  %v46.i = load i8, ptr %v45.i, align 1
  %v50.i = getelementptr i8, ptr %4, i64 209
  %v51.i = load i8, ptr %v50.i, align 1
  %v55.sroa.2.0.insert.ext.i = zext i8 %v51.i to i16
  %v55.sroa.2.0.insert.shift.i = shl nuw i16 %v55.sroa.2.0.insert.ext.i, 8
  %v55.sroa.0.0.insert.ext.i = zext i8 %v46.i to i16
  %v4.i.i = lshr i16 %v55.sroa.2.0.insert.ext.i, 7
  %v6.i.i = zext nneg i16 %v4.i.i to i32
  %v9.i.i = lshr i16 %v55.sroa.2.0.insert.ext.i, 2
  %v10.i67.i = and i16 %v9.i.i, 31
  %v55.sroa.2.0.insert.shift.masked.i = and i16 %v55.sroa.2.0.insert.shift.i, 768
  %v12.i.i = or disjoint i16 %v55.sroa.2.0.insert.shift.masked.i, %v55.sroa.0.0.insert.ext.i
  %v13.i68.i = zext nneg i16 %v12.i.i to i32
  %8 = getelementptr i8, ptr %4, i64 192
  %9 = getelementptr i8, ptr %8, i64 %v63.i
  %10 = getelementptr i8, ptr %9, i64 %v653.i
  %v71.i = getelementptr i8, ptr %10, i64 %v67.i
  %v72.i = load i8, ptr %v71.i, align 1
  %v74.i = sitofp i8 %v72.i to float
  %v38.i.i = shl nuw i32 %v6.i.i, 31
  %v41.i69.i = shl nuw nsw i32 %v13.i68.i, 13
  %v39.i.i = or disjoint i32 %v41.i69.i, %v38.i.i
  %v42.i70.i = or disjoint i32 %v39.i.i, 2139095040
  %v15.i.i = icmp eq i16 %v12.i.i, 0
  %v13.masked.numleadingzeros.i.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i68.i, i1 true)
  %v13.masked.leadingonepos.i.i = xor i32 %v13.masked.numleadingzeros.i.i, 31
  %bb5.tripcount.i.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i
  %v23.i71.i = shl nuw nsw i32 %v13.i68.i, %bb5.tripcount.i.i
  %reass.sub.i = or disjoint i32 %v38.i.i, 1124073472
  %11 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i, 23
  %v31.i.i = sub nuw nsw i32 %reass.sub.i, %11
  %v25.i.i = shl i32 %v23.i71.i, 13
  %v33.i.i = and i32 %v25.i.i, 8380416
  %v34.i.i = or disjoint i32 %v31.i.i, %v33.i.i
  %12 = add nuw nsw i16 %v10.i67.i, 112
  %v46.i73.i = zext nneg i16 %12 to i32
  %v48.i.i = shl nuw nsw i32 %v46.i73.i, 23
  %v49.i.i = or disjoint i32 %v48.i.i, %v38.i.i
  %v52.i.i = or disjoint i32 %v49.i.i, %v41.i69.i
  %v17.i.i.v34.i.i = select i1 %v15.i.i, i32 %v38.i.i, i32 %v34.i.i
  br label %bb12

bb12:                                             ; preds = %bb10, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit
  %v70.not = phi i1 [ true, %bb10 ], [ false, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit ]
  %v698 = phi i64 [ 0, %bb10 ], [ 4, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit ]
  %v687 = phi float [ %v6310, %bb10 ], [ %v98, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit ]
  %v73 = or disjoint i64 %v698, %v67
  %v76 = or disjoint i64 %v73, %v75
  %v78 = getelementptr inbounds i8, ptr %v2, i64 %v76
  %v33.sroa.0.0.copyload = load i32, ptr %v78, align 1
  %v862 = lshr i64 %v76, 5
  %v90 = getelementptr inbounds nuw float, ptr %v4, i64 %v862
  %v91 = load float, ptr %v90, align 4
  %v11.i.i = and i64 %v73, 28
  %v23.i.i = and i64 %v73, 60
  %v28.i.i = getelementptr i8, ptr %5, i64 %v23.i.i
  %v29.i.i = load i8, ptr %v28.i.i, align 1
  %v41.i.i = getelementptr i8, ptr %7, i64 %v11.i.i
  %v42.i.i = load i8, ptr %v41.i.i, align 1
  %v19.i = or disjoint i64 %v73, 1
  %v11.i5.i = and i64 %v19.i, 29
  %v23.i12.i = and i64 %v19.i, 61
  %v28.i13.i = getelementptr i8, ptr %5, i64 %v23.i12.i
  %v29.i14.i = load i8, ptr %v28.i13.i, align 1
  %v41.i19.i = getelementptr i8, ptr %7, i64 %v11.i5.i
  %v42.i20.i = load i8, ptr %v41.i19.i, align 1
  %v24.i = or disjoint i64 %v73, 2
  %v11.i26.i = and i64 %v24.i, 30
  %v23.i33.i = and i64 %v24.i, 62
  %v28.i34.i = getelementptr i8, ptr %5, i64 %v23.i33.i
  %v29.i35.i = load i8, ptr %v28.i34.i, align 1
  %v41.i40.i = getelementptr i8, ptr %7, i64 %v11.i26.i
  %v42.i41.i = load i8, ptr %v41.i40.i, align 1
  %v29.i = or disjoint i64 %v73, 3
  %v11.i47.i = and i64 %v29.i, 31
  %v23.i54.i = and i64 %v29.i, 63
  %v28.i55.i = getelementptr i8, ptr %5, i64 %v23.i54.i
  %v29.i56.i = load i8, ptr %v28.i55.i, align 1
  %v41.i61.i = getelementptr i8, ptr %7, i64 %v11.i47.i
  %v42.i62.i = load i8, ptr %v41.i61.i, align 1
  switch i16 %v10.i67.i, label %bb10.i.i [
    i16 0, label %bb1.i.i
    i16 31, label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit
  ]

bb1.i.i:                                          ; preds = %bb12
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit

bb10.i.i:                                         ; preds = %bb12
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit

cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit: ; preds = %bb12, %bb1.i.i, %bb10.i.i
  %v54.i.i = phi i32 [ %v52.i.i, %bb10.i.i ], [ %v17.i.i.v34.i.i, %bb1.i.i ], [ %v42.i70.i, %bb12 ]
  %v45.i63.i = lshr i8 %v42.i62.i, %v19.i53.i
  %v46.i64.i = shl i8 %v45.i63.i, 4
  %13 = and i8 %v46.i64.i, 48
  %v50.i65.i = add nsw i8 %13, -32
  %v35.i59.i = lshr i8 %v29.i56.i, 4
  %v32.i58.i = and i8 %v29.i56.i, 15
  %v36.i60.i = select i1 %v30.i57.i, i8 %v35.i59.i, i8 %v32.i58.i
  %v51.i66.i = or disjoint i8 %v50.i65.i, %v36.i60.i
  %v39.sroa.4.0.insert.ext.i = zext i8 %v51.i66.i to i32
  %v39.sroa.4.0.insert.shift.i = shl nuw i32 %v39.sroa.4.0.insert.ext.i, 24
  %v45.i42.i = lshr i8 %v42.i41.i, %v19.i53.i
  %v46.i43.i = shl i8 %v45.i42.i, 4
  %14 = and i8 %v46.i43.i, 48
  %v50.i44.i = add nsw i8 %14, -32
  %v35.i38.i = lshr i8 %v29.i35.i, 4
  %v32.i37.i = and i8 %v29.i35.i, 15
  %v36.i39.i = select i1 %v30.i57.i, i8 %v35.i38.i, i8 %v32.i37.i
  %v51.i45.i = or disjoint i8 %v50.i44.i, %v36.i39.i
  %v39.sroa.3.0.insert.ext.i = zext i8 %v51.i45.i to i32
  %v39.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v39.sroa.3.0.insert.ext.i, 16
  %v39.sroa.3.0.insert.insert.i = or disjoint i32 %v39.sroa.4.0.insert.shift.i, %v39.sroa.3.0.insert.shift.i
  %v45.i21.i = lshr i8 %v42.i20.i, %v19.i53.i
  %v46.i22.i = shl i8 %v45.i21.i, 4
  %15 = and i8 %v46.i22.i, 48
  %v50.i23.i = add nsw i8 %15, -32
  %v35.i17.i = lshr i8 %v29.i14.i, 4
  %v32.i16.i = and i8 %v29.i14.i, 15
  %v36.i18.i = select i1 %v30.i57.i, i8 %v35.i17.i, i8 %v32.i16.i
  %v51.i24.i = or disjoint i8 %v50.i23.i, %v36.i18.i
  %v39.sroa.2.0.insert.ext.i = zext i8 %v51.i24.i to i32
  %v39.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v39.sroa.2.0.insert.ext.i, 8
  %v39.sroa.2.0.insert.insert.i = or disjoint i32 %v39.sroa.3.0.insert.insert.i, %v39.sroa.2.0.insert.shift.i
  %v45.i.i = lshr i8 %v42.i.i, %v19.i53.i
  %v46.i.i = shl i8 %v45.i.i, 4
  %16 = and i8 %v46.i.i, 48
  %v50.i.i = add nsw i8 %16, -32
  %v35.i.i = lshr i8 %v29.i.i, 4
  %v32.i.i = and i8 %v29.i.i, 15
  %v36.i.i = select i1 %v30.i57.i, i8 %v35.i.i, i8 %v32.i.i
  %v51.i.i = or disjoint i8 %v50.i.i, %v36.i.i
  %v39.sroa.0.0.insert.ext.i = zext i8 %v51.i.i to i32
  %v39.sroa.0.0.insert.insert.i = or disjoint i32 %v39.sroa.2.0.insert.insert.i, %v39.sroa.0.0.insert.ext.i
  %v55.i.i = bitcast i32 %v54.i.i to float
  %v75.i = fmul contract float %v55.i.i, %v74.i
  %v76.i = fmul contract float %v91, %v75.i
  %v77.i = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v39.sroa.0.0.insert.insert.i, i32 %v33.sroa.0.0.copyload, i32 0) #19
  %v78.i = sitofp i32 %v77.i to float
  %v79.i = fmul contract float %v76.i, %v78.i
  %v98 = fadd contract float %v687, %v79.i
  br i1 %v70.not, label %bb12, label %bb14

bb14:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit
  %v100 = add i64 %v6411, %v45
  %v65.not = icmp ult i64 %v100, %v59
  br i1 %v65.not, label %bb10, label %bb16.preheader

bb19:                                             ; preds = %bb16.preheader
  %v133.4 = fadd contract float %v133.3, %v105.4
  %v108 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_10, i64 %v40
  store float %v133.4, ptr addrspace(3) %v108, align 4
  br label %bb21

bb21:                                             ; preds = %bb19, %bb16.preheader
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v110 = icmp eq i64 %v40, 0
  br i1 %v110, label %bb23, label %bb37

bb23:                                             ; preds = %bb21
  %v111.not = icmp samesign ult i64 %v37, %v45
  br i1 %v111.not, label %bb24, label %bb27

bb24:                                             ; preds = %bb23
  %v114 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_10, i64 %v37
  %v115 = load float, ptr addrspace(3) %v114, align 4
  br label %bb27

bb27:                                             ; preds = %bb23, %bb24
  %v116 = phi float [ %v115, %bb24 ], [ 0.000000e+00, %bb23 ]
  %v121 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v116, i32 16, i32 31) #19
  %v135 = fadd contract float %v116, %v121
  %v121.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v135, i32 8, i32 31) #19
  %v135.1 = fadd contract float %v135, %v121.1
  %v121.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v135.1, i32 4, i32 31) #19
  %v135.2 = fadd contract float %v135.1, %v121.2
  %v121.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v135.2, i32 2, i32 31) #19
  %v135.3 = fadd contract float %v135.2, %v121.3
  %v121.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v135.3, i32 1, i32 31) #19
  %v135.4 = fadd contract float %v135.3, %v121.4
  br i1 %v106.not, label %bb31, label %bb37

bb31:                                             ; preds = %bb27
  %v123 = icmp eq i32 %v8, 0
  br i1 %v123, label %bb34, label %bb32

bb32:                                             ; preds = %bb31
  %v127 = getelementptr inbounds nuw float, ptr %v6, i64 %v47
  %v128 = load float, ptr %v127, align 4
  br label %bb34

bb34:                                             ; preds = %bb31, %bb32
  %v129 = phi float [ %v128, %bb32 ], [ 0.000000e+00, %bb31 ]
  %v131 = getelementptr inbounds nuw float, ptr %v12, i64 %v47
  %v132 = fadd contract float %v135.4, %v129
  store float %v132, ptr %v131, align 4
  br label %bb37

bb37:                                             ; preds = %bb21, %bb34, %bb27, %entry
  ret void

bb40:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define ptx_kernel void @q6k_q8_gemv_warp4(ptr readonly %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr readonly captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, ptr writeonly captures(none) %v9, i64 %v10) #3 {
entry:
  %v27 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v29 = zext i32 %v6 to i64
  %v30 = add nuw nsw i64 %v29, 3
  %v311 = lshr i64 %v30, 2
  %v32 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v33 = zext nneg i32 %v32 to i64
  %v34 = zext i32 %v8 to i64
  %v35 = mul nuw nsw i64 %v311, %v34
  %v36.not = icmp samesign ugt i64 %v35, %v33
  br i1 %v36.not, label %bb4, label %bb40

bb4:                                              ; preds = %entry
  %v38.not = icmp eq i64 %v311, 0
  br i1 %v38.not, label %bb45, label %bb5

bb5:                                              ; preds = %bb4
  %v40.rhs.trunc = trunc nuw nsw i64 %v311 to i32
  %v40.rhs.trunc.frozen = freeze i32 %v40.rhs.trunc
  %v40467 = udiv i32 %v32, %v40.rhs.trunc.frozen
  %v40.zext = zext nneg i32 %v40467 to i64
  %0 = mul i32 %v40467, %v40.rhs.trunc.frozen
  %v41468.decomposed = sub i32 %v32, %0
  %v41.zext = zext nneg i32 %v41468.decomposed to i64
  %v42 = shl nuw nsw i64 %v41.zext, 2
  %v43 = zext i32 %v7 to i64
  %v45 = mul nuw nsw i64 %v40.zext, %v43
  %v52.not477.not = icmp eq i32 %v7, 0
  br i1 %v52.not477.not, label %bb24.preheader, label %bb7.lr.ph

bb7.lr.ph:                                        ; preds = %bb5
  %v44 = mul nuw nsw i64 %v43, 210
  %1 = shl nuw nsw i32 %v27, 3
  %v54 = zext nneg i32 %1 to i64
  %v104.not = icmp samesign ult i64 %v42, %v29
  %v114 = or disjoint i64 %v42, 1
  %v115.not = icmp samesign ult i64 %v114, %v29
  %v125 = or disjoint i64 %v42, 2
  %v126.not = icmp samesign ult i64 %v125, %v29
  %v136 = or disjoint i64 %v42, 3
  %v137.not = icmp samesign ult i64 %v136, %v29
  %v106 = mul i64 %v42, %v44
  %v91.i.i = lshr i64 %v54, 7
  %v13.i.i = shl nuw nsw i64 %v91.i.i, 6
  %v16.i.i = shl nuw nsw i64 %v91.i.i, 5
  %2 = getelementptr i8, ptr %v0, i64 %v106
  %v10.i48.i = lshr i64 %v54, 5
  %v122.i49.i = and i64 %v10.i48.i, 3
  %v122.tr.i52.i = trunc nuw nsw i64 %v122.i49.i to i8
  %v19.i53.i = shl nuw nsw i8 %v122.tr.i52.i, 1
  %v30.i57.i = icmp samesign ugt i64 %v122.i49.i, 1
  %3 = lshr i64 %v54, 4
  %v63.i = and i64 %3, 504
  %v653.i = and i64 %3, 1
  %v67.i = and i64 %3, 6
  %v117 = mul i64 %v114, %v44
  %4 = getelementptr i8, ptr %v0, i64 %v117
  %v128 = mul i64 %v125, %v44
  %5 = getelementptr i8, ptr %v0, i64 %v128
  %v139 = mul i64 %v136, %v44
  %6 = getelementptr i8, ptr %v0, i64 %v139
  br label %bb7

bb24.preheader:                                   ; preds = %bb22, %bb5
  %v47.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v113, %bb22 ]
  %v48.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v124, %bb22 ]
  %v49.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v135, %bb22 ]
  %v50.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v146, %bb22 ]
  %v156 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v47.lcssa, i32 16, i32 31) #19
  %v182 = fadd contract float %v47.lcssa, %v156
  %v183 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v48.lcssa, i32 16, i32 31) #19
  %v184 = fadd contract float %v48.lcssa, %v183
  %v185 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v49.lcssa, i32 16, i32 31) #19
  %v186 = fadd contract float %v49.lcssa, %v185
  %v187 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v50.lcssa, i32 16, i32 31) #19
  %v188 = fadd contract float %v50.lcssa, %v187
  %v156.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182, i32 8, i32 31) #19
  %v182.1 = fadd contract float %v182, %v156.1
  %v183.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184, i32 8, i32 31) #19
  %v184.1 = fadd contract float %v184, %v183.1
  %v185.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186, i32 8, i32 31) #19
  %v186.1 = fadd contract float %v186, %v185.1
  %v187.1 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188, i32 8, i32 31) #19
  %v188.1 = fadd contract float %v188, %v187.1
  %v156.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182.1, i32 4, i32 31) #19
  %v182.2 = fadd contract float %v182.1, %v156.2
  %v183.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184.1, i32 4, i32 31) #19
  %v184.2 = fadd contract float %v184.1, %v183.2
  %v185.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186.1, i32 4, i32 31) #19
  %v186.2 = fadd contract float %v186.1, %v185.2
  %v187.2 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188.1, i32 4, i32 31) #19
  %v188.2 = fadd contract float %v188.1, %v187.2
  %v156.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182.2, i32 2, i32 31) #19
  %v182.3 = fadd contract float %v182.2, %v156.3
  %v183.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184.2, i32 2, i32 31) #19
  %v184.3 = fadd contract float %v184.2, %v183.3
  %v185.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186.2, i32 2, i32 31) #19
  %v186.3 = fadd contract float %v186.2, %v185.3
  %v187.3 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188.2, i32 2, i32 31) #19
  %v188.3 = fadd contract float %v188.2, %v187.3
  %v156.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v182.3, i32 1, i32 31) #19
  %v182.4 = fadd contract float %v182.3, %v156.4
  %v183.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v184.3, i32 1, i32 31) #19
  %v184.4 = fadd contract float %v184.3, %v183.4
  %v185.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v186.3, i32 1, i32 31) #19
  %v186.4 = fadd contract float %v186.3, %v185.4
  %v187.4 = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %v188.3, i32 1, i32 31) #19
  %v188.4 = fadd contract float %v188.3, %v187.4
  %v157 = icmp eq i32 %v27, 0
  br i1 %v157, label %bb27, label %bb40

bb7:                                              ; preds = %bb7.lr.ph, %bb22
  %v51482 = phi i64 [ 0, %bb7.lr.ph ], [ %v148, %bb22 ]
  %v50481 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v146, %bb22 ]
  %v49480 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v135, %bb22 ]
  %v48479 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v124, %bb22 ]
  %v47478 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v113, %bb22 ]
  %v462 = add nuw nsw i64 %v51482, %v45
  %v65 = shl i64 %v462, 8
  %v107 = mul nuw nsw i64 %v51482, 210
  %7 = getelementptr i8, ptr %2, i64 %v107
  %8 = getelementptr i8, ptr %7, i64 %v13.i.i
  %9 = getelementptr i8, ptr %7, i64 128
  %10 = getelementptr i8, ptr %9, i64 %v16.i.i
  %v45.i = getelementptr i8, ptr %7, i64 208
  %v50.i = getelementptr i8, ptr %7, i64 209
  %11 = getelementptr i8, ptr %7, i64 192
  %12 = getelementptr i8, ptr %11, i64 %v63.i
  %13 = getelementptr i8, ptr %12, i64 %v653.i
  %v71.i = getelementptr i8, ptr %13, i64 %v67.i
  %14 = getelementptr i8, ptr %4, i64 %v107
  %15 = getelementptr i8, ptr %14, i64 %v13.i.i
  %16 = getelementptr i8, ptr %14, i64 128
  %17 = getelementptr i8, ptr %16, i64 %v16.i.i
  %v45.i50 = getelementptr i8, ptr %14, i64 208
  %v50.i52 = getelementptr i8, ptr %14, i64 209
  %18 = getelementptr i8, ptr %14, i64 192
  %19 = getelementptr i8, ptr %18, i64 %v63.i
  %20 = getelementptr i8, ptr %19, i64 %v653.i
  %v71.i132 = getelementptr i8, ptr %20, i64 %v67.i
  %21 = getelementptr i8, ptr %5, i64 %v107
  %22 = getelementptr i8, ptr %21, i64 %v13.i.i
  %23 = getelementptr i8, ptr %21, i64 128
  %24 = getelementptr i8, ptr %23, i64 %v16.i.i
  %v45.i202 = getelementptr i8, ptr %21, i64 208
  %v50.i204 = getelementptr i8, ptr %21, i64 209
  %25 = getelementptr i8, ptr %21, i64 192
  %26 = getelementptr i8, ptr %25, i64 %v63.i
  %27 = getelementptr i8, ptr %26, i64 %v653.i
  %v71.i284 = getelementptr i8, ptr %27, i64 %v67.i
  %28 = getelementptr i8, ptr %6, i64 %v107
  %29 = getelementptr i8, ptr %28, i64 %v13.i.i
  %30 = getelementptr i8, ptr %28, i64 128
  %31 = getelementptr i8, ptr %30, i64 %v16.i.i
  %v45.i354 = getelementptr i8, ptr %28, i64 208
  %v50.i356 = getelementptr i8, ptr %28, i64 209
  %32 = getelementptr i8, ptr %28, i64 192
  %33 = getelementptr i8, ptr %32, i64 %v63.i
  %34 = getelementptr i8, ptr %33, i64 %v653.i
  %v71.i436 = getelementptr i8, ptr %34, i64 %v67.i
  br label %bb9

bb9:                                              ; preds = %bb7, %bb21
  %v60.not = phi i1 [ true, %bb7 ], [ false, %bb21 ]
  %v59476 = phi i64 [ 0, %bb7 ], [ 4, %bb21 ]
  %v58475 = phi float [ %v50481, %bb7 ], [ %v146, %bb21 ]
  %v57474 = phi float [ %v49480, %bb7 ], [ %v135, %bb21 ]
  %v56473 = phi float [ %v48479, %bb7 ], [ %v124, %bb21 ]
  %v55472 = phi float [ %v47478, %bb7 ], [ %v113, %bb21 ]
  %v63 = or disjoint i64 %v59476, %v54
  %v66 = add i64 %v63, %v65
  %v70 = getelementptr inbounds i8, ptr %v2, i64 %v66
  %35 = load i32, ptr %v70, align 1
  %v989 = lshr i64 %v66, 5
  %v102 = getelementptr inbounds nuw float, ptr %v4, i64 %v989
  %v103 = load float, ptr %v102, align 4
  br i1 %v104.not, label %bb10, label %bb12

bb10:                                             ; preds = %bb9
  %v11.i.i = and i64 %v63, 28
  %v23.i.i = and i64 %v63, 60
  %v28.i.i = getelementptr i8, ptr %8, i64 %v23.i.i
  %v29.i.i = load i8, ptr %v28.i.i, align 1
  %v41.i.i = getelementptr i8, ptr %10, i64 %v11.i.i
  %v42.i.i = load i8, ptr %v41.i.i, align 1
  %v19.i = or disjoint i64 %v63, 1
  %v11.i5.i = and i64 %v19.i, 29
  %v23.i12.i = and i64 %v19.i, 61
  %v28.i13.i = getelementptr i8, ptr %8, i64 %v23.i12.i
  %v29.i14.i = load i8, ptr %v28.i13.i, align 1
  %v41.i19.i = getelementptr i8, ptr %10, i64 %v11.i5.i
  %v42.i20.i = load i8, ptr %v41.i19.i, align 1
  %v24.i = or disjoint i64 %v63, 2
  %v11.i26.i = and i64 %v24.i, 30
  %v23.i33.i = and i64 %v24.i, 62
  %v28.i34.i = getelementptr i8, ptr %8, i64 %v23.i33.i
  %v29.i35.i = load i8, ptr %v28.i34.i, align 1
  %v41.i40.i = getelementptr i8, ptr %10, i64 %v11.i26.i
  %v42.i41.i = load i8, ptr %v41.i40.i, align 1
  %v29.i = or disjoint i64 %v63, 3
  %v11.i47.i = and i64 %v29.i, 31
  %v23.i54.i = and i64 %v29.i, 63
  %v28.i55.i = getelementptr i8, ptr %8, i64 %v23.i54.i
  %v29.i56.i = load i8, ptr %v28.i55.i, align 1
  %v41.i61.i = getelementptr i8, ptr %10, i64 %v11.i47.i
  %v42.i62.i = load i8, ptr %v41.i61.i, align 1
  %v46.i = load i8, ptr %v45.i, align 1
  %v51.i = load i8, ptr %v50.i, align 1
  %v55.sroa.2.0.insert.ext.i = zext i8 %v51.i to i16
  %v55.sroa.2.0.insert.shift.i = shl nuw i16 %v55.sroa.2.0.insert.ext.i, 8
  %v55.sroa.0.0.insert.ext.i = zext i8 %v46.i to i16
  %v4.i.i = lshr i16 %v55.sroa.2.0.insert.ext.i, 7
  %v6.i.i = zext nneg i16 %v4.i.i to i32
  %v9.i.i = lshr i16 %v55.sroa.2.0.insert.ext.i, 2
  %v10.i67.i = and i16 %v9.i.i, 31
  %v55.sroa.2.0.insert.shift.masked.i = and i16 %v55.sroa.2.0.insert.shift.i, 768
  %v12.i.i = or disjoint i16 %v55.sroa.2.0.insert.shift.masked.i, %v55.sroa.0.0.insert.ext.i
  %v13.i68.i = zext nneg i16 %v12.i.i to i32
  switch i16 %v10.i67.i, label %bb10.i.i [
    i16 0, label %bb1.i.i
    i16 31, label %bb9.i.i
  ]

bb1.i.i:                                          ; preds = %bb10
  %v15.i.i = icmp eq i16 %v12.i.i, 0
  br i1 %v15.i.i, label %bb2.i.i, label %bb6.i.i

bb2.i.i:                                          ; preds = %bb1.i.i
  %v17.i.i = shl nuw i32 %v6.i.i, 31
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit

bb6.i.i:                                          ; preds = %bb1.i.i
  %v13.masked.numleadingzeros.i.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i68.i, i1 true)
  %v13.masked.leadingonepos.i.i = xor i32 %v13.masked.numleadingzeros.i.i, 31
  %bb5.tripcount.i.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i
  %v23.i71.i = shl nuw nsw i32 %v13.i68.i, %bb5.tripcount.i.i
  %v27.i.i = shl nuw i32 %v6.i.i, 31
  %reass.sub.i = or disjoint i32 %v27.i.i, 1124073472
  %36 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i, 23
  %v31.i.i = sub nuw nsw i32 %reass.sub.i, %36
  %v25.i.i = shl i32 %v23.i71.i, 13
  %v33.i.i = and i32 %v25.i.i, 8380416
  %v34.i.i = or disjoint i32 %v31.i.i, %v33.i.i
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit

bb9.i.i:                                          ; preds = %bb10
  %v38.i.i = shl nuw i32 %v6.i.i, 31
  %v41.i69.i = shl nuw nsw i32 %v13.i68.i, 13
  %v39.i.i = or disjoint i32 %v41.i69.i, %v38.i.i
  %v42.i70.i = or disjoint i32 %v39.i.i, 2139095040
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit

bb10.i.i:                                         ; preds = %bb10
  %v44.i.i = shl nuw i32 %v6.i.i, 31
  %37 = add nuw nsw i16 %v10.i67.i, 112
  %v46.i73.i = zext nneg i16 %37 to i32
  %v48.i.i = shl nuw nsw i32 %v46.i73.i, 23
  %v49.i.i = or disjoint i32 %v48.i.i, %v44.i.i
  %v51.i74.i = shl nuw nsw i32 %v13.i68.i, 13
  %v52.i.i = or disjoint i32 %v49.i.i, %v51.i74.i
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit

cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit: ; preds = %bb2.i.i, %bb6.i.i, %bb9.i.i, %bb10.i.i
  %v54.i.i = phi i32 [ %v34.i.i, %bb6.i.i ], [ %v17.i.i, %bb2.i.i ], [ %v42.i70.i, %bb9.i.i ], [ %v52.i.i, %bb10.i.i ]
  %v45.i63.i = lshr i8 %v42.i62.i, %v19.i53.i
  %v46.i64.i = shl i8 %v45.i63.i, 4
  %38 = and i8 %v46.i64.i, 48
  %v50.i65.i = add nsw i8 %38, -32
  %v35.i59.i = lshr i8 %v29.i56.i, 4
  %v32.i58.i = and i8 %v29.i56.i, 15
  %v36.i60.i = select i1 %v30.i57.i, i8 %v35.i59.i, i8 %v32.i58.i
  %v51.i66.i = or disjoint i8 %v50.i65.i, %v36.i60.i
  %v39.sroa.4.0.insert.ext.i = zext i8 %v51.i66.i to i32
  %v39.sroa.4.0.insert.shift.i = shl nuw i32 %v39.sroa.4.0.insert.ext.i, 24
  %v45.i42.i = lshr i8 %v42.i41.i, %v19.i53.i
  %v46.i43.i = shl i8 %v45.i42.i, 4
  %39 = and i8 %v46.i43.i, 48
  %v50.i44.i = add nsw i8 %39, -32
  %v35.i38.i = lshr i8 %v29.i35.i, 4
  %v32.i37.i = and i8 %v29.i35.i, 15
  %v36.i39.i = select i1 %v30.i57.i, i8 %v35.i38.i, i8 %v32.i37.i
  %v51.i45.i = or disjoint i8 %v50.i44.i, %v36.i39.i
  %v39.sroa.3.0.insert.ext.i = zext i8 %v51.i45.i to i32
  %v39.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v39.sroa.3.0.insert.ext.i, 16
  %v39.sroa.3.0.insert.insert.i = or disjoint i32 %v39.sroa.4.0.insert.shift.i, %v39.sroa.3.0.insert.shift.i
  %v45.i21.i = lshr i8 %v42.i20.i, %v19.i53.i
  %v46.i22.i = shl i8 %v45.i21.i, 4
  %40 = and i8 %v46.i22.i, 48
  %v50.i23.i = add nsw i8 %40, -32
  %v35.i17.i = lshr i8 %v29.i14.i, 4
  %v32.i16.i = and i8 %v29.i14.i, 15
  %v36.i18.i = select i1 %v30.i57.i, i8 %v35.i17.i, i8 %v32.i16.i
  %v51.i24.i = or disjoint i8 %v50.i23.i, %v36.i18.i
  %v39.sroa.2.0.insert.ext.i = zext i8 %v51.i24.i to i32
  %v39.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v39.sroa.2.0.insert.ext.i, 8
  %v39.sroa.2.0.insert.insert.i = or disjoint i32 %v39.sroa.3.0.insert.insert.i, %v39.sroa.2.0.insert.shift.i
  %v45.i.i = lshr i8 %v42.i.i, %v19.i53.i
  %v46.i.i = shl i8 %v45.i.i, 4
  %41 = and i8 %v46.i.i, 48
  %v50.i.i = add nsw i8 %41, -32
  %v35.i.i = lshr i8 %v29.i.i, 4
  %v32.i.i = and i8 %v29.i.i, 15
  %v36.i.i = select i1 %v30.i57.i, i8 %v35.i.i, i8 %v32.i.i
  %v51.i.i = or disjoint i8 %v50.i.i, %v36.i.i
  %v39.sroa.0.0.insert.ext.i = zext i8 %v51.i.i to i32
  %v39.sroa.0.0.insert.insert.i = or disjoint i32 %v39.sroa.2.0.insert.insert.i, %v39.sroa.0.0.insert.ext.i
  %v55.i.i = bitcast i32 %v54.i.i to float
  %v72.i = load i8, ptr %v71.i, align 1
  %v74.i = sitofp i8 %v72.i to float
  %v75.i = fmul contract float %v55.i.i, %v74.i
  %v76.i = fmul contract float %v103, %v75.i
  %v77.i = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v39.sroa.0.0.insert.insert.i, i32 %35, i32 0) #19
  %v78.i = sitofp i32 %v77.i to float
  %v79.i = fmul contract float %v76.i, %v78.i
  %v112 = fadd contract float %v55472, %v79.i
  br label %bb12

bb12:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit, %bb9
  %v113 = phi float [ %v55472, %bb9 ], [ %v112, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit ]
  br i1 %v115.not, label %bb13, label %bb15

bb13:                                             ; preds = %bb12
  %v11.i.i12 = and i64 %v63, 28
  %v23.i.i15 = and i64 %v63, 60
  %v28.i.i16 = getelementptr i8, ptr %15, i64 %v23.i.i15
  %v29.i.i17 = load i8, ptr %v28.i.i16, align 1
  %v41.i.i18 = getelementptr i8, ptr %17, i64 %v11.i.i12
  %v42.i.i19 = load i8, ptr %v41.i.i18, align 1
  %v19.i20 = or disjoint i64 %v63, 1
  %v11.i5.i22 = and i64 %v19.i20, 29
  %v23.i12.i25 = and i64 %v19.i20, 61
  %v28.i13.i26 = getelementptr i8, ptr %15, i64 %v23.i12.i25
  %v29.i14.i27 = load i8, ptr %v28.i13.i26, align 1
  %v41.i19.i28 = getelementptr i8, ptr %17, i64 %v11.i5.i22
  %v42.i20.i29 = load i8, ptr %v41.i19.i28, align 1
  %v24.i30 = or disjoint i64 %v63, 2
  %v11.i26.i32 = and i64 %v24.i30, 30
  %v23.i33.i35 = and i64 %v24.i30, 62
  %v28.i34.i36 = getelementptr i8, ptr %15, i64 %v23.i33.i35
  %v29.i35.i37 = load i8, ptr %v28.i34.i36, align 1
  %v41.i40.i38 = getelementptr i8, ptr %17, i64 %v11.i26.i32
  %v42.i41.i39 = load i8, ptr %v41.i40.i38, align 1
  %v29.i40 = or disjoint i64 %v63, 3
  %v11.i47.i42 = and i64 %v29.i40, 31
  %v23.i54.i45 = and i64 %v29.i40, 63
  %v28.i55.i46 = getelementptr i8, ptr %15, i64 %v23.i54.i45
  %v29.i56.i47 = load i8, ptr %v28.i55.i46, align 1
  %v41.i61.i48 = getelementptr i8, ptr %17, i64 %v11.i47.i42
  %v42.i62.i49 = load i8, ptr %v41.i61.i48, align 1
  %v46.i51 = load i8, ptr %v45.i50, align 1
  %v51.i53 = load i8, ptr %v50.i52, align 1
  %v55.sroa.2.0.insert.ext.i54 = zext i8 %v51.i53 to i16
  %v55.sroa.2.0.insert.shift.i55 = shl nuw i16 %v55.sroa.2.0.insert.ext.i54, 8
  %v55.sroa.0.0.insert.ext.i56 = zext i8 %v46.i51 to i16
  %v4.i.i57 = lshr i16 %v55.sroa.2.0.insert.ext.i54, 7
  %v6.i.i58 = zext nneg i16 %v4.i.i57 to i32
  %v9.i.i59 = lshr i16 %v55.sroa.2.0.insert.ext.i54, 2
  %v10.i67.i60 = and i16 %v9.i.i59, 31
  %v55.sroa.2.0.insert.shift.masked.i61 = and i16 %v55.sroa.2.0.insert.shift.i55, 768
  %v12.i.i62 = or disjoint i16 %v55.sroa.2.0.insert.shift.masked.i61, %v55.sroa.0.0.insert.ext.i56
  %v13.i68.i63 = zext nneg i16 %v12.i.i62 to i32
  switch i16 %v10.i67.i60, label %bb10.i.i155 [
    i16 0, label %bb1.i.i140
    i16 31, label %bb9.i.i64
  ]

bb1.i.i140:                                       ; preds = %bb13
  %v15.i.i141 = icmp eq i16 %v12.i.i62, 0
  br i1 %v15.i.i141, label %bb2.i.i153, label %bb6.i.i142

bb2.i.i153:                                       ; preds = %bb1.i.i140
  %v17.i.i154 = shl nuw i32 %v6.i.i58, 31
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162

bb6.i.i142:                                       ; preds = %bb1.i.i140
  %v13.masked.numleadingzeros.i.i143 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i68.i63, i1 true)
  %v13.masked.leadingonepos.i.i144 = xor i32 %v13.masked.numleadingzeros.i.i143, 31
  %bb5.tripcount.i.i145 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i144
  %v23.i71.i146 = shl nuw nsw i32 %v13.i68.i63, %bb5.tripcount.i.i145
  %v27.i.i147 = shl nuw i32 %v6.i.i58, 31
  %reass.sub.i148 = or disjoint i32 %v27.i.i147, 1124073472
  %42 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i143, 23
  %v31.i.i149 = sub nuw nsw i32 %reass.sub.i148, %42
  %v25.i.i150 = shl i32 %v23.i71.i146, 13
  %v33.i.i151 = and i32 %v25.i.i150, 8380416
  %v34.i.i152 = or disjoint i32 %v31.i.i149, %v33.i.i151
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162

bb9.i.i64:                                        ; preds = %bb13
  %v38.i.i65 = shl nuw i32 %v6.i.i58, 31
  %v41.i69.i66 = shl nuw nsw i32 %v13.i68.i63, 13
  %v39.i.i67 = or disjoint i32 %v41.i69.i66, %v38.i.i65
  %v42.i70.i68 = or disjoint i32 %v39.i.i67, 2139095040
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162

bb10.i.i155:                                      ; preds = %bb13
  %v44.i.i156 = shl nuw i32 %v6.i.i58, 31
  %43 = add nuw nsw i16 %v10.i67.i60, 112
  %v46.i73.i157 = zext nneg i16 %43 to i32
  %v48.i.i158 = shl nuw nsw i32 %v46.i73.i157, 23
  %v49.i.i159 = or disjoint i32 %v48.i.i158, %v44.i.i156
  %v51.i74.i160 = shl nuw nsw i32 %v13.i68.i63, 13
  %v52.i.i161 = or disjoint i32 %v49.i.i159, %v51.i74.i160
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162

cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162: ; preds = %bb2.i.i153, %bb6.i.i142, %bb9.i.i64, %bb10.i.i155
  %v54.i.i69 = phi i32 [ %v34.i.i152, %bb6.i.i142 ], [ %v17.i.i154, %bb2.i.i153 ], [ %v42.i70.i68, %bb9.i.i64 ], [ %v52.i.i161, %bb10.i.i155 ]
  %v45.i63.i74 = lshr i8 %v42.i62.i49, %v19.i53.i
  %v46.i64.i75 = shl i8 %v45.i63.i74, 4
  %44 = and i8 %v46.i64.i75, 48
  %v50.i65.i76 = add nsw i8 %44, -32
  %v35.i59.i78 = lshr i8 %v29.i56.i47, 4
  %v32.i58.i79 = and i8 %v29.i56.i47, 15
  %v36.i60.i80 = select i1 %v30.i57.i, i8 %v35.i59.i78, i8 %v32.i58.i79
  %v51.i66.i81 = or disjoint i8 %v50.i65.i76, %v36.i60.i80
  %v39.sroa.4.0.insert.ext.i82 = zext i8 %v51.i66.i81 to i32
  %v39.sroa.4.0.insert.shift.i83 = shl nuw i32 %v39.sroa.4.0.insert.ext.i82, 24
  %v45.i42.i88 = lshr i8 %v42.i41.i39, %v19.i53.i
  %v46.i43.i89 = shl i8 %v45.i42.i88, 4
  %45 = and i8 %v46.i43.i89, 48
  %v50.i44.i90 = add nsw i8 %45, -32
  %v35.i38.i92 = lshr i8 %v29.i35.i37, 4
  %v32.i37.i93 = and i8 %v29.i35.i37, 15
  %v36.i39.i94 = select i1 %v30.i57.i, i8 %v35.i38.i92, i8 %v32.i37.i93
  %v51.i45.i95 = or disjoint i8 %v50.i44.i90, %v36.i39.i94
  %v39.sroa.3.0.insert.ext.i96 = zext i8 %v51.i45.i95 to i32
  %v39.sroa.3.0.insert.shift.i97 = shl nuw nsw i32 %v39.sroa.3.0.insert.ext.i96, 16
  %v39.sroa.3.0.insert.insert.i98 = or disjoint i32 %v39.sroa.4.0.insert.shift.i83, %v39.sroa.3.0.insert.shift.i97
  %v45.i21.i103 = lshr i8 %v42.i20.i29, %v19.i53.i
  %v46.i22.i104 = shl i8 %v45.i21.i103, 4
  %46 = and i8 %v46.i22.i104, 48
  %v50.i23.i105 = add nsw i8 %46, -32
  %v35.i17.i107 = lshr i8 %v29.i14.i27, 4
  %v32.i16.i108 = and i8 %v29.i14.i27, 15
  %v36.i18.i109 = select i1 %v30.i57.i, i8 %v35.i17.i107, i8 %v32.i16.i108
  %v51.i24.i110 = or disjoint i8 %v50.i23.i105, %v36.i18.i109
  %v39.sroa.2.0.insert.ext.i111 = zext i8 %v51.i24.i110 to i32
  %v39.sroa.2.0.insert.shift.i112 = shl nuw nsw i32 %v39.sroa.2.0.insert.ext.i111, 8
  %v39.sroa.2.0.insert.insert.i113 = or disjoint i32 %v39.sroa.3.0.insert.insert.i98, %v39.sroa.2.0.insert.shift.i112
  %v45.i.i118 = lshr i8 %v42.i.i19, %v19.i53.i
  %v46.i.i119 = shl i8 %v45.i.i118, 4
  %47 = and i8 %v46.i.i119, 48
  %v50.i.i120 = add nsw i8 %47, -32
  %v35.i.i122 = lshr i8 %v29.i.i17, 4
  %v32.i.i123 = and i8 %v29.i.i17, 15
  %v36.i.i124 = select i1 %v30.i57.i, i8 %v35.i.i122, i8 %v32.i.i123
  %v51.i.i125 = or disjoint i8 %v50.i.i120, %v36.i.i124
  %v39.sroa.0.0.insert.ext.i126 = zext i8 %v51.i.i125 to i32
  %v39.sroa.0.0.insert.insert.i127 = or disjoint i32 %v39.sroa.2.0.insert.insert.i113, %v39.sroa.0.0.insert.ext.i126
  %v55.i.i128 = bitcast i32 %v54.i.i69 to float
  %v72.i133 = load i8, ptr %v71.i132, align 1
  %v74.i134 = sitofp i8 %v72.i133 to float
  %v75.i135 = fmul contract float %v55.i.i128, %v74.i134
  %v76.i136 = fmul contract float %v103, %v75.i135
  %v77.i137 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v39.sroa.0.0.insert.insert.i127, i32 %35, i32 0) #19
  %v78.i138 = sitofp i32 %v77.i137 to float
  %v79.i139 = fmul contract float %v76.i136, %v78.i138
  %v123 = fadd contract float %v56473, %v79.i139
  br label %bb15

bb15:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162, %bb12
  %v124 = phi float [ %v56473, %bb12 ], [ %v123, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit162 ]
  br i1 %v126.not, label %bb16, label %bb18

bb16:                                             ; preds = %bb15
  %v11.i.i164 = and i64 %v63, 28
  %v23.i.i167 = and i64 %v63, 60
  %v28.i.i168 = getelementptr i8, ptr %22, i64 %v23.i.i167
  %v29.i.i169 = load i8, ptr %v28.i.i168, align 1
  %v41.i.i170 = getelementptr i8, ptr %24, i64 %v11.i.i164
  %v42.i.i171 = load i8, ptr %v41.i.i170, align 1
  %v19.i172 = or disjoint i64 %v63, 1
  %v11.i5.i174 = and i64 %v19.i172, 29
  %v23.i12.i177 = and i64 %v19.i172, 61
  %v28.i13.i178 = getelementptr i8, ptr %22, i64 %v23.i12.i177
  %v29.i14.i179 = load i8, ptr %v28.i13.i178, align 1
  %v41.i19.i180 = getelementptr i8, ptr %24, i64 %v11.i5.i174
  %v42.i20.i181 = load i8, ptr %v41.i19.i180, align 1
  %v24.i182 = or disjoint i64 %v63, 2
  %v11.i26.i184 = and i64 %v24.i182, 30
  %v23.i33.i187 = and i64 %v24.i182, 62
  %v28.i34.i188 = getelementptr i8, ptr %22, i64 %v23.i33.i187
  %v29.i35.i189 = load i8, ptr %v28.i34.i188, align 1
  %v41.i40.i190 = getelementptr i8, ptr %24, i64 %v11.i26.i184
  %v42.i41.i191 = load i8, ptr %v41.i40.i190, align 1
  %v29.i192 = or disjoint i64 %v63, 3
  %v11.i47.i194 = and i64 %v29.i192, 31
  %v23.i54.i197 = and i64 %v29.i192, 63
  %v28.i55.i198 = getelementptr i8, ptr %22, i64 %v23.i54.i197
  %v29.i56.i199 = load i8, ptr %v28.i55.i198, align 1
  %v41.i61.i200 = getelementptr i8, ptr %24, i64 %v11.i47.i194
  %v42.i62.i201 = load i8, ptr %v41.i61.i200, align 1
  %v46.i203 = load i8, ptr %v45.i202, align 1
  %v51.i205 = load i8, ptr %v50.i204, align 1
  %v55.sroa.2.0.insert.ext.i206 = zext i8 %v51.i205 to i16
  %v55.sroa.2.0.insert.shift.i207 = shl nuw i16 %v55.sroa.2.0.insert.ext.i206, 8
  %v55.sroa.0.0.insert.ext.i208 = zext i8 %v46.i203 to i16
  %v4.i.i209 = lshr i16 %v55.sroa.2.0.insert.ext.i206, 7
  %v6.i.i210 = zext nneg i16 %v4.i.i209 to i32
  %v9.i.i211 = lshr i16 %v55.sroa.2.0.insert.ext.i206, 2
  %v10.i67.i212 = and i16 %v9.i.i211, 31
  %v55.sroa.2.0.insert.shift.masked.i213 = and i16 %v55.sroa.2.0.insert.shift.i207, 768
  %v12.i.i214 = or disjoint i16 %v55.sroa.2.0.insert.shift.masked.i213, %v55.sroa.0.0.insert.ext.i208
  %v13.i68.i215 = zext nneg i16 %v12.i.i214 to i32
  switch i16 %v10.i67.i212, label %bb10.i.i307 [
    i16 0, label %bb1.i.i292
    i16 31, label %bb9.i.i216
  ]

bb1.i.i292:                                       ; preds = %bb16
  %v15.i.i293 = icmp eq i16 %v12.i.i214, 0
  br i1 %v15.i.i293, label %bb2.i.i305, label %bb6.i.i294

bb2.i.i305:                                       ; preds = %bb1.i.i292
  %v17.i.i306 = shl nuw i32 %v6.i.i210, 31
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314

bb6.i.i294:                                       ; preds = %bb1.i.i292
  %v13.masked.numleadingzeros.i.i295 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i68.i215, i1 true)
  %v13.masked.leadingonepos.i.i296 = xor i32 %v13.masked.numleadingzeros.i.i295, 31
  %bb5.tripcount.i.i297 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i296
  %v23.i71.i298 = shl nuw nsw i32 %v13.i68.i215, %bb5.tripcount.i.i297
  %v27.i.i299 = shl nuw i32 %v6.i.i210, 31
  %reass.sub.i300 = or disjoint i32 %v27.i.i299, 1124073472
  %48 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i295, 23
  %v31.i.i301 = sub nuw nsw i32 %reass.sub.i300, %48
  %v25.i.i302 = shl i32 %v23.i71.i298, 13
  %v33.i.i303 = and i32 %v25.i.i302, 8380416
  %v34.i.i304 = or disjoint i32 %v31.i.i301, %v33.i.i303
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314

bb9.i.i216:                                       ; preds = %bb16
  %v38.i.i217 = shl nuw i32 %v6.i.i210, 31
  %v41.i69.i218 = shl nuw nsw i32 %v13.i68.i215, 13
  %v39.i.i219 = or disjoint i32 %v41.i69.i218, %v38.i.i217
  %v42.i70.i220 = or disjoint i32 %v39.i.i219, 2139095040
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314

bb10.i.i307:                                      ; preds = %bb16
  %v44.i.i308 = shl nuw i32 %v6.i.i210, 31
  %49 = add nuw nsw i16 %v10.i67.i212, 112
  %v46.i73.i309 = zext nneg i16 %49 to i32
  %v48.i.i310 = shl nuw nsw i32 %v46.i73.i309, 23
  %v49.i.i311 = or disjoint i32 %v48.i.i310, %v44.i.i308
  %v51.i74.i312 = shl nuw nsw i32 %v13.i68.i215, 13
  %v52.i.i313 = or disjoint i32 %v49.i.i311, %v51.i74.i312
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314

cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314: ; preds = %bb2.i.i305, %bb6.i.i294, %bb9.i.i216, %bb10.i.i307
  %v54.i.i221 = phi i32 [ %v34.i.i304, %bb6.i.i294 ], [ %v17.i.i306, %bb2.i.i305 ], [ %v42.i70.i220, %bb9.i.i216 ], [ %v52.i.i313, %bb10.i.i307 ]
  %v45.i63.i226 = lshr i8 %v42.i62.i201, %v19.i53.i
  %v46.i64.i227 = shl i8 %v45.i63.i226, 4
  %50 = and i8 %v46.i64.i227, 48
  %v50.i65.i228 = add nsw i8 %50, -32
  %v35.i59.i230 = lshr i8 %v29.i56.i199, 4
  %v32.i58.i231 = and i8 %v29.i56.i199, 15
  %v36.i60.i232 = select i1 %v30.i57.i, i8 %v35.i59.i230, i8 %v32.i58.i231
  %v51.i66.i233 = or disjoint i8 %v50.i65.i228, %v36.i60.i232
  %v39.sroa.4.0.insert.ext.i234 = zext i8 %v51.i66.i233 to i32
  %v39.sroa.4.0.insert.shift.i235 = shl nuw i32 %v39.sroa.4.0.insert.ext.i234, 24
  %v45.i42.i240 = lshr i8 %v42.i41.i191, %v19.i53.i
  %v46.i43.i241 = shl i8 %v45.i42.i240, 4
  %51 = and i8 %v46.i43.i241, 48
  %v50.i44.i242 = add nsw i8 %51, -32
  %v35.i38.i244 = lshr i8 %v29.i35.i189, 4
  %v32.i37.i245 = and i8 %v29.i35.i189, 15
  %v36.i39.i246 = select i1 %v30.i57.i, i8 %v35.i38.i244, i8 %v32.i37.i245
  %v51.i45.i247 = or disjoint i8 %v50.i44.i242, %v36.i39.i246
  %v39.sroa.3.0.insert.ext.i248 = zext i8 %v51.i45.i247 to i32
  %v39.sroa.3.0.insert.shift.i249 = shl nuw nsw i32 %v39.sroa.3.0.insert.ext.i248, 16
  %v39.sroa.3.0.insert.insert.i250 = or disjoint i32 %v39.sroa.4.0.insert.shift.i235, %v39.sroa.3.0.insert.shift.i249
  %v45.i21.i255 = lshr i8 %v42.i20.i181, %v19.i53.i
  %v46.i22.i256 = shl i8 %v45.i21.i255, 4
  %52 = and i8 %v46.i22.i256, 48
  %v50.i23.i257 = add nsw i8 %52, -32
  %v35.i17.i259 = lshr i8 %v29.i14.i179, 4
  %v32.i16.i260 = and i8 %v29.i14.i179, 15
  %v36.i18.i261 = select i1 %v30.i57.i, i8 %v35.i17.i259, i8 %v32.i16.i260
  %v51.i24.i262 = or disjoint i8 %v50.i23.i257, %v36.i18.i261
  %v39.sroa.2.0.insert.ext.i263 = zext i8 %v51.i24.i262 to i32
  %v39.sroa.2.0.insert.shift.i264 = shl nuw nsw i32 %v39.sroa.2.0.insert.ext.i263, 8
  %v39.sroa.2.0.insert.insert.i265 = or disjoint i32 %v39.sroa.3.0.insert.insert.i250, %v39.sroa.2.0.insert.shift.i264
  %v45.i.i270 = lshr i8 %v42.i.i171, %v19.i53.i
  %v46.i.i271 = shl i8 %v45.i.i270, 4
  %53 = and i8 %v46.i.i271, 48
  %v50.i.i272 = add nsw i8 %53, -32
  %v35.i.i274 = lshr i8 %v29.i.i169, 4
  %v32.i.i275 = and i8 %v29.i.i169, 15
  %v36.i.i276 = select i1 %v30.i57.i, i8 %v35.i.i274, i8 %v32.i.i275
  %v51.i.i277 = or disjoint i8 %v50.i.i272, %v36.i.i276
  %v39.sroa.0.0.insert.ext.i278 = zext i8 %v51.i.i277 to i32
  %v39.sroa.0.0.insert.insert.i279 = or disjoint i32 %v39.sroa.2.0.insert.insert.i265, %v39.sroa.0.0.insert.ext.i278
  %v55.i.i280 = bitcast i32 %v54.i.i221 to float
  %v72.i285 = load i8, ptr %v71.i284, align 1
  %v74.i286 = sitofp i8 %v72.i285 to float
  %v75.i287 = fmul contract float %v55.i.i280, %v74.i286
  %v76.i288 = fmul contract float %v103, %v75.i287
  %v77.i289 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v39.sroa.0.0.insert.insert.i279, i32 %35, i32 0) #19
  %v78.i290 = sitofp i32 %v77.i289 to float
  %v79.i291 = fmul contract float %v76.i288, %v78.i290
  %v134 = fadd contract float %v57474, %v79.i291
  br label %bb18

bb18:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314, %bb15
  %v135 = phi float [ %v57474, %bb15 ], [ %v134, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit314 ]
  br i1 %v137.not, label %bb19, label %bb21

bb19:                                             ; preds = %bb18
  %v11.i.i316 = and i64 %v63, 28
  %v23.i.i319 = and i64 %v63, 60
  %v28.i.i320 = getelementptr i8, ptr %29, i64 %v23.i.i319
  %v29.i.i321 = load i8, ptr %v28.i.i320, align 1
  %v41.i.i322 = getelementptr i8, ptr %31, i64 %v11.i.i316
  %v42.i.i323 = load i8, ptr %v41.i.i322, align 1
  %v19.i324 = or disjoint i64 %v63, 1
  %v11.i5.i326 = and i64 %v19.i324, 29
  %v23.i12.i329 = and i64 %v19.i324, 61
  %v28.i13.i330 = getelementptr i8, ptr %29, i64 %v23.i12.i329
  %v29.i14.i331 = load i8, ptr %v28.i13.i330, align 1
  %v41.i19.i332 = getelementptr i8, ptr %31, i64 %v11.i5.i326
  %v42.i20.i333 = load i8, ptr %v41.i19.i332, align 1
  %v24.i334 = or disjoint i64 %v63, 2
  %v11.i26.i336 = and i64 %v24.i334, 30
  %v23.i33.i339 = and i64 %v24.i334, 62
  %v28.i34.i340 = getelementptr i8, ptr %29, i64 %v23.i33.i339
  %v29.i35.i341 = load i8, ptr %v28.i34.i340, align 1
  %v41.i40.i342 = getelementptr i8, ptr %31, i64 %v11.i26.i336
  %v42.i41.i343 = load i8, ptr %v41.i40.i342, align 1
  %v29.i344 = or disjoint i64 %v63, 3
  %v11.i47.i346 = and i64 %v29.i344, 31
  %v23.i54.i349 = and i64 %v29.i344, 63
  %v28.i55.i350 = getelementptr i8, ptr %29, i64 %v23.i54.i349
  %v29.i56.i351 = load i8, ptr %v28.i55.i350, align 1
  %v41.i61.i352 = getelementptr i8, ptr %31, i64 %v11.i47.i346
  %v42.i62.i353 = load i8, ptr %v41.i61.i352, align 1
  %v46.i355 = load i8, ptr %v45.i354, align 1
  %v51.i357 = load i8, ptr %v50.i356, align 1
  %v55.sroa.2.0.insert.ext.i358 = zext i8 %v51.i357 to i16
  %v55.sroa.2.0.insert.shift.i359 = shl nuw i16 %v55.sroa.2.0.insert.ext.i358, 8
  %v55.sroa.0.0.insert.ext.i360 = zext i8 %v46.i355 to i16
  %v4.i.i361 = lshr i16 %v55.sroa.2.0.insert.ext.i358, 7
  %v6.i.i362 = zext nneg i16 %v4.i.i361 to i32
  %v9.i.i363 = lshr i16 %v55.sroa.2.0.insert.ext.i358, 2
  %v10.i67.i364 = and i16 %v9.i.i363, 31
  %v55.sroa.2.0.insert.shift.masked.i365 = and i16 %v55.sroa.2.0.insert.shift.i359, 768
  %v12.i.i366 = or disjoint i16 %v55.sroa.2.0.insert.shift.masked.i365, %v55.sroa.0.0.insert.ext.i360
  %v13.i68.i367 = zext nneg i16 %v12.i.i366 to i32
  switch i16 %v10.i67.i364, label %bb10.i.i459 [
    i16 0, label %bb1.i.i444
    i16 31, label %bb9.i.i368
  ]

bb1.i.i444:                                       ; preds = %bb19
  %v15.i.i445 = icmp eq i16 %v12.i.i366, 0
  br i1 %v15.i.i445, label %bb2.i.i457, label %bb6.i.i446

bb2.i.i457:                                       ; preds = %bb1.i.i444
  %v17.i.i458 = shl nuw i32 %v6.i.i362, 31
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466

bb6.i.i446:                                       ; preds = %bb1.i.i444
  %v13.masked.numleadingzeros.i.i447 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i68.i367, i1 true)
  %v13.masked.leadingonepos.i.i448 = xor i32 %v13.masked.numleadingzeros.i.i447, 31
  %bb5.tripcount.i.i449 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i.i448
  %v23.i71.i450 = shl nuw nsw i32 %v13.i68.i367, %bb5.tripcount.i.i449
  %v27.i.i451 = shl nuw i32 %v6.i.i362, 31
  %reass.sub.i452 = or disjoint i32 %v27.i.i451, 1124073472
  %54 = shl nuw nsw i32 %v13.masked.numleadingzeros.i.i447, 23
  %v31.i.i453 = sub nuw nsw i32 %reass.sub.i452, %54
  %v25.i.i454 = shl i32 %v23.i71.i450, 13
  %v33.i.i455 = and i32 %v25.i.i454, 8380416
  %v34.i.i456 = or disjoint i32 %v31.i.i453, %v33.i.i455
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466

bb9.i.i368:                                       ; preds = %bb19
  %v38.i.i369 = shl nuw i32 %v6.i.i362, 31
  %v41.i69.i370 = shl nuw nsw i32 %v13.i68.i367, 13
  %v39.i.i371 = or disjoint i32 %v41.i69.i370, %v38.i.i369
  %v42.i70.i372 = or disjoint i32 %v39.i.i371, 2139095040
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466

bb10.i.i459:                                      ; preds = %bb19
  %v44.i.i460 = shl nuw i32 %v6.i.i362, 31
  %55 = add nuw nsw i16 %v10.i67.i364, 112
  %v46.i73.i461 = zext nneg i16 %55 to i32
  %v48.i.i462 = shl nuw nsw i32 %v46.i73.i461, 23
  %v49.i.i463 = or disjoint i32 %v48.i.i462, %v44.i.i460
  %v51.i74.i464 = shl nuw nsw i32 %v13.i68.i367, 13
  %v52.i.i465 = or disjoint i32 %v49.i.i463, %v51.i74.i464
  br label %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466

cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466: ; preds = %bb2.i.i457, %bb6.i.i446, %bb9.i.i368, %bb10.i.i459
  %v54.i.i373 = phi i32 [ %v34.i.i456, %bb6.i.i446 ], [ %v17.i.i458, %bb2.i.i457 ], [ %v42.i70.i372, %bb9.i.i368 ], [ %v52.i.i465, %bb10.i.i459 ]
  %v45.i63.i378 = lshr i8 %v42.i62.i353, %v19.i53.i
  %v46.i64.i379 = shl i8 %v45.i63.i378, 4
  %56 = and i8 %v46.i64.i379, 48
  %v50.i65.i380 = add nsw i8 %56, -32
  %v35.i59.i382 = lshr i8 %v29.i56.i351, 4
  %v32.i58.i383 = and i8 %v29.i56.i351, 15
  %v36.i60.i384 = select i1 %v30.i57.i, i8 %v35.i59.i382, i8 %v32.i58.i383
  %v51.i66.i385 = or disjoint i8 %v50.i65.i380, %v36.i60.i384
  %v39.sroa.4.0.insert.ext.i386 = zext i8 %v51.i66.i385 to i32
  %v39.sroa.4.0.insert.shift.i387 = shl nuw i32 %v39.sroa.4.0.insert.ext.i386, 24
  %v45.i42.i392 = lshr i8 %v42.i41.i343, %v19.i53.i
  %v46.i43.i393 = shl i8 %v45.i42.i392, 4
  %57 = and i8 %v46.i43.i393, 48
  %v50.i44.i394 = add nsw i8 %57, -32
  %v35.i38.i396 = lshr i8 %v29.i35.i341, 4
  %v32.i37.i397 = and i8 %v29.i35.i341, 15
  %v36.i39.i398 = select i1 %v30.i57.i, i8 %v35.i38.i396, i8 %v32.i37.i397
  %v51.i45.i399 = or disjoint i8 %v50.i44.i394, %v36.i39.i398
  %v39.sroa.3.0.insert.ext.i400 = zext i8 %v51.i45.i399 to i32
  %v39.sroa.3.0.insert.shift.i401 = shl nuw nsw i32 %v39.sroa.3.0.insert.ext.i400, 16
  %v39.sroa.3.0.insert.insert.i402 = or disjoint i32 %v39.sroa.4.0.insert.shift.i387, %v39.sroa.3.0.insert.shift.i401
  %v45.i21.i407 = lshr i8 %v42.i20.i333, %v19.i53.i
  %v46.i22.i408 = shl i8 %v45.i21.i407, 4
  %58 = and i8 %v46.i22.i408, 48
  %v50.i23.i409 = add nsw i8 %58, -32
  %v35.i17.i411 = lshr i8 %v29.i14.i331, 4
  %v32.i16.i412 = and i8 %v29.i14.i331, 15
  %v36.i18.i413 = select i1 %v30.i57.i, i8 %v35.i17.i411, i8 %v32.i16.i412
  %v51.i24.i414 = or disjoint i8 %v50.i23.i409, %v36.i18.i413
  %v39.sroa.2.0.insert.ext.i415 = zext i8 %v51.i24.i414 to i32
  %v39.sroa.2.0.insert.shift.i416 = shl nuw nsw i32 %v39.sroa.2.0.insert.ext.i415, 8
  %v39.sroa.2.0.insert.insert.i417 = or disjoint i32 %v39.sroa.3.0.insert.insert.i402, %v39.sroa.2.0.insert.shift.i416
  %v45.i.i422 = lshr i8 %v42.i.i323, %v19.i53.i
  %v46.i.i423 = shl i8 %v45.i.i422, 4
  %59 = and i8 %v46.i.i423, 48
  %v50.i.i424 = add nsw i8 %59, -32
  %v35.i.i426 = lshr i8 %v29.i.i321, 4
  %v32.i.i427 = and i8 %v29.i.i321, 15
  %v36.i.i428 = select i1 %v30.i57.i, i8 %v35.i.i426, i8 %v32.i.i427
  %v51.i.i429 = or disjoint i8 %v50.i.i424, %v36.i.i428
  %v39.sroa.0.0.insert.ext.i430 = zext i8 %v51.i.i429 to i32
  %v39.sroa.0.0.insert.insert.i431 = or disjoint i32 %v39.sroa.2.0.insert.insert.i417, %v39.sroa.0.0.insert.ext.i430
  %v55.i.i432 = bitcast i32 %v54.i.i373 to float
  %v72.i437 = load i8, ptr %v71.i436, align 1
  %v74.i438 = sitofp i8 %v72.i437 to float
  %v75.i439 = fmul contract float %v55.i.i432, %v74.i438
  %v76.i440 = fmul contract float %v103, %v75.i439
  %v77.i441 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %v39.sroa.0.0.insert.insert.i431, i32 %35, i32 0) #19
  %v78.i442 = sitofp i32 %v77.i441 to float
  %v79.i443 = fmul contract float %v76.i440, %v78.i442
  %v145 = fadd contract float %v58475, %v79.i443
  br label %bb21

bb21:                                             ; preds = %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466, %bb18
  %v146 = phi float [ %v58475, %bb18 ], [ %v145, %cuda_kernels__oxide_kernels__kernels__q6k_q8_chunk.exit466 ]
  br i1 %v60.not, label %bb9, label %bb22

bb22:                                             ; preds = %bb21
  %v148 = add nuw nsw i64 %v51482, 1
  %exitcond.not = icmp eq i64 %v148, %v43
  br i1 %exitcond.not, label %bb24.preheader, label %bb7

bb27:                                             ; preds = %bb24.preheader
  %v158 = mul nuw nsw i64 %v40.zext, %v29
  %v159 = add nuw nsw i64 %v42, %v158
  %v160.not = icmp samesign ult i64 %v42, %v29
  br i1 %v160.not, label %bb28, label %bb29

bb28:                                             ; preds = %bb27
  %v163 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  store float %v182.4, ptr %v163, align 4
  br label %bb29

bb29:                                             ; preds = %bb28, %bb27
  %v164 = or disjoint i64 %v42, 1
  %v165.not = icmp samesign ult i64 %v164, %v29
  br i1 %v165.not, label %bb30, label %bb32

bb30:                                             ; preds = %bb29
  %60 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  %v169 = getelementptr inbounds nuw i8, ptr %60, i64 4
  store float %v184.4, ptr %v169, align 4
  br label %bb32

bb32:                                             ; preds = %bb29, %bb30
  %v170 = or disjoint i64 %v42, 2
  %v171.not = icmp samesign ult i64 %v170, %v29
  br i1 %v171.not, label %bb33, label %bb35

bb33:                                             ; preds = %bb32
  %61 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  %v175 = getelementptr inbounds nuw i8, ptr %61, i64 8
  store float %v186.4, ptr %v175, align 4
  br label %bb35

bb35:                                             ; preds = %bb32, %bb33
  %v176 = or disjoint i64 %v42, 3
  %v177.not = icmp samesign ult i64 %v176, %v29
  br i1 %v177.not, label %bb36, label %bb40

bb36:                                             ; preds = %bb35
  %62 = getelementptr inbounds nuw float, ptr %v9, i64 %v159
  %v181 = getelementptr inbounds nuw i8, ptr %62, i64 12
  store float %v188.4, ptr %v181, align 4
  br label %bb40

bb40:                                             ; preds = %bb24.preheader, %bb35, %bb36, %entry
  ret void

bb45:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @q8_0_gemm_element(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(address_is_null) %v7, i64 %v8) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i5 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i6 = icmp eq i32 %v4.i4, 1
  %v7.i7 = icmp eq i32 %v6.i5, 1
  %v8.not.not.i = and i1 %v5.i6, %v7.i7
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i8 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i8
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v25 = zext i32 %v4 to i64
  %v26 = zext i32 %v6 to i64
  %v27 = mul nuw i64 %v26, %v25
  %v28.not = icmp ult i64 %v22.i, %v27
  br i1 %v28.not, label %bb3, label %bb19

bb3:                                              ; preds = %entry
  %v30.not = icmp eq i32 %v4, 0
  br i1 %v30.not, label %bb27, label %bb4

bb4:                                              ; preds = %bb3
  %v34 = zext i32 %v5 to i64
  %v38.not22.not = icmp eq i32 %v5, 0
  br i1 %v38.not22.not, label %bb15, label %bb6.lr.ph

bb6.lr.ph:                                        ; preds = %bb4
  %v25.frozen = freeze i64 %v25
  %v33 = udiv i64 %v22.i, %v25.frozen
  %0 = mul i64 %v33, %v25.frozen
  %v32.decomposed = sub i64 %v22.i, %0
  %v40 = mul nuw i64 %v32.decomposed, %v34
  %v59 = mul i64 %v33, %v34
  br label %bb6

bb6:                                              ; preds = %bb6.lr.ph, %bb14
  %v3724 = phi i64 [ 0, %bb6.lr.ph ], [ %v84, %bb14 ]
  %v3623 = phi float [ 0.000000e+00, %bb6.lr.ph ], [ %v82, %bb14 ]
  %reass.add = add nuw i64 %v3724, %v40
  %reass.mul = mul i64 %reass.add, 34
  %v44 = icmp ult i64 %reass.mul, %v1
  br i1 %v44, label %bb7, label %bb28

bb7:                                              ; preds = %bb6
  %v48 = or disjoint i64 %reass.mul, 1
  %v49 = icmp ult i64 %v48, %v1
  br i1 %v49, label %bb8, label %bb29

bb8:                                              ; preds = %bb7
  %v46 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v47 = load i8, ptr %v46, align 1
  %v51 = getelementptr inbounds i8, ptr %v0, i64 %v48
  %v52 = load i8, ptr %v51, align 1
  %v56 = alloca [2 x i8], align 2
  store i8 %v47, ptr %v56, align 2
  %v56.repack1 = getelementptr inbounds nuw i8, ptr %v56, i64 1
  store i8 %v52, ptr %v56.repack1, align 1
  %v57 = load i16, ptr %v56, align 2
  %v4.i9 = lshr i16 %v57, 15
  %v6.i10 = zext nneg i16 %v4.i9 to i32
  %v9.i = lshr i16 %v57, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v57, 1023
  %v13.i11 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb8
  %v15.i12 = icmp eq i16 %v12.i, 0
  br i1 %v15.i12, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i13 = shl nuw i32 %v6.i10, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i11, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i11, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i10, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb8
  %v38.i = shl nuw i32 %v6.i10, 31
  %v41.i = shl nuw nsw i32 %v13.i11, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb8
  %v44.i = shl nuw i32 %v6.i10, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i11, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i13, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v60 = add i64 %v3724, %v59
  %v61 = shl i64 %v60, 5
  %v66 = add nuw i64 %reass.mul, 2
  br label %bb11

bb11:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb13
  %v6321 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v83, %bb13 ]
  %v6220 = phi float [ %v3623, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v82, %bb13 ]
  %v67 = add nuw i64 %v66, %v6321
  %v68 = icmp ult i64 %v67, %v1
  br i1 %v68, label %bb12, label %bb30

bb12:                                             ; preds = %bb11
  %v75 = add nuw nsw i64 %v6321, %v61
  %v77 = icmp ult i64 %v75, %v3
  br i1 %v77, label %bb13, label %bb31

bb13:                                             ; preds = %bb12
  %v70 = getelementptr inbounds i8, ptr %v0, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v73 = sitofp i8 %v71 to float
  %v74 = fmul contract float %v55.i, %v73
  %v79 = getelementptr inbounds float, ptr %v2, i64 %v75
  %v80 = load float, ptr %v79, align 4
  %v81 = fmul contract float %v80, %v74
  %v82 = fadd contract float %v6220, %v81
  %v83 = add nuw nsw i64 %v6321, 1
  %exitcond = icmp eq i64 %v83, 32
  br i1 %exitcond, label %bb14, label %bb11

bb14:                                             ; preds = %bb13
  %v84 = add nuw nsw i64 %v3724, 1
  %exitcond25.not = icmp eq i64 %v84, %v34
  br i1 %exitcond25.not, label %bb15, label %bb6

bb15:                                             ; preds = %bb14, %bb4
  %v36.lcssa = phi float [ 0.000000e+00, %bb4 ], [ %v82, %bb14 ]
  %v88 = icmp ult i64 %v22.i, %v8
  %or.cond.not = select i1 %.v18.i, i1 %v88, i1 false
  %v1023 = icmp ne ptr %v7, null
  %v102 = select i1 %or.cond.not, i1 %v1023, i1 false
  br i1 %v102, label %bb16, label %bb19

bb16:                                             ; preds = %bb15
  %v91 = getelementptr inbounds nuw float, ptr %v7, i64 %v22.i
  store float %v36.lcssa, ptr %v91, align 4
  br label %bb19

bb19:                                             ; preds = %bb15, %bb16, %entry
  ret void

bb27:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb28:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb29:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb30:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb31:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind
define ptx_kernel void @q8_0_gemm_warp(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, ptr writeonly captures(none) %v7, i64 %v8) #6 {
entry:
  %v22 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v23 = zext nneg i32 %v22 to i64
  %v24 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v25 = zext nneg i32 %v24 to i64
  %v26 = zext i32 %v6 to i64
  %v27 = zext i32 %v4 to i64
  %v28 = mul nuw i64 %v26, %v27
  %v29.not = icmp ugt i64 %v28, %v25
  br i1 %v29.not, label %bb4, label %bb31

bb4:                                              ; preds = %entry
  %v31.not = icmp eq i32 %v4, 0
  br i1 %v31.not, label %bb32, label %bb5

bb5:                                              ; preds = %bb4
  %v35 = zext i32 %v5 to i64
  %v39.not14 = icmp ult i32 %v22, %v5
  br i1 %v39.not14, label %bb7.lr.ph, label %bb16

bb7.lr.ph:                                        ; preds = %bb5
  %v4.frozen = freeze i32 %v4
  %v345 = udiv i32 %v24, %v4.frozen
  %v34.zext = zext nneg i32 %v345 to i64
  %0 = mul i32 %v345, %v4.frozen
  %v334.decomposed = sub i32 %v24, %0
  %v33.zext = zext nneg i32 %v334.decomposed to i64
  %v41 = mul nuw nsw i64 %v33.zext, %v35
  %v60 = mul nuw nsw i64 %v34.zext, %v35
  br label %bb7

bb7:                                              ; preds = %bb7.lr.ph, %bb15
  %v3816 = phi i64 [ %v23, %bb7.lr.ph ], [ %v85, %bb15 ]
  %v3715 = phi float [ 0.000000e+00, %bb7.lr.ph ], [ %v83, %bb15 ]
  %reass.add = add nuw i64 %v3816, %v41
  %reass.mul = mul i64 %reass.add, 34
  %v45 = icmp ult i64 %reass.mul, %v1
  br i1 %v45, label %bb8, label %bb33

bb8:                                              ; preds = %bb7
  %v49 = or disjoint i64 %reass.mul, 1
  %v50 = icmp ult i64 %v49, %v1
  br i1 %v50, label %bb9, label %bb34

bb9:                                              ; preds = %bb8
  %v47 = getelementptr inbounds i8, ptr %v0, i64 %reass.mul
  %v48 = load i8, ptr %v47, align 1
  %v52 = getelementptr inbounds i8, ptr %v0, i64 %v49
  %v53 = load i8, ptr %v52, align 1
  %v57 = alloca [2 x i8], align 2
  store i8 %v48, ptr %v57, align 2
  %v57.repack1 = getelementptr inbounds nuw i8, ptr %v57, i64 1
  store i8 %v53, ptr %v57.repack1, align 1
  %v58 = load i16, ptr %v57, align 2
  %v4.i = lshr i16 %v58, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v58, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v58, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb9
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %1 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %1
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb9
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb9
  %v44.i = shl nuw i32 %v6.i, 31
  %2 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %2 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v61 = add nuw i64 %v3816, %v60
  %v62 = shl i64 %v61, 5
  %v67 = add nuw i64 %reass.mul, 2
  br label %bb12

bb12:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb14
  %v6413 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v84, %bb14 ]
  %v6312 = phi float [ %v3715, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v83, %bb14 ]
  %v68 = add nuw i64 %v67, %v6413
  %v69 = icmp ult i64 %v68, %v1
  br i1 %v69, label %bb13, label %bb35

bb13:                                             ; preds = %bb12
  %v76 = add nuw nsw i64 %v6413, %v62
  %v78 = icmp ult i64 %v76, %v3
  br i1 %v78, label %bb14, label %bb36

bb14:                                             ; preds = %bb13
  %v71 = getelementptr inbounds i8, ptr %v0, i64 %v68
  %v72 = load i8, ptr %v71, align 1
  %v74 = sitofp i8 %v72 to float
  %v75 = fmul contract float %v55.i, %v74
  %v80 = getelementptr inbounds float, ptr %v2, i64 %v76
  %v81 = load float, ptr %v80, align 4
  %v82 = fmul contract float %v81, %v75
  %v83 = fadd contract float %v6312, %v82
  %v84 = add nuw nsw i64 %v6413, 1
  %exitcond = icmp eq i64 %v84, 32
  br i1 %exitcond, label %bb15, label %bb12

bb15:                                             ; preds = %bb14
  %v85 = add nuw nsw i64 %v3816, 32
  %v39.not = icmp samesign ult i64 %v85, %v35
  br i1 %v39.not, label %bb7, label %bb16

bb16:                                             ; preds = %bb15, %bb5
  %v37.lcssa = phi float [ 0.000000e+00, %bb5 ], [ %v83, %bb15 ]
  %v86 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_2, i64 %v23
  store float %v37.lcssa, ptr addrspace(3) %v86, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not = icmp samesign ult i32 %v22, 16
  br i1 %v91.not, label %bb21, label %bb25

bb21:                                             ; preds = %bb16
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v86, i64 64
  %v96 = load float, ptr addrspace(3) %gep, align 4
  %v98 = load float, ptr addrspace(3) %v86, align 4
  %v99 = fadd contract float %v96, %v98
  store float %v99, ptr addrspace(3) %v86, align 4
  br label %bb25

bb25:                                             ; preds = %bb16, %bb21
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.1 = icmp samesign ult i32 %v22, 8
  br i1 %v91.not.1, label %bb21.1, label %bb25.1

bb21.1:                                           ; preds = %bb25
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v86, i64 32
  %v96.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v98.1 = load float, ptr addrspace(3) %v86, align 4
  %v99.1 = fadd contract float %v96.1, %v98.1
  store float %v99.1, ptr addrspace(3) %v86, align 4
  br label %bb25.1

bb25.1:                                           ; preds = %bb21.1, %bb25
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.2 = icmp samesign ult i32 %v22, 4
  br i1 %v91.not.2, label %bb21.2, label %bb25.2

bb21.2:                                           ; preds = %bb25.1
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v86, i64 16
  %v96.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v98.2 = load float, ptr addrspace(3) %v86, align 4
  %v99.2 = fadd contract float %v96.2, %v98.2
  store float %v99.2, ptr addrspace(3) %v86, align 4
  br label %bb25.2

bb25.2:                                           ; preds = %bb21.2, %bb25.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.3 = icmp samesign ult i32 %v22, 2
  br i1 %v91.not.3, label %bb21.3, label %bb25.3

bb21.3:                                           ; preds = %bb25.2
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v86, i64 8
  %v96.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v98.3 = load float, ptr addrspace(3) %v86, align 4
  %v99.3 = fadd contract float %v96.3, %v98.3
  store float %v99.3, ptr addrspace(3) %v86, align 4
  br label %bb25.3

bb25.3:                                           ; preds = %bb21.3, %bb25.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v91.not.4 = icmp eq i32 %v22, 0
  br i1 %v91.not.4, label %bb21.4, label %bb25.4

bb21.4:                                           ; preds = %bb25.3
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v86, i64 4
  %v96.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v98.4 = load float, ptr addrspace(3) %v86, align 4
  %v99.4 = fadd contract float %v96.4, %v98.4
  store float %v99.4, ptr addrspace(3) %v86, align 4
  br label %bb25.4

bb25.4:                                           ; preds = %bb21.4, %bb25.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v102 = icmp eq i32 %v22, 0
  br i1 %v102, label %bb28, label %bb31

bb28:                                             ; preds = %bb25.4
  %v107 = getelementptr inbounds nuw float, ptr %v7, i64 %v25
  %v105 = load float, ptr addrspace(3) @__shared_mem_2, align 4
  store float %v105, ptr %v107, align 4
  br label %bb31

bb31:                                             ; preds = %bb25.4, %bb28, %entry
  ret void

bb32:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb34:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb35:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb36:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent norecurse nounwind
define ptx_kernel void @quantize_q8_32(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3, ptr writeonly captures(none) %v4, i64 %v5) #1 {
entry:
  %v15 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v16 = zext nneg i32 %v15 to i64
  %v17 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v18 = zext nneg i32 %v17 to i64
  %v19 = shl nuw nsw i64 %v18, 5
  %v20 = add nuw nsw i64 %v19, %v16
  %v22.not = icmp ult i64 %v20, %v1
  br i1 %v22.not, label %bb5, label %bb25

bb5:                                              ; preds = %entry
  %v26 = getelementptr inbounds nuw float, ptr %v0, i64 %v20
  %v27 = load float, ptr %v26, align 4
  %v28 = tail call float @llvm.fabs.f32(float %v27) #19
  %v63 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_4, i64 %v16
  store float %v28, ptr addrspace(3) %v63, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v33.not = icmp samesign ult i32 %v15, 16
  br i1 %v33.not, label %bb10, label %bb15

bb10:                                             ; preds = %bb5
  %v37 = load float, ptr addrspace(3) %v63, align 4
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) %v63, i64 64
  %v41 = load float, ptr addrspace(3) %gep, align 4
  %v42 = fcmp uno float %v37, 0.000000e+00
  %v43 = fcmp oge float %v41, %v37
  %0 = select i1 %v42, i1 true, i1 %v43
  %v45 = select i1 %0, float %v41, float %v37
  store float %v45, ptr addrspace(3) %v63, align 4
  br label %bb15

bb15:                                             ; preds = %bb5, %bb10
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v33.not.1 = icmp samesign ult i32 %v15, 8
  br i1 %v33.not.1, label %bb10.1, label %bb15.1

bb10.1:                                           ; preds = %bb15
  %v37.1 = load float, ptr addrspace(3) %v63, align 4
  %gep.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %v63, i64 32
  %v41.1 = load float, ptr addrspace(3) %gep.1, align 4
  %v42.1 = fcmp uno float %v37.1, 0.000000e+00
  %v43.1 = fcmp oge float %v41.1, %v37.1
  %1 = select i1 %v42.1, i1 true, i1 %v43.1
  %v45.1 = select i1 %1, float %v41.1, float %v37.1
  store float %v45.1, ptr addrspace(3) %v63, align 4
  br label %bb15.1

bb15.1:                                           ; preds = %bb10.1, %bb15
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v33.not.2 = icmp samesign ult i32 %v15, 4
  br i1 %v33.not.2, label %bb10.2, label %bb15.2

bb10.2:                                           ; preds = %bb15.1
  %v37.2 = load float, ptr addrspace(3) %v63, align 4
  %gep.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %v63, i64 16
  %v41.2 = load float, ptr addrspace(3) %gep.2, align 4
  %v42.2 = fcmp uno float %v37.2, 0.000000e+00
  %v43.2 = fcmp oge float %v41.2, %v37.2
  %2 = select i1 %v42.2, i1 true, i1 %v43.2
  %v45.2 = select i1 %2, float %v41.2, float %v37.2
  store float %v45.2, ptr addrspace(3) %v63, align 4
  br label %bb15.2

bb15.2:                                           ; preds = %bb10.2, %bb15.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v33.not.3 = icmp samesign ult i32 %v15, 2
  br i1 %v33.not.3, label %bb10.3, label %bb15.3

bb10.3:                                           ; preds = %bb15.2
  %v37.3 = load float, ptr addrspace(3) %v63, align 4
  %gep.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %v63, i64 8
  %v41.3 = load float, ptr addrspace(3) %gep.3, align 4
  %v42.3 = fcmp uno float %v37.3, 0.000000e+00
  %v43.3 = fcmp oge float %v41.3, %v37.3
  %3 = select i1 %v42.3, i1 true, i1 %v43.3
  %v45.3 = select i1 %3, float %v41.3, float %v37.3
  store float %v45.3, ptr addrspace(3) %v63, align 4
  br label %bb15.3

bb15.3:                                           ; preds = %bb10.3, %bb15.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v33.not.4 = icmp eq i32 %v15, 0
  br i1 %v33.not.4, label %bb10.4, label %bb15.4

bb10.4:                                           ; preds = %bb15.3
  %v37.4 = load float, ptr addrspace(3) %v63, align 4
  %gep.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %v63, i64 4
  %v41.4 = load float, ptr addrspace(3) %gep.4, align 4
  %v42.4 = fcmp uno float %v37.4, 0.000000e+00
  %v43.4 = fcmp oge float %v41.4, %v37.4
  %4 = select i1 %v42.4, i1 true, i1 %v43.4
  %v45.4 = select i1 %4, float %v41.4, float %v37.4
  store float %v45.4, ptr addrspace(3) %v63, align 4
  br label %bb15.4

bb15.4:                                           ; preds = %bb10.4, %bb15.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v50 = load float, ptr addrspace(3) @__shared_mem_4, align 4
  %v51 = fdiv contract float %v50, 1.270000e+02
  %v52 = fcmp ule float %v51, 0.000000e+00
  br i1 %v52, label %bb22, label %bb19

bb19:                                             ; preds = %bb15.4
  %v54 = fdiv contract float %v27, %v51
  %v55 = tail call float @llvm.round.f32(float %v54) #19
  %v8.inv.i = fcmp olt float %v55, -1.270000e+02
  %v0.v1.i = select i1 %v8.inv.i, float -1.270000e+02, float %v55
  %v12.inv.i = fcmp ogt float %v0.v1.i, 1.270000e+02
  %v14.i = select i1 %v12.inv.i, float 1.270000e+02, float %v0.v1.i
  %v56 = tail call i8 @llvm.fptosi.sat.i8.f32(float %v14.i) #19
  br label %bb22

bb22:                                             ; preds = %bb15.4, %bb19
  %v57 = phi i8 [ %v56, %bb19 ], [ 0, %bb15.4 ]
  %v59 = getelementptr inbounds nuw i8, ptr %v2, i64 %v20
  store i8 %v57, ptr %v59, align 1
  %v60 = icmp eq i32 %v15, 0
  br i1 %v60, label %bb23, label %bb25

bb23:                                             ; preds = %bb22
  %v62 = getelementptr inbounds nuw float, ptr %v4, i64 %v18
  store float %v51, ptr %v62, align 4
  br label %bb25

bb25:                                             ; preds = %bb22, %bb23, %entry
  ret void
}

; Function Attrs: convergent nounwind
define ptx_kernel void @rmsnorm_group(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, float %v4, i32 %v5, i32 %v6, ptr writeonly captures(none) %v7, i64 %v8) #6 {
entry:
  %v21 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v22 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v23 = mul i32 %v6, %v22
  %v24 = zext i32 %v23 to i64
  %v25 = zext i32 %v5 to i64
  %v26 = zext nneg i32 %v21 to i64
  %v29.not3 = icmp ult i32 %v21, %v5
  br i1 %v29.not3, label %bb4, label %bb6

bb4:                                              ; preds = %entry, %bb5
  %v285 = phi i64 [ %v39, %bb5 ], [ %v26, %entry ]
  %v274 = phi float [ %v38, %bb5 ], [ 0.000000e+00, %entry ]
  %v31 = add nuw i64 %v285, %v24
  %v33 = icmp ult i64 %v31, %v1
  br i1 %v33, label %bb5, label %bb25

bb5:                                              ; preds = %bb4
  %v35 = getelementptr inbounds float, ptr %v0, i64 %v31
  %v36 = load float, ptr %v35, align 4
  %v37 = fmul contract float %v36, %v36
  %v38 = fadd contract float %v274, %v37
  %v39 = add nuw nsw i64 %v285, 256
  %v29.not = icmp samesign ult i64 %v39, %v25
  br i1 %v29.not, label %bb4, label %bb6

bb6:                                              ; preds = %bb5, %entry
  %v27.lcssa = phi float [ 0.000000e+00, %entry ], [ %v38, %bb5 ]
  %v41 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %v26
  store float %v27.lcssa, ptr addrspace(3) %v41, align 4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not = icmp samesign ult i32 %v21, 128
  br i1 %v46.not, label %bb11, label %bb15

bb11:                                             ; preds = %bb6
  %0 = zext nneg i32 %v21 to i64
  %1 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %0
  %v51 = getelementptr inbounds nuw i8, ptr addrspace(3) %1, i64 512
  %v52 = load float, ptr addrspace(3) %v51, align 4
  %v54 = load float, ptr addrspace(3) %v41, align 4
  %v55 = fadd contract float %v52, %v54
  store float %v55, ptr addrspace(3) %v41, align 4
  br label %bb15

bb15:                                             ; preds = %bb6, %bb11
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.1 = icmp samesign ult i32 %v21, 64
  br i1 %v46.not.1, label %bb11.1, label %bb15.1

bb11.1:                                           ; preds = %bb15
  %2 = zext nneg i32 %v21 to i64
  %3 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %2
  %v51.1 = getelementptr inbounds nuw i8, ptr addrspace(3) %3, i64 256
  %v52.1 = load float, ptr addrspace(3) %v51.1, align 4
  %v54.1 = load float, ptr addrspace(3) %v41, align 4
  %v55.1 = fadd contract float %v52.1, %v54.1
  store float %v55.1, ptr addrspace(3) %v41, align 4
  br label %bb15.1

bb15.1:                                           ; preds = %bb11.1, %bb15
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.2 = icmp samesign ult i32 %v21, 32
  br i1 %v46.not.2, label %bb11.2, label %bb15.2

bb11.2:                                           ; preds = %bb15.1
  %4 = zext nneg i32 %v21 to i64
  %5 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %4
  %v51.2 = getelementptr inbounds nuw i8, ptr addrspace(3) %5, i64 128
  %v52.2 = load float, ptr addrspace(3) %v51.2, align 4
  %v54.2 = load float, ptr addrspace(3) %v41, align 4
  %v55.2 = fadd contract float %v52.2, %v54.2
  store float %v55.2, ptr addrspace(3) %v41, align 4
  br label %bb15.2

bb15.2:                                           ; preds = %bb11.2, %bb15.1
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.3 = icmp samesign ult i32 %v21, 16
  br i1 %v46.not.3, label %bb11.3, label %bb15.3

bb11.3:                                           ; preds = %bb15.2
  %6 = zext nneg i32 %v21 to i64
  %7 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %6
  %v51.3 = getelementptr inbounds nuw i8, ptr addrspace(3) %7, i64 64
  %v52.3 = load float, ptr addrspace(3) %v51.3, align 4
  %v54.3 = load float, ptr addrspace(3) %v41, align 4
  %v55.3 = fadd contract float %v52.3, %v54.3
  store float %v55.3, ptr addrspace(3) %v41, align 4
  br label %bb15.3

bb15.3:                                           ; preds = %bb11.3, %bb15.2
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.4 = icmp samesign ult i32 %v21, 8
  br i1 %v46.not.4, label %bb11.4, label %bb15.4

bb11.4:                                           ; preds = %bb15.3
  %8 = zext nneg i32 %v21 to i64
  %9 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %8
  %v51.4 = getelementptr inbounds nuw i8, ptr addrspace(3) %9, i64 32
  %v52.4 = load float, ptr addrspace(3) %v51.4, align 4
  %v54.4 = load float, ptr addrspace(3) %v41, align 4
  %v55.4 = fadd contract float %v52.4, %v54.4
  store float %v55.4, ptr addrspace(3) %v41, align 4
  br label %bb15.4

bb15.4:                                           ; preds = %bb11.4, %bb15.3
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.5 = icmp samesign ult i32 %v21, 4
  br i1 %v46.not.5, label %bb11.5, label %bb15.5

bb11.5:                                           ; preds = %bb15.4
  %10 = zext nneg i32 %v21 to i64
  %11 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %10
  %v51.5 = getelementptr inbounds nuw i8, ptr addrspace(3) %11, i64 16
  %v52.5 = load float, ptr addrspace(3) %v51.5, align 4
  %v54.5 = load float, ptr addrspace(3) %v41, align 4
  %v55.5 = fadd contract float %v52.5, %v54.5
  store float %v55.5, ptr addrspace(3) %v41, align 4
  br label %bb15.5

bb15.5:                                           ; preds = %bb11.5, %bb15.4
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.6 = icmp samesign ult i32 %v21, 2
  br i1 %v46.not.6, label %bb11.6, label %bb15.6

bb11.6:                                           ; preds = %bb15.5
  %12 = zext nneg i32 %v21 to i64
  %13 = getelementptr inbounds nuw float, ptr addrspace(3) @__shared_mem_14, i64 %12
  %v51.6 = getelementptr inbounds nuw i8, ptr addrspace(3) %13, i64 8
  %v52.6 = load float, ptr addrspace(3) %v51.6, align 4
  %v54.6 = load float, ptr addrspace(3) %v41, align 4
  %v55.6 = fadd contract float %v52.6, %v54.6
  store float %v55.6, ptr addrspace(3) %v41, align 4
  br label %bb15.6

bb15.6:                                           ; preds = %bb11.6, %bb15.5
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v46.not.7 = icmp eq i32 %v21, 0
  br i1 %v46.not.7, label %bb11.7, label %bb15.7

bb11.7:                                           ; preds = %bb15.6
  %v52.7 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @__shared_mem_14, i64 4), align 4
  %v54.7 = load float, ptr addrspace(3) %v41, align 4
  %v55.7 = fadd contract float %v52.7, %v54.7
  store float %v55.7, ptr addrspace(3) %v41, align 4
  br label %bb15.7

bb15.7:                                           ; preds = %bb11.7, %bb15.6
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #19
  %v60 = load float, ptr addrspace(3) @__shared_mem_14, align 4
  %v61 = uitofp i32 %v5 to float
  %v62 = fdiv contract float %v60, %v61
  %v63 = fadd contract float %v4, %v62
  %14 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %14, 0
  %15 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #20
  %.not1.i = icmp eq i32 %15, 0
  br i1 %.not.i, label %21, label %16

16:                                               ; preds = %bb15.7
  br i1 %.not1.i, label %19, label %17

17:                                               ; preds = %16
  %18 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v63) #20
  br label %__nv_sqrtf.exit

19:                                               ; preds = %16
  %20 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v63) #20
  br label %__nv_sqrtf.exit

21:                                               ; preds = %bb15.7
  br i1 %.not1.i, label %24, label %22

22:                                               ; preds = %21
  %23 = tail call float @llvm.nvvm.sqrt.rn.f(float %v63) #20
  br label %__nv_sqrtf.exit

24:                                               ; preds = %21
  %25 = tail call float @llvm.nvvm.sqrt.approx.f(float %v63) #20
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %17, %19, %22, %24
  %.0.i = phi float [ %18, %17 ], [ %20, %19 ], [ %23, %22 ], [ %25, %24 ]
  %v85 = fdiv contract float 1.000000e+00, %.0.i
  br i1 %v29.not3, label %bb20, label %bb23

bb20:                                             ; preds = %__nv_sqrtf.exit, %bb22
  %v658 = phi i64 [ %v84, %bb22 ], [ %v26, %__nv_sqrtf.exit ]
  %v68 = add nuw i64 %v658, %v24
  %v70 = icmp ult i64 %v68, %v1
  br i1 %v70, label %bb21, label %bb26

bb21:                                             ; preds = %bb20
  %v76 = icmp ult i64 %v658, %v3
  br i1 %v76, label %bb22, label %bb27

bb22:                                             ; preds = %bb21
  %v72 = getelementptr inbounds float, ptr %v0, i64 %v68
  %v73 = load float, ptr %v72, align 4
  %v74 = fmul contract float %v85, %v73
  %v78 = getelementptr inbounds nuw float, ptr %v2, i64 %v658
  %v79 = load float, ptr %v78, align 4
  %v82 = getelementptr inbounds float, ptr %v7, i64 %v68
  %v83 = fmul contract float %v74, %v79
  store float %v83, ptr %v82, align 4
  %v84 = add nuw nsw i64 %v658, 256
  %v66.not = icmp samesign ult i64 %v84, %v25
  br i1 %v66.not, label %bb20, label %bb23

bb23:                                             ; preds = %bb22, %__nv_sqrtf.exit
  ret void

bb25:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb26:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb27:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @rope(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, float %v7, i32 %v8, ptr writeonly captures(none) %v9, i64 %v10) #0 {
entry:
  %result.i.i.i.i7 = alloca [7 x i32], align 4
  %result.i.i.i.i = alloca [7 x i32], align 4
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v28 = zext i32 %v8 to i64
  %v29.not = icmp ult i64 %v22.i, %v28
  br i1 %v29.not, label %bb3, label %bb15

bb3:                                              ; preds = %entry
  %v32.not = icmp eq i32 %v5, 0
  br i1 %v32.not, label %bb18, label %bb4

bb4:                                              ; preds = %bb3
  %v34.lhs.trunc = trunc nuw i64 %v22.i to i32
  %v3442 = urem i32 %v34.lhs.trunc, %v5
  %v34.zext = zext i32 %v3442 to i64
  %v38.not = icmp eq i32 %v4, 0
  br i1 %v38.not, label %bb19, label %bb5

bb5:                                              ; preds = %bb4
  %v31 = zext i32 %v5 to i64
  %v36 = zext i32 %v4 to i64
  %v37 = mul nuw i64 %v31, %v36
  %v40 = udiv i64 %v22.i, %v37
  %v42.not = icmp ult i32 %v3442, %v6
  br i1 %v42.not, label %bb8, label %bb6

bb6:                                              ; preds = %bb5
  %v45 = icmp ult i64 %v22.i, %v1
  br i1 %v45, label %bb7, label %bb20

bb7:                                              ; preds = %bb6
  %v47 = getelementptr inbounds nuw float, ptr %v0, i64 %v22.i
  %v48 = load float, ptr %v47, align 4
  %v50 = getelementptr inbounds nuw float, ptr %v9, i64 %v22.i
  store float %v48, ptr %v50, align 4
  br label %bb15

bb8:                                              ; preds = %bb5
  %v51 = and i64 %v34.zext, 1
  %v52.not = icmp eq i64 %v51, 0
  br i1 %v52.not, label %bb10, label %bb15

bb10:                                             ; preds = %bb8
  %v531 = lshr exact i64 %v34.zext, 1
  %v57 = icmp ult i64 %v531, %v3
  br i1 %v57, label %bb11, label %bb21

bb11:                                             ; preds = %bb10
  %v54 = uitofp nneg i64 %v40 to float
  %v55 = fadd contract float %v7, %v54
  %v59 = getelementptr inbounds nuw float, ptr %v2, i64 %v531
  %v60 = load float, ptr %v59, align 4
  %v61 = fmul contract float %v55, %v60
  call void @llvm.lifetime.start.p0(ptr nonnull %result.i.i.i.i)
  %0 = fmul float %v61, 0x3FE45F3060000000
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %1, 0
  %2 = tail call i32 @llvm.nvvm.f2i.rn.ftz(float %0) #20
  %3 = tail call i32 @llvm.nvvm.f2i.rn(float %0) #20
  %.01.i = select i1 %.not.i, i32 %3, i32 %2
  %4 = sitofp i32 %.01.i to float
  %5 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %4, float 0xBFF921FB40000000, float %v61) #20
  %6 = tail call float @llvm.fma.f32(float %4, float 0xBFF921FB40000000, float %v61)
  %.02.i = select i1 %.not.i, float %6, float %5
  %7 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %4, float 0xBE74442D00000000, float %.02.i) #20
  %8 = tail call float @llvm.fma.f32(float %4, float 0xBE74442D00000000, float %.02.i)
  %.03.i = select i1 %.not.i, float %8, float %7
  %9 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %4, float 0xBCF84698A0000000, float %.03.i) #20
  %10 = tail call float @llvm.fma.f32(float %4, float 0xBCF84698A0000000, float %.03.i)
  %.04.i = select i1 %.not.i, float %10, float %9
  %11 = tail call float @llvm.nvvm.fabs.ftz.f32(float %v61)
  %12 = tail call float @llvm.nvvm.fabs.f32(float %v61)
  %.06.i = select i1 %.not.i, float %12, float %11
  %13 = fcmp ult float %.06.i, 1.056150e+05
  br i1 %13, label %__nv_sinf.exit, label %__nv_isinff.exit.i.i.i

__nv_isinff.exit.i.i.i:                           ; preds = %bb11
  %14 = fcmp oeq float %.06.i, 0x7FF0000000000000
  br i1 %14, label %__nv_fmul_rn.exit.i.i.i, label %17

__nv_fmul_rn.exit.i.i.i:                          ; preds = %__nv_isinff.exit.i.i.i
  %15 = tail call float @llvm.nvvm.mul.rn.ftz.f(float %v61, float 0.000000e+00) #20
  %16 = tail call float @llvm.nvvm.mul.rn.f(float %v61, float 0.000000e+00) #20
  %.08.i = select i1 %.not.i, float %16, float %15
  br label %__nv_sinf.exit

17:                                               ; preds = %__nv_isinff.exit.i.i.i
  %18 = bitcast float %v61 to i32
  %19 = shl i32 %18, 8
  %20 = or i32 %19, -2147483648
  br label %21

21:                                               ; preds = %17, %21
  %iq.i.i.i.0.i44 = phi i32 [ 0, %17 ], [ %29, %21 ]
  %hi.i.i.i.0.i43 = phi i32 [ 0, %17 ], [ %27, %21 ]
  %22 = zext nneg i32 %iq.i.i.i.0.i44 to i64
  %23 = getelementptr inbounds nuw i32, ptr addrspace(1) @__cudart_i2opi_f, i64 %22
  %24 = load i32, ptr addrspace(1) %23, align 4
  %25 = tail call { i32, i32 } asm "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r"(i32 %24, i32 %20, i32 %hi.i.i.i.0.i43) #21, !srcloc !14
  %26 = extractvalue { i32, i32 } %25, 0
  %27 = extractvalue { i32, i32 } %25, 1
  %28 = getelementptr inbounds nuw i32, ptr %result.i.i.i.i, i64 %22
  store i32 %26, ptr %28, align 4
  %29 = add nuw nsw i32 %iq.i.i.i.0.i44, 1
  %exitcond.not = icmp eq i32 %29, 6
  br i1 %exitcond.not, label %30, label %21, !llvm.loop !15

30:                                               ; preds = %21
  %31 = lshr i32 %18, 23
  %32 = and i32 %31, 224
  %33 = add nsw i32 %32, -128
  %34 = lshr exact i32 %33, 5
  %35 = getelementptr inbounds nuw i8, ptr %result.i.i.i.i, i64 24
  store i32 %27, ptr %35, align 4
  %36 = sub nsw i32 6, %34
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds i32, ptr %result.i.i.i.i, i64 %37
  %39 = load i32, ptr %38, align 4
  %40 = sub nsw i32 5, %34
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds i32, ptr %result.i.i.i.i, i64 %41
  %43 = load i32, ptr %42, align 4
  %44 = freeze i32 %43
  %45 = and i32 %18, 260046848
  %.not8.i = icmp eq i32 %45, 0
  br i1 %.not8.i, label %__internal_trig_reduction_slowpath.exit.i.i.i, label %46

46:                                               ; preds = %30
  %47 = sub nsw i32 4, %34
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds i32, ptr %result.i.i.i.i, i64 %48
  %50 = load i32, ptr %49, align 4
  %51 = tail call i32 @llvm.fshl.i32(i32 %44, i32 %50, i32 %31)
  br label %__internal_trig_reduction_slowpath.exit.i.i.i

__internal_trig_reduction_slowpath.exit.i.i.i:    ; preds = %46, %30
  %lo.i.i.i.0.i = phi i32 [ %51, %46 ], [ %44, %30 ]
  %52 = tail call i32 @llvm.fshl.i32(i32 %39, i32 %44, i32 %31)
  %53 = lshr i32 %52, 30
  %54 = tail call i32 @llvm.fshl.i32(i32 %52, i32 %lo.i.i.i.0.i, i32 2)
  %55 = shl i32 %lo.i.i.i.0.i, 2
  %56 = lshr i32 %54, 31
  %57 = add nuw nsw i32 %56, %53
  %58 = sub nsw i32 0, %57
  %.not911.i = icmp slt i32 %18, 0
  %spec.select.i = select i1 %.not911.i, i32 %58, i32 %57
  %59 = xor i32 %54, %18
  %.lobit.i = ashr i32 %54, 31
  %hi.i.i.i.2.i = xor i32 %.lobit.i, %54
  %lo.i.i.i.1.i = xor i32 %.lobit.i, %55
  %60 = zext i32 %hi.i.i.i.2.i to i64
  %61 = shl nuw i64 %60, 32
  %62 = zext i32 %lo.i.i.i.1.i to i64
  %63 = or disjoint i64 %61, %62
  %64 = sitofp i64 %63 to double
  %65 = fmul double %64, 0x3BF921FB54442D19
  %66 = fptrunc double %65 to float
  %67 = fneg float %66
  %.not1314.i = icmp slt i32 %59, 0
  %r.i.i.i.0.i = select i1 %.not1314.i, float %67, float %66
  br label %__nv_sinf.exit

__nv_sinf.exit:                                   ; preds = %bb11, %__nv_fmul_rn.exit.i.i.i, %__internal_trig_reduction_slowpath.exit.i.i.i
  %i.i.1.i = phi i32 [ %.01.i, %bb11 ], [ 0, %__nv_fmul_rn.exit.i.i.i ], [ %spec.select.i, %__internal_trig_reduction_slowpath.exit.i.i.i ]
  %t.i.i.1.i = phi float [ %.04.i, %bb11 ], [ %.08.i, %__nv_fmul_rn.exit.i.i.i ], [ %r.i.i.i.0.i, %__internal_trig_reduction_slowpath.exit.i.i.i ]
  %68 = tail call float @llvm.nvvm.mul.rn.ftz.f(float %t.i.i.1.i, float %t.i.i.1.i) #20
  %69 = tail call float @llvm.nvvm.mul.rn.f(float %t.i.i.1.i, float %t.i.i.1.i) #20
  %.011.i = select i1 %.not.i, float %69, float %68
  %70 = and i32 %i.i.1.i, 1
  %.not15.i = icmp eq i32 %70, 0
  %71 = select i1 %.not15.i, float %t.i.i.1.i, float 1.000000e+00
  %72 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.011.i, float %71, float 0.000000e+00) #20
  %73 = tail call float @llvm.fma.f32(float %.011.i, float %71, float 0.000000e+00)
  %.012.i = select i1 %.not.i, float %73, float %72
  %74 = tail call float @llvm.nvvm.fma.rn.ftz.f(float 0x3EF9758000000000, float %.011.i, float 0xBF56C0FDA0000000) #20
  %75 = tail call float @llvm.fma.f32(float %.011.i, float 0x3EF9758000000000, float 0xBF56C0FDA0000000)
  %.013.i = select i1 %.not.i, float %75, float %74
  %76 = select i1 %.not15.i, float 0xBFC5555500000000, float 0xBFDFFFFFE0000000
  %77 = select i1 %.not15.i, float 0x3F8110BC80000000, float 0x3FA5555760000000
  %78 = select i1 %.not15.i, float 0xBF29A82A60000000, float %.013.i
  %79 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %78, float %.011.i, float %77) #20
  %80 = tail call float @llvm.fma.f32(float %78, float %.011.i, float %77)
  %.010.i = select i1 %.not.i, float %80, float %79
  %81 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.010.i, float %.011.i, float %76) #20
  %82 = tail call float @llvm.fma.f32(float %.010.i, float %.011.i, float %76)
  %.09.i = select i1 %.not.i, float %82, float %81
  %83 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.09.i, float %.012.i, float %71) #20
  %84 = tail call float @llvm.fma.f32(float %.09.i, float %.012.i, float %71)
  %.05.i = select i1 %.not.i, float %84, float %83
  %85 = and i32 %i.i.1.i, 2
  %.not16.i = icmp eq i32 %85, 0
  %86 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.05.i, float -1.000000e+00, float 0.000000e+00) #20
  %87 = fsub float 0.000000e+00, %84
  %.0.i = select i1 %.not.i, float %87, float %86
  %z.i.i.0.i = select i1 %.not16.i, float %.05.i, float %.0.i
  call void @llvm.lifetime.end.p0(ptr nonnull %result.i.i.i.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %result.i.i.i.i7)
  br i1 %13, label %__nv_cosf.exit, label %__nv_isinff.exit.i.i.i14

__nv_isinff.exit.i.i.i14:                         ; preds = %__nv_sinf.exit
  %88 = fcmp oeq float %.06.i, 0x7FF0000000000000
  br i1 %88, label %__nv_fmul_rn.exit.i.i.i40, label %91

__nv_fmul_rn.exit.i.i.i40:                        ; preds = %__nv_isinff.exit.i.i.i14
  %89 = tail call float @llvm.nvvm.mul.rn.ftz.f(float %v61, float 0.000000e+00) #20
  %90 = tail call float @llvm.nvvm.mul.rn.f(float %v61, float 0.000000e+00) #20
  %.08.i41 = select i1 %.not.i, float %90, float %89
  br label %__nv_cosf.exit

91:                                               ; preds = %__nv_isinff.exit.i.i.i14
  %92 = bitcast float %v61 to i32
  %93 = shl i32 %92, 8
  %94 = or i32 %93, -2147483648
  br label %95

95:                                               ; preds = %91, %95
  %iq.i.i.i.0.i1646 = phi i32 [ 0, %91 ], [ %103, %95 ]
  %hi.i.i.i.0.i1545 = phi i32 [ 0, %91 ], [ %101, %95 ]
  %96 = zext nneg i32 %iq.i.i.i.0.i1646 to i64
  %97 = getelementptr inbounds nuw i32, ptr addrspace(1) @__cudart_i2opi_f, i64 %96
  %98 = load i32, ptr addrspace(1) %97, align 4
  %99 = tail call { i32, i32 } asm "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r"(i32 %98, i32 %94, i32 %hi.i.i.i.0.i1545) #21, !srcloc !14
  %100 = extractvalue { i32, i32 } %99, 0
  %101 = extractvalue { i32, i32 } %99, 1
  %102 = getelementptr inbounds nuw i32, ptr %result.i.i.i.i7, i64 %96
  store i32 %100, ptr %102, align 4
  %103 = add nuw nsw i32 %iq.i.i.i.0.i1646, 1
  %exitcond50.not = icmp eq i32 %103, 6
  br i1 %exitcond50.not, label %104, label %95, !llvm.loop !15

104:                                              ; preds = %95
  %105 = lshr i32 %92, 23
  %106 = and i32 %105, 224
  %107 = add nsw i32 %106, -128
  %108 = lshr exact i32 %107, 5
  %109 = getelementptr inbounds nuw i8, ptr %result.i.i.i.i7, i64 24
  store i32 %101, ptr %109, align 4
  %110 = sub nsw i32 6, %108
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds i32, ptr %result.i.i.i.i7, i64 %111
  %113 = load i32, ptr %112, align 4
  %114 = sub nsw i32 5, %108
  %115 = sext i32 %114 to i64
  %116 = getelementptr inbounds i32, ptr %result.i.i.i.i7, i64 %115
  %117 = load i32, ptr %116, align 4
  %118 = freeze i32 %117
  %119 = and i32 %92, 260046848
  %.not8.i17 = icmp eq i32 %119, 0
  br i1 %.not8.i17, label %__internal_trig_reduction_slowpath.exit.i.i.i18, label %120

120:                                              ; preds = %104
  %121 = sub nsw i32 4, %108
  %122 = sext i32 %121 to i64
  %123 = getelementptr inbounds i32, ptr %result.i.i.i.i7, i64 %122
  %124 = load i32, ptr %123, align 4
  %125 = tail call i32 @llvm.fshl.i32(i32 %118, i32 %124, i32 %105)
  br label %__internal_trig_reduction_slowpath.exit.i.i.i18

__internal_trig_reduction_slowpath.exit.i.i.i18:  ; preds = %120, %104
  %lo.i.i.i.0.i20 = phi i32 [ %125, %120 ], [ %118, %104 ]
  %126 = tail call i32 @llvm.fshl.i32(i32 %113, i32 %118, i32 %105)
  %127 = lshr i32 %126, 30
  %128 = tail call i32 @llvm.fshl.i32(i32 %126, i32 %lo.i.i.i.0.i20, i32 2)
  %129 = shl i32 %lo.i.i.i.0.i20, 2
  %130 = lshr i32 %128, 31
  %131 = add nuw nsw i32 %130, %127
  %132 = sub nsw i32 0, %131
  %.not911.i21 = icmp slt i32 %92, 0
  %spec.select.i22 = select i1 %.not911.i21, i32 %132, i32 %131
  %133 = xor i32 %128, %92
  %.lobit.i23 = ashr i32 %128, 31
  %hi.i.i.i.2.i24 = xor i32 %.lobit.i23, %128
  %lo.i.i.i.1.i26 = xor i32 %.lobit.i23, %129
  %134 = zext i32 %hi.i.i.i.2.i24 to i64
  %135 = shl nuw i64 %134, 32
  %136 = zext i32 %lo.i.i.i.1.i26 to i64
  %137 = or disjoint i64 %135, %136
  %138 = sitofp i64 %137 to double
  %139 = fmul double %138, 0x3BF921FB54442D19
  %140 = fptrunc double %139 to float
  %141 = fneg float %140
  %.not1314.i27 = icmp slt i32 %133, 0
  %r.i.i.i.0.i28 = select i1 %.not1314.i27, float %141, float %140
  br label %__nv_cosf.exit

__nv_cosf.exit:                                   ; preds = %__nv_sinf.exit, %__nv_fmul_rn.exit.i.i.i40, %__internal_trig_reduction_slowpath.exit.i.i.i18
  %i.i.1.i29 = phi i32 [ %.01.i, %__nv_sinf.exit ], [ 0, %__nv_fmul_rn.exit.i.i.i40 ], [ %spec.select.i22, %__internal_trig_reduction_slowpath.exit.i.i.i18 ]
  %t.i.i.1.i30 = phi float [ %.04.i, %__nv_sinf.exit ], [ %.08.i41, %__nv_fmul_rn.exit.i.i.i40 ], [ %r.i.i.i.0.i28, %__internal_trig_reduction_slowpath.exit.i.i.i18 ]
  %142 = add i32 %i.i.1.i29, 1
  %143 = tail call float @llvm.nvvm.mul.rn.ftz.f(float %t.i.i.1.i30, float %t.i.i.1.i30) #20
  %144 = tail call float @llvm.nvvm.mul.rn.f(float %t.i.i.1.i30, float %t.i.i.1.i30) #20
  %.011.i31 = select i1 %.not.i, float %144, float %143
  %145 = and i32 %i.i.1.i29, 1
  %.not15.not.i = icmp eq i32 %145, 0
  %146 = select i1 %.not15.not.i, float 1.000000e+00, float %t.i.i.1.i30
  %147 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.011.i31, float %146, float 0.000000e+00) #20
  %148 = tail call float @llvm.fma.f32(float %.011.i31, float %146, float 0.000000e+00)
  %.012.i32 = select i1 %.not.i, float %148, float %147
  %149 = tail call float @llvm.nvvm.fma.rn.ftz.f(float 0x3EF9758000000000, float %.011.i31, float 0xBF56C0FDA0000000) #20
  %150 = tail call float @llvm.fma.f32(float %.011.i31, float 0x3EF9758000000000, float 0xBF56C0FDA0000000)
  %.013.i33 = select i1 %.not.i, float %150, float %149
  %151 = select i1 %.not15.not.i, float 0xBFDFFFFFE0000000, float 0xBFC5555500000000
  %152 = select i1 %.not15.not.i, float 0x3FA5555760000000, float 0x3F8110BC80000000
  %153 = select i1 %.not15.not.i, float %.013.i33, float 0xBF29A82A60000000
  %154 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %153, float %.011.i31, float %152) #20
  %155 = tail call float @llvm.fma.f32(float %153, float %.011.i31, float %152)
  %.010.i34 = select i1 %.not.i, float %155, float %154
  %156 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.010.i34, float %.011.i31, float %151) #20
  %157 = tail call float @llvm.fma.f32(float %.010.i34, float %.011.i31, float %151)
  %.09.i35 = select i1 %.not.i, float %157, float %156
  %158 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.09.i35, float %.012.i32, float %146) #20
  %159 = tail call float @llvm.fma.f32(float %.09.i35, float %.012.i32, float %146)
  %.05.i36 = select i1 %.not.i, float %159, float %158
  %160 = and i32 %142, 2
  %.not16.i37 = icmp eq i32 %160, 0
  %161 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %.05.i36, float -1.000000e+00, float 0.000000e+00) #20
  %162 = fsub float 0.000000e+00, %159
  %.0.i38 = select i1 %.not.i, float %162, float %161
  %z.i.i.0.i39 = select i1 %.not16.i37, float %.05.i36, float %.0.i38
  call void @llvm.lifetime.end.p0(ptr nonnull %result.i.i.i.i7)
  %v83 = icmp ult i64 %v22.i, %v1
  br i1 %v83, label %bb12, label %bb23

bb12:                                             ; preds = %__nv_cosf.exit
  %v66 = add nuw nsw i64 %v22.i, 1
  %v67 = icmp ult i64 %v66, %v1
  br i1 %v67, label %bb13, label %bb22

bb13:                                             ; preds = %bb12
  %v64 = getelementptr inbounds nuw float, ptr %v0, i64 %v22.i
  %v65 = load float, ptr %v64, align 4
  %v69 = getelementptr inbounds nuw float, ptr %v0, i64 %v66
  %v70 = load float, ptr %v69, align 4
  %v71 = fmul contract float %z.i.i.0.i39, %v65
  %v72 = fmul contract float %z.i.i.0.i, %v70
  %v74 = getelementptr inbounds nuw float, ptr %v9, i64 %v22.i
  %v75 = fsub contract float %v71, %v72
  store float %v75, ptr %v74, align 4
  %v76 = fmul contract float %z.i.i.0.i, %v65
  %v77 = fmul contract float %z.i.i.0.i39, %v70
  %v78 = getelementptr inbounds nuw float, ptr %v9, i64 %v66
  %v79 = fadd contract float %v76, %v77
  store float %v79, ptr %v78, align 4
  br label %bb15

bb15:                                             ; preds = %bb7, %entry, %bb8, %bb13
  ret void

bb18:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb19:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb20:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb21:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb22:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb23:                                             ; preds = %__nv_cosf.exit
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @scale_f32(float %v0, ptr readonly captures(none) %v1, i64 %v2, ptr writeonly captures(address_is_null) %v3, i64 %v4) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v24 = icmp ult i64 %v22.i, %v4
  %or.cond.not = select i1 %.v18.i, i1 %v24, i1 false
  %v27 = getelementptr inbounds float, ptr %v3, i64 %v22.i
  %v381 = icmp ne ptr %v3, null
  %v38 = select i1 %or.cond.not, i1 %v381, i1 false
  br i1 %v38, label %bb2, label %bb5

bb2:                                              ; preds = %entry
  %v18 = icmp ult i64 %v22.i, %v2
  br i1 %v18, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  %v20 = getelementptr inbounds float, ptr %v1, i64 %v22.i
  %v21 = load float, ptr %v20, align 4
  %v22 = fmul contract float %v0, %v21
  store float %v22, ptr %v27, align 4
  br label %bb5

bb5:                                              ; preds = %entry, %bb3
  ret void

bb13:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @shortconv_mix(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr captures(none) %v4, i64 %v5, i32 %v6, i32 %v7, ptr writeonly captures(address_is_null) %v8, i64 %v9) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i4 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i5 = icmp eq i32 %v4.i3, 1
  %v7.i6 = icmp eq i32 %v6.i4, 1
  %v8.not.not.i = and i1 %v5.i5, %v7.i6
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i7 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i7
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v27 = zext i32 %v6 to i64
  %v28 = icmp uge i64 %v22.i, %v27
  %v30 = icmp eq i32 %v7, 0
  %or.cond = select i1 %v28, i1 true, i1 %v30
  br i1 %or.cond, label %bb22, label %bb4

bb4:                                              ; preds = %entry
  %v31 = zext i32 %v7 to i64
  %v32 = add nsw i64 %v31, -1
  %v34 = icmp ult i64 %v22.i, %v1
  br i1 %v34, label %bb5, label %bb30

bb5:                                              ; preds = %bb4
  %v36 = getelementptr inbounds nuw float, ptr %v0, i64 %v22.i
  %v37 = load float, ptr %v36, align 4
  %v38 = add nuw nsw i64 %v22.i, %v27
  %v39 = icmp ult i64 %v38, %v1
  br i1 %v39, label %bb6, label %bb31

bb6:                                              ; preds = %bb5
  %v41 = getelementptr inbounds nuw float, ptr %v0, i64 %v38
  %v42 = load float, ptr %v41, align 4
  %v43 = shl nuw nsw i64 %v27, 1
  %v44 = add nuw nsw i64 %v43, %v22.i
  %v45 = icmp ult i64 %v44, %v1
  br i1 %v45, label %bb7, label %bb32

bb7:                                              ; preds = %bb6
  %v47 = getelementptr inbounds nuw float, ptr %v0, i64 %v44
  %v48 = load float, ptr %v47, align 4
  %v49 = fmul contract float %v37, %v48
  %v50 = mul nuw i64 %v22.i, %v31
  %v51 = add nuw i64 %v50, %v32
  %v53 = icmp ult i64 %v51, %v3
  br i1 %v53, label %bb8, label %bb33

bb8:                                              ; preds = %bb7
  %v55 = getelementptr inbounds float, ptr %v2, i64 %v51
  %v56 = load float, ptr %v55, align 4
  %v57 = fmul contract float %v49, %v56
  %invariant.gep = getelementptr float, ptr %v4, i64 %v22.i
  %v60.not9.not = icmp eq i64 %v32, 0
  br i1 %v60.not9.not, label %bb12, label %bb10.preheader

bb10.preheader:                                   ; preds = %bb8
  %0 = tail call i64 @llvm.usub.sat.i64(i64 %v3, i64 %v50)
  %1 = add nsw i64 %v31, -2
  %.not.not = icmp ugt i64 %0, %1
  br i1 %.not.not, label %bb10.preheader.split, label %bb34

bb10.preheader.split:                             ; preds = %bb10.preheader
  %invariant.gep19 = getelementptr float, ptr %v2, i64 %v50
  %xtraiter = and i64 %v32, 1
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %bb10.epil.preheader, label %bb10.preheader.split.new

bb10.preheader.split.new:                         ; preds = %bb10.preheader.split
  %unroll_iter = and i64 %v32, -2
  br label %bb10

bb10:                                             ; preds = %bb10, %bb10.preheader.split.new
  %v5911 = phi i64 [ 0, %bb10.preheader.split.new ], [ %v74.1, %bb10 ]
  %v5810 = phi float [ %v57, %bb10.preheader.split.new ], [ %v73.1, %bb10 ]
  %niter = phi i64 [ 0, %bb10.preheader.split.new ], [ %niter.next.1, %bb10 ]
  %v62 = mul i64 %v5911, %v27
  %gep = getelementptr float, ptr %invariant.gep, i64 %v62
  %v66 = load float, ptr %gep, align 4
  %gep20 = getelementptr float, ptr %invariant.gep19, i64 %v5911
  %v71 = load float, ptr %gep20, align 4
  %v72 = fmul contract float %v66, %v71
  %v73 = fadd contract float %v5810, %v72
  %v74 = or disjoint i64 %v5911, 1
  %v62.1 = mul i64 %v74, %v27
  %gep.1 = getelementptr float, ptr %invariant.gep, i64 %v62.1
  %v66.1 = load float, ptr %gep.1, align 4
  %gep20.1 = getelementptr float, ptr %invariant.gep19, i64 %v74
  %v71.1 = load float, ptr %gep20.1, align 4
  %v72.1 = fmul contract float %v66.1, %v71.1
  %v73.1 = fadd contract float %v73, %v72.1
  %v74.1 = add nuw i64 %v5911, 2
  %niter.next.1 = add i64 %niter, 2
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter
  br i1 %niter.ncmp.1, label %bb12.loopexit.unr-lcssa, label %bb10

bb12.loopexit.unr-lcssa:                          ; preds = %bb10
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb12, label %bb10.epil.preheader

bb10.epil.preheader:                              ; preds = %bb12.loopexit.unr-lcssa, %bb10.preheader.split
  %v5911.epil.init = phi i64 [ 0, %bb10.preheader.split ], [ %v74.1, %bb12.loopexit.unr-lcssa ]
  %v5810.epil.init = phi float [ %v57, %bb10.preheader.split ], [ %v73.1, %bb12.loopexit.unr-lcssa ]
  %lcmp.mod22 = icmp ne i64 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod22)
  %v62.epil = mul i64 %v5911.epil.init, %v27
  %gep.epil = getelementptr float, ptr %invariant.gep, i64 %v62.epil
  %v66.epil = load float, ptr %gep.epil, align 4
  %gep20.epil = getelementptr float, ptr %invariant.gep19, i64 %v5911.epil.init
  %v71.epil = load float, ptr %gep20.epil, align 4
  %v72.epil = fmul contract float %v66.epil, %v71.epil
  %v73.epil = fadd contract float %v5810.epil.init, %v72.epil
  br label %bb12

bb12:                                             ; preds = %bb10.epil.preheader, %bb12.loopexit.unr-lcssa, %bb8
  %v58.lcssa = phi float [ %v57, %bb8 ], [ %v73.1, %bb12.loopexit.unr-lcssa ], [ %v73.epil, %bb10.epil.preheader ]
  %v97.not = icmp ult i64 %v22.i, %v9
  %v1112 = icmp ne ptr %v8, null
  %v111 = select i1 %v97.not, i1 %v1112, i1 false
  br i1 %v111, label %bb13, label %bb15

bb13:                                             ; preds = %bb12
  %v77 = fmul contract float %v42, %v58.lcssa
  %v100 = getelementptr inbounds nuw float, ptr %v8, i64 %v22.i
  store float %v77, ptr %v100, align 4
  br label %bb15

bb15:                                             ; preds = %bb12, %bb13
  br i1 %v60.not9.not, label %bb22, label %bb17.preheader

bb17.preheader:                                   ; preds = %bb15
  %xtraiter23 = and i64 %v32, 1
  %3 = icmp eq i32 %v7, 2
  br i1 %3, label %bb17.epil.preheader, label %bb17.preheader.new

bb17.preheader.new:                               ; preds = %bb17.preheader
  %unroll_iter26 = and i64 %v32, -2
  br label %bb17

bb17:                                             ; preds = %bb20.1, %bb17.preheader.new
  %v7817 = phi i64 [ 0, %bb17.preheader.new ], [ %v81.1, %bb20.1 ]
  %niter27 = phi i64 [ 0, %bb17.preheader.new ], [ %niter27.next.1, %bb20.1 ]
  %v81 = or disjoint i64 %v7817, 1
  %v82.not = icmp ult i64 %v81, %v32
  br i1 %v82.not, label %bb18, label %bb20

bb18:                                             ; preds = %bb17
  %v85 = mul nuw i64 %v81, %v27
  %gep13 = getelementptr float, ptr %invariant.gep, i64 %v85
  %v89 = load float, ptr %gep13, align 4
  br label %bb20

bb20:                                             ; preds = %bb17, %bb18
  %v90 = phi float [ %v89, %bb18 ], [ %v49, %bb17 ]
  %v91 = mul i64 %v7817, %v27
  %gep15 = getelementptr float, ptr %invariant.gep, i64 %v91
  store float %v90, ptr %gep15, align 4
  %v81.1 = add nuw i64 %v7817, 2
  %v82.not.1 = icmp ult i64 %v81.1, %v32
  br i1 %v82.not.1, label %bb18.1, label %bb20.1

bb18.1:                                           ; preds = %bb20
  %v85.1 = mul nuw i64 %v81.1, %v27
  %gep13.1 = getelementptr float, ptr %invariant.gep, i64 %v85.1
  %v89.1 = load float, ptr %gep13.1, align 4
  br label %bb20.1

bb20.1:                                           ; preds = %bb18.1, %bb20
  %v90.1 = phi float [ %v89.1, %bb18.1 ], [ %v49, %bb20 ]
  %v91.1 = mul i64 %v81, %v27
  %gep15.1 = getelementptr float, ptr %invariant.gep, i64 %v91.1
  store float %v90.1, ptr %gep15.1, align 4
  %niter27.next.1 = add i64 %niter27, 2
  %niter27.ncmp.1 = icmp eq i64 %niter27.next.1, %unroll_iter26
  br i1 %niter27.ncmp.1, label %bb22.loopexit.unr-lcssa, label %bb17

bb22.loopexit.unr-lcssa:                          ; preds = %bb20.1
  %lcmp.mod24.not = icmp eq i64 %xtraiter23, 0
  br i1 %lcmp.mod24.not, label %bb22, label %bb17.epil.preheader

bb17.epil.preheader:                              ; preds = %bb22.loopexit.unr-lcssa, %bb17.preheader
  %v7817.epil.init = phi i64 [ 0, %bb17.preheader ], [ %v81.1, %bb22.loopexit.unr-lcssa ]
  %lcmp.mod25 = icmp ne i64 %xtraiter23, 0
  tail call void @llvm.assume(i1 %lcmp.mod25)
  %v81.epil = add nuw i64 %v7817.epil.init, 1
  %v82.not.epil = icmp ult i64 %v81.epil, %v32
  br i1 %v82.not.epil, label %bb18.epil, label %bb20.epil

bb18.epil:                                        ; preds = %bb17.epil.preheader
  %v85.epil = mul nuw i64 %v81.epil, %v27
  %gep13.epil = getelementptr float, ptr %invariant.gep, i64 %v85.epil
  %v89.epil = load float, ptr %gep13.epil, align 4
  br label %bb20.epil

bb20.epil:                                        ; preds = %bb18.epil, %bb17.epil.preheader
  %v90.epil = phi float [ %v89.epil, %bb18.epil ], [ %v49, %bb17.epil.preheader ]
  %v91.epil = mul i64 %v7817.epil.init, %v27
  %gep15.epil = getelementptr float, ptr %invariant.gep, i64 %v91.epil
  store float %v90.epil, ptr %gep15.epil, align 4
  br label %bb22

bb22:                                             ; preds = %bb20.epil, %bb22.loopexit.unr-lcssa, %bb15, %entry
  ret void

bb30:                                             ; preds = %bb4
  tail call void @llvm.trap() #19
  unreachable

bb31:                                             ; preds = %bb5
  tail call void @llvm.trap() #19
  unreachable

bb32:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb34:                                             ; preds = %bb10.preheader
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @silu_gate(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, ptr writeonly captures(address_is_null) %v4, i64 %v5) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i3 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i4 = icmp eq i32 %v4.i2, 1
  %v7.i5 = icmp eq i32 %v6.i3, 1
  %v8.not.not.i = and i1 %v5.i4, %v7.i5
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i6 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i6
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v33 = icmp ult i64 %v22.i, %v5
  %or.cond.not = select i1 %.v18.i, i1 %v33, i1 false
  %v36 = getelementptr inbounds float, ptr %v4, i64 %v22.i
  %v471 = icmp ne ptr %v4, null
  %v47 = select i1 %or.cond.not, i1 %v471, i1 false
  br i1 %v47, label %bb2, label %bb6

bb2:                                              ; preds = %entry
  %v21 = icmp ult i64 %v22.i, %v1
  br i1 %v21, label %bb3, label %bb15

bb3:                                              ; preds = %bb2
  %v26 = icmp ult i64 %v22.i, %v3
  br i1 %v26, label %bb4, label %bb16

bb4:                                              ; preds = %bb3
  %v23 = getelementptr inbounds float, ptr %v0, i64 %v22.i
  %v24 = load float, ptr %v23, align 4
  %v28 = getelementptr inbounds float, ptr %v2, i64 %v22.i
  %v29 = load float, ptr %v28, align 4
  %v30 = fneg float %v24
  %0 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %0, 0
  %1 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v30, float 0x3F777313A0000000, float 5.000000e-01) #20
  %2 = tail call float @llvm.fma.f32(float %v30, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %2, float %1
  %3 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %4 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %4, float %3
  %5 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %6 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %6, float %5
  %7 = fadd float %.04.i, 0xC168000FE0000000
  %8 = fneg float %7
  %9 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v30, float 0x3FF7154760000000, float %8) #20
  %10 = tail call float @llvm.fma.f32(float %v30, float 0x3FF7154760000000, float %8)
  %.0.i = select i1 %.not.i, float %10, float %9
  %11 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v30, float 0x3E54AE0C00000000, float %.0.i) #20
  %12 = tail call float @llvm.fma.f32(float %v30, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %12, float %11
  %13 = bitcast float %.04.i to i32
  %14 = shl i32 %13, 23
  %15 = bitcast i32 %14 to float
  %16 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %17 = fmul float %16, %15
  %v52 = fadd contract float %17, 1.000000e+00
  %v53 = fdiv contract float %v24, %v52
  %v54 = fmul contract float %v29, %v53
  store float %v54, ptr %v36, align 4
  br label %bb6

bb6:                                              ; preds = %entry, %bb4
  ret void

bb15:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable

bb16:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @weighted_embedding_q6k_topk(ptr readonly captures(none) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, ptr writeonly captures(address_is_null) %v8, i64 %v9) #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #19
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #19
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #19
  %v5.i = zext nneg i32 %v2.i to i64
  %v6.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v5.i, %v6.i
  %v7.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v7.i
  %v4.i14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #19
  %v6.i15 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #19
  %v5.i16 = icmp eq i32 %v4.i14, 1
  %v7.i17 = icmp eq i32 %v6.i15, 1
  %v8.not.not.i = and i1 %v5.i16, %v7.i17
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #19
  %v14.i = icmp eq i32 %v13.i, 1
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #19
  %v16.i = icmp eq i32 %v15.i, 1
  %v17.i18 = and i1 %v14.i, %v16.i
  %.v18.i = and i1 %v8.not.not.i, %v17.i18
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1
  %v27 = zext i32 %v4 to i64
  %v28 = zext i32 %v6 to i64
  %v29 = mul nuw i64 %v28, %v27
  %v30.not = icmp ult i64 %v22.i, %v29
  br i1 %v30.not, label %bb3, label %bb42

bb3:                                              ; preds = %entry
  %v32.not = icmp eq i32 %v6, 0
  br i1 %v32.not, label %bb52, label %bb4

bb4:                                              ; preds = %bb3
  %v28.frozen = freeze i64 %v28
  %v34 = udiv i64 %v22.i, %v28.frozen
  %0 = mul i64 %v34, %v28.frozen
  %v35.decomposed = sub i64 %v22.i, %0
  %v36 = zext i32 %v5 to i64
  %v37 = mul i64 %v34, %v36
  %v41.not40.not = icmp eq i32 %v5, 0
  br i1 %v41.not40.not, label %bb12.preheader, label %bb6

bb12.preheader:                                   ; preds = %bb7, %bb4
  %v39.lcssa = phi float [ 0xC7EFFFFFE0000000, %bb4 ], [ %v39.v50, %bb7 ]
  br i1 %v41.not40.not, label %bb38, label %bb13

bb6:                                              ; preds = %bb4, %bb7
  %v4042 = phi i64 [ %v54, %bb7 ], [ 0, %bb4 ]
  %v3941 = phi float [ %v39.v50, %bb7 ], [ 0xC7EFFFFFE0000000, %bb4 ]
  %v382 = add i64 %v4042, %v37
  %v44 = shl i64 %v382, 1
  %v45 = or disjoint i64 %v44, 1
  %v47 = icmp ult i64 %v45, %v3
  br i1 %v47, label %bb7, label %bb53

bb7:                                              ; preds = %bb6
  %v49 = getelementptr inbounds float, ptr %v2, i64 %v45
  %v50 = load float, ptr %v49, align 4
  %v51.inv = fcmp ogt float %v50, %v3941
  %v39.v50 = select i1 %v51.inv, float %v50, float %v3941
  %v54 = add nuw nsw i64 %v4042, 1
  %exitcond.not = icmp eq i64 %v54, %v36
  br i1 %exitcond.not, label %bb12.preheader, label %bb6

bb13:                                             ; preds = %bb12.preheader, %bb14
  %v5645 = phi float [ %v187, %bb14 ], [ 0.000000e+00, %bb12.preheader ]
  %v5544 = phi i64 [ %v188, %bb14 ], [ 0, %bb12.preheader ]
  %v383 = add i64 %v5544, %v37
  %v60 = shl i64 %v383, 1
  %v61 = or disjoint i64 %v60, 1
  %v63 = icmp ult i64 %v61, %v3
  br i1 %v63, label %bb14, label %bb54

bb14:                                             ; preds = %bb13
  %v65 = getelementptr inbounds float, ptr %v2, i64 %v61
  %v66 = load float, ptr %v65, align 4
  %v67 = fsub contract float %v66, %v39.lcssa
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i = icmp eq i32 %1, 0
  %2 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v67, float 0x3F777313A0000000, float 5.000000e-01) #20
  %3 = tail call float @llvm.fma.f32(float %v67, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i = select i1 %.not.i, float %3, float %2
  %4 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i) #20
  %5 = tail call float @llvm.nvvm.saturate.f(float %.02.i) #20
  %.03.i = select i1 %.not.i, float %5, float %4
  %6 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %7 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i = select i1 %.not.i, float %7, float %6
  %8 = fadd float %.04.i, 0xC168000FE0000000
  %9 = fneg float %8
  %10 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v67, float 0x3FF7154760000000, float %9) #20
  %11 = tail call float @llvm.fma.f32(float %v67, float 0x3FF7154760000000, float %9)
  %.0.i = select i1 %.not.i, float %11, float %10
  %12 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v67, float 0x3E54AE0C00000000, float %.0.i) #20
  %13 = tail call float @llvm.fma.f32(float %v67, float 0x3E54AE0C00000000, float %.0.i)
  %.01.i = select i1 %.not.i, float %13, float %12
  %14 = bitcast float %.04.i to i32
  %15 = shl i32 %14, 23
  %16 = bitcast i32 %15 to float
  %17 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i)
  %18 = fmul float %17, %16
  %v187 = fadd contract float %v5645, %18
  %v188 = add nuw nsw i64 %v5544, 1
  %exitcond51.not = icmp eq i64 %v188, %v36
  br i1 %exitcond51.not, label %bb15, label %bb13

bb15:                                             ; preds = %bb14
  %v694 = lshr i64 %v28, 8
  %v715 = lshr i64 %v35.decomposed, 8
  %v87 = fcmp ule float %v187, 0.000000e+00
  %v72 = lshr i64 %v35.decomposed, 7
  %v1109 = and i64 %v72, 1
  %v112 = and i64 %v35.decomposed, 31
  %v111 = lshr i64 %v35.decomposed, 5
  %v11310 = and i64 %v111, 3
  %v114 = shl nuw nsw i64 %v1109, 6
  %v117 = shl nuw nsw i64 %v1109, 5
  %v120 = shl nuw nsw i64 %v1109, 3
  %v122 = trunc nuw nsw i64 %v11310 to i8
  %v123 = shl nuw nsw i8 %v122, 1
  %v126 = or disjoint i64 %v112, 32
  %v128 = icmp samesign ugt i64 %v11310, 1
  %v116 = or disjoint i64 %v112, %v117
  %v118 = or disjoint i64 %v116, 128
  %v16112 = lshr i64 %v112, 4
  %v163 = shl nuw nsw i64 %v11310, 1
  %v162 = or disjoint i64 %v16112, %v120
  %v119 = or disjoint i64 %v162, %v163
  %v121 = or disjoint i64 %v119, 192
  br label %bb17

bb17:                                             ; preds = %bb15, %bb37
  %v7449 = phi float [ 0.000000e+00, %bb15 ], [ %v183, %bb37 ]
  %v7348 = phi i64 [ 0, %bb15 ], [ %v184, %bb37 ]
  %v386 = add i64 %v7348, %v37
  %v78 = shl i64 %v386, 1
  %v80 = icmp ult i64 %v78, %v3
  br i1 %v80, label %bb18, label %bb55

bb18:                                             ; preds = %bb17
  %v82 = getelementptr inbounds float, ptr %v2, i64 %v78
  %v83 = load float, ptr %v82, align 4
  %v84 = tail call i32 @llvm.fptoui.sat.i32.f32(float %v83) #19
  %v85 = icmp uge i32 %v84, %v7
  %or.cond = select i1 %v85, i1 true, i1 %v87
  br i1 %or.cond, label %bb37, label %bb20

bb20:                                             ; preds = %bb18
  %v89 = zext i32 %v84 to i64
  %v90 = mul nuw nsw i64 %v694, %v89
  %reass.add = add nuw nsw i64 %v90, %v715
  %reass.mul = mul nuw i64 %reass.add, 210
  %v93 = add nuw i64 %reass.mul, 208
  %v95 = icmp ult i64 %v93, %v1
  br i1 %v95, label %bb21, label %bb56

bb21:                                             ; preds = %bb20
  %v99 = add nuw i64 %reass.mul, 209
  %v100 = icmp ult i64 %v99, %v1
  br i1 %v100, label %bb22, label %bb57

bb22:                                             ; preds = %bb21
  %v97 = getelementptr inbounds i8, ptr %v0, i64 %v93
  %v98 = load i8, ptr %v97, align 1
  %v102 = getelementptr inbounds i8, ptr %v0, i64 %v99
  %v103 = load i8, ptr %v102, align 1
  %v107 = alloca [2 x i8], align 2
  store i8 %v98, ptr %v107, align 2
  %v107.repack7 = getelementptr inbounds nuw i8, ptr %v107, i64 1
  store i8 %v103, ptr %v107.repack7, align 1
  %v108 = load i16, ptr %v107, align 2
  %v4.i25 = lshr i16 %v108, 15
  %v6.i26 = zext nneg i16 %v4.i25 to i32
  %v9.i = lshr i16 %v108, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v108, 1023
  %v13.i27 = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb22
  %v15.i28 = icmp eq i16 %v12.i, 0
  br i1 %v15.i28, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i29 = shl nuw i32 %v6.i26, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i27, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i27, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i26, 31
  %19 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %19
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb22
  %v38.i = shl nuw i32 %v6.i26, 31
  %v41.i = shl nuw nsw i32 %v13.i27, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb22
  %v44.i = shl nuw i32 %v6.i26, 31
  %20 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %20 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i27, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i29, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v115 = add nuw i64 %reass.mul, %v114
  switch i64 %v11310, label %bb26 [
    i64 0, label %bb27
    i64 2, label %bb27
  ]

bb26:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  br label %bb27

bb27:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb26
  %v127 = phi i64 [ %v126, %bb26 ], [ %v112, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v112, %cuda_kernels__oxide_kernels__f16_to_f32.exit ]
  %v137 = add nuw i64 %v127, %v115
  %v138 = icmp ult i64 %v137, %v1
  br i1 %v128, label %bb30, label %bb28

bb28:                                             ; preds = %bb27
  br i1 %v138, label %bb29, label %bb58

bb29:                                             ; preds = %bb28
  %v133 = getelementptr inbounds i8, ptr %v0, i64 %v137
  %v134 = load i8, ptr %v133, align 1
  %v135 = and i8 %v134, 15
  br label %bb32

bb30:                                             ; preds = %bb27
  br i1 %v138, label %bb31, label %bb59

bb31:                                             ; preds = %bb30
  %v140 = getelementptr inbounds i8, ptr %v0, i64 %v137
  %v141 = load i8, ptr %v140, align 1
  %v144 = lshr i8 %v141, 4
  br label %bb32

bb32:                                             ; preds = %bb31, %bb29
  %v146.in = phi i8 [ %v135, %bb29 ], [ %v144, %bb31 ]
  %v147 = add nuw i64 %v118, %reass.mul
  %v148 = icmp ult i64 %v147, %v1
  br i1 %v148, label %bb33, label %bb60

bb33:                                             ; preds = %bb32
  %v164 = add nuw i64 %v121, %reass.mul
  %v165 = icmp ult i64 %v164, %v1
  br i1 %v165, label %bb34, label %bb61

bb34:                                             ; preds = %bb33
  %v176 = or disjoint i64 %v78, 1
  %v177 = icmp ult i64 %v176, %v3
  br i1 %v177, label %bb35, label %bb62

bb35:                                             ; preds = %bb34
  %v167 = getelementptr inbounds i8, ptr %v0, i64 %v164
  %v168 = load i8, ptr %v167, align 1
  %v170 = sitofp i8 %v168 to float
  %v171 = fmul contract float %v55.i, %v170
  %v150 = getelementptr inbounds i8, ptr %v0, i64 %v147
  %v151 = load i8, ptr %v150, align 1
  %v154 = lshr i8 %v151, %v123
  %v155 = shl i8 %v154, 4
  %21 = and i8 %v155, 48
  %v15911 = or disjoint i8 %21, %v146.in
  %v159 = zext nneg i8 %v15911 to i32
  %v160 = add nsw i32 %v159, -32
  %v172 = sitofp i32 %v160 to float
  %v173 = fmul contract float %v171, %v172
  %v179 = getelementptr inbounds float, ptr %v2, i64 %v176
  %v180 = load float, ptr %v179, align 4
  %v181 = fsub contract float %v180, %v39.lcssa
  %22 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #20
  %.not.i19 = icmp eq i32 %22, 0
  %23 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v181, float 0x3F777313A0000000, float 5.000000e-01) #20
  %24 = tail call float @llvm.fma.f32(float %v181, float 0x3F777313A0000000, float 5.000000e-01)
  %.02.i20 = select i1 %.not.i19, float %24, float %23
  %25 = tail call float @llvm.nvvm.saturate.ftz.f(float %.02.i20) #20
  %26 = tail call float @llvm.nvvm.saturate.f(float %.02.i20) #20
  %.03.i21 = select i1 %.not.i19, float %26, float %25
  %27 = tail call float @llvm.nvvm.fma.rm.ftz.f(float %.03.i21, float 2.520000e+02, float 0x4168000020000000) #20
  %28 = tail call float @llvm.nvvm.fma.rm.f(float %.03.i21, float 2.520000e+02, float 0x4168000020000000) #20
  %.04.i22 = select i1 %.not.i19, float %28, float %27
  %29 = fadd float %.04.i22, 0xC168000FE0000000
  %30 = fneg float %29
  %31 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v181, float 0x3FF7154760000000, float %30) #20
  %32 = tail call float @llvm.fma.f32(float %v181, float 0x3FF7154760000000, float %30)
  %.0.i23 = select i1 %.not.i19, float %32, float %31
  %33 = tail call float @llvm.nvvm.fma.rn.ftz.f(float %v181, float 0x3E54AE0C00000000, float %.0.i23) #20
  %34 = tail call float @llvm.fma.f32(float %v181, float 0x3E54AE0C00000000, float %.0.i23)
  %.01.i24 = select i1 %.not.i19, float %34, float %33
  %35 = bitcast float %.04.i22 to i32
  %36 = shl i32 %35, 23
  %37 = bitcast i32 %36 to float
  %38 = tail call float @llvm.nvvm.ex2.approx.ftz.f32(float %.01.i24)
  %39 = fmul float %38, %37
  %v189 = fdiv contract float %39, %v187
  %v190 = fmul contract float %v173, %v189
  %v191 = fadd contract float %v7449, %v190
  br label %bb37

bb37:                                             ; preds = %bb35, %bb18
  %v183 = phi float [ %v7449, %bb18 ], [ %v191, %bb35 ]
  %v184 = add nuw nsw i64 %v7348, 1
  %exitcond52.not = icmp eq i64 %v184, %v36
  br i1 %exitcond52.not, label %bb38, label %bb17

bb38:                                             ; preds = %bb37, %bb12.preheader
  %v74.lcssa = phi float [ 0.000000e+00, %bb12.preheader ], [ %v183, %bb37 ]
  %v193 = icmp ult i64 %v22.i, %v9
  %or.cond1.not = select i1 %.v18.i, i1 %v193, i1 false
  %v20713 = icmp ne ptr %v8, null
  %v207 = select i1 %or.cond1.not, i1 %v20713, i1 false
  br i1 %v207, label %bb39, label %bb42

bb39:                                             ; preds = %bb38
  %v196 = getelementptr inbounds nuw float, ptr %v8, i64 %v22.i
  store float %v74.lcssa, ptr %v196, align 4
  br label %bb42

bb42:                                             ; preds = %bb38, %bb39, %entry
  ret void

bb52:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb53:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb54:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb55:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb56:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb57:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable

bb58:                                             ; preds = %bb28
  tail call void @llvm.trap() #19
  unreachable

bb59:                                             ; preds = %bb30
  tail call void @llvm.trap() #19
  unreachable

bb60:                                             ; preds = %bb32
  tail call void @llvm.trap() #19
  unreachable

bb61:                                             ; preds = %bb33
  tail call void @llvm.trap() #19
  unreachable

bb62:                                             ; preds = %bb34
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fptoui.sat.i32.f32(float) #7

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #9

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #10

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.round.f32(float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.fptosi.sat.i8.f32(float) #7

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32) #11

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.idp4a.s.s(i32, i32, i32) #9

; Function Attrs: convergent nounwind memory(argmem: read, inaccessiblemem: write)
define internal fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k_lane(ptr readonly captures(none) %v0, i64 %v1, i64 %v2, ptr readonly captures(none) %v3, i64 %v4, i64 %v5, i32 %v6, i64 range(i64 0, 1024) %v7) unnamed_addr #12 {
entry:
  %v21 = zext i32 %v6 to i64
  %v22.not34.not = icmp eq i32 %v6, 0
  br i1 %v22.not34.not, label %bb20, label %bb2.lr.ph

bb2.lr.ph:                                        ; preds = %entry
  %v593 = lshr i64 %v7, 4
  %v51 = add nuw nsw i64 %v7, 128
  %v54 = or disjoint i64 %v593, 192
  %v44 = add i64 %v7, %v5
  br label %bb2

bb2:                                              ; preds = %bb2.lr.ph, %bb19
  %v2036 = phi i64 [ 0, %bb2.lr.ph ], [ %v190, %bb19 ]
  %v1935 = phi float [ 0.000000e+00, %bb2.lr.ph ], [ %v188, %bb19 ]
  %v24 = mul nuw nsw i64 %v2036, 210
  %v25 = add i64 %v24, %v2
  %v26 = add i64 %v25, 208
  %v28 = icmp ult i64 %v26, %v1
  br i1 %v28, label %bb3, label %bb21

bb3:                                              ; preds = %bb2
  %v32 = add i64 %v25, 209
  %v33 = icmp ult i64 %v32, %v1
  br i1 %v33, label %bb4, label %bb22

bb4:                                              ; preds = %bb3
  %v30 = getelementptr inbounds i8, ptr %v0, i64 %v26
  %v31 = load i8, ptr %v30, align 1
  %v35 = getelementptr inbounds i8, ptr %v0, i64 %v32
  %v36 = load i8, ptr %v35, align 1
  %v40 = alloca [2 x i8], align 2
  store i8 %v31, ptr %v40, align 2
  %v40.repack1 = getelementptr inbounds nuw i8, ptr %v40, i64 1
  store i8 %v36, ptr %v40.repack1, align 1
  %v41 = load i16, ptr %v40, align 2
  %v4.i = lshr i16 %v41, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v41, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v41, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb4
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %0 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %0
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb4
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb4
  %v44.i = shl nuw i32 %v6.i, 31
  %1 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %1 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v43 = shl nuw nsw i64 %v2036, 8
  %v50 = add i64 %v25, %v7
  %v53 = add i64 %v51, %v25
  %v56 = add i64 %v54, %v25
  %v58 = add i64 %v44, %v43
  br label %bb7

bb7:                                              ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb18
  %v47.not = phi i1 [ true, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ false, %bb18 ]
  %v4633 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ 1, %bb18 ]
  %v4532 = phi float [ %v1935, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v188, %bb18 ]
  %v49 = shl nuw nsw i64 %v4633, 6
  %v55 = shl nuw nsw i64 %v4633, 3
  %v57 = shl nuw nsw i64 %v4633, 7
  %v60 = add i64 %v50, %v49
  %v61 = icmp ult i64 %v60, %v1
  br i1 %v61, label %bb8, label %bb23

bb8:                                              ; preds = %bb7
  %v52 = shl nuw nsw i64 %v4633, 5
  %v63 = getelementptr inbounds i8, ptr %v0, i64 %v60
  %v64 = load i8, ptr %v63, align 1
  %v67 = add i64 %v53, %v52
  %v68 = icmp ult i64 %v67, %v1
  br i1 %v68, label %bb9, label %bb24

bb9:                                              ; preds = %bb8
  %v65 = and i8 %v64, 15
  %v70 = getelementptr inbounds i8, ptr %v0, i64 %v67
  %v71 = load i8, ptr %v70, align 1
  %v72 = shl i8 %v71, 4
  %2 = and i8 %v72, 48
  %v764 = or disjoint i8 %2, %v65
  %v76 = zext nneg i8 %v764 to i32
  %v77 = add nsw i32 %v76, -32
  %v78 = add i64 %v60, 32
  %v79 = icmp ult i64 %v78, %v1
  br i1 %v79, label %bb10, label %bb25

bb10:                                             ; preds = %bb9
  %v81 = getelementptr inbounds i8, ptr %v0, i64 %v78
  %v82 = load i8, ptr %v81, align 1
  %v83 = and i8 %v82, 15
  %3 = shl i8 %v71, 2
  %4 = and i8 %3, 48
  %v925 = or disjoint i8 %v83, %4
  %v92 = zext nneg i8 %v925 to i32
  %v93 = add nsw i32 %v92, -32
  %v96 = lshr i8 %v64, 4
  %v101 = and i8 %v71, 48
  %v1056 = or disjoint i8 %v101, %v96
  %v105 = zext nneg i8 %v1056 to i32
  %v106 = add nsw i32 %v105, -32
  %v109 = lshr i8 %v82, 4
  %5 = lshr i8 %v71, 2
  %6 = and i8 %5, 48
  %v1187 = or disjoint i8 %v109, %6
  %v118 = zext nneg i8 %v1187 to i32
  %v119 = add nsw i32 %v118, -32
  %v120 = add i64 %v56, %v55
  %v121 = icmp ult i64 %v120, %v1
  br i1 %v121, label %bb11, label %bb26

bb11:                                             ; preds = %bb10
  %v130 = add i64 %v58, %v57
  %v132 = icmp ult i64 %v130, %v4
  br i1 %v132, label %bb12, label %bb27

bb12:                                             ; preds = %bb11
  %v123 = getelementptr inbounds i8, ptr %v0, i64 %v120
  %v124 = load i8, ptr %v123, align 1
  %v126 = sitofp i8 %v124 to float
  %v127 = fmul contract float %v55.i, %v126
  %v128 = sitofp i32 %v77 to float
  %v129 = fmul contract float %v127, %v128
  %v134 = getelementptr inbounds float, ptr %v3, i64 %v130
  %v135 = load float, ptr %v134, align 4
  %v136 = fmul contract float %v135, %v129
  %v137 = fadd contract float %v4532, %v136
  %v138 = add i64 %v120, 2
  %v139 = icmp ult i64 %v138, %v1
  br i1 %v139, label %bb13, label %bb28

bb13:                                             ; preds = %bb12
  %v148 = add i64 %v130, 32
  %v149 = icmp ult i64 %v148, %v4
  br i1 %v149, label %bb14, label %bb29

bb14:                                             ; preds = %bb13
  %v141 = getelementptr inbounds i8, ptr %v0, i64 %v138
  %v142 = load i8, ptr %v141, align 1
  %v144 = sitofp i8 %v142 to float
  %v145 = fmul contract float %v55.i, %v144
  %v146 = sitofp i32 %v93 to float
  %v147 = fmul contract float %v145, %v146
  %v151 = getelementptr inbounds float, ptr %v3, i64 %v148
  %v152 = load float, ptr %v151, align 4
  %v153 = fmul contract float %v152, %v147
  %v154 = fadd contract float %v137, %v153
  %v155 = add i64 %v120, 4
  %v156 = icmp ult i64 %v155, %v1
  br i1 %v156, label %bb15, label %bb30

bb15:                                             ; preds = %bb14
  %v165 = add i64 %v130, 64
  %v166 = icmp ult i64 %v165, %v4
  br i1 %v166, label %bb16, label %bb31

bb16:                                             ; preds = %bb15
  %v158 = getelementptr inbounds i8, ptr %v0, i64 %v155
  %v159 = load i8, ptr %v158, align 1
  %v161 = sitofp i8 %v159 to float
  %v162 = fmul contract float %v55.i, %v161
  %v163 = sitofp i32 %v106 to float
  %v164 = fmul contract float %v162, %v163
  %v168 = getelementptr inbounds float, ptr %v3, i64 %v165
  %v169 = load float, ptr %v168, align 4
  %v170 = fmul contract float %v169, %v164
  %v171 = fadd contract float %v154, %v170
  %v172 = add i64 %v120, 6
  %v173 = icmp ult i64 %v172, %v1
  br i1 %v173, label %bb17, label %bb32

bb17:                                             ; preds = %bb16
  %v182 = add i64 %v130, 96
  %v183 = icmp ult i64 %v182, %v4
  br i1 %v183, label %bb18, label %bb33

bb18:                                             ; preds = %bb17
  %v175 = getelementptr inbounds i8, ptr %v0, i64 %v172
  %v176 = load i8, ptr %v175, align 1
  %v178 = sitofp i8 %v176 to float
  %v179 = fmul contract float %v55.i, %v178
  %v180 = sitofp i32 %v119 to float
  %v181 = fmul contract float %v179, %v180
  %v185 = getelementptr inbounds float, ptr %v3, i64 %v182
  %v186 = load float, ptr %v185, align 4
  %v187 = fmul contract float %v186, %v181
  %v188 = fadd contract float %v171, %v187
  br i1 %v47.not, label %bb7, label %bb19

bb19:                                             ; preds = %bb18
  %v190 = add nuw nsw i64 %v2036, 1
  %exitcond.not = icmp eq i64 %v190, %v21
  br i1 %exitcond.not, label %bb20, label %bb2

bb20:                                             ; preds = %bb19, %entry
  %v19.lcssa = phi float [ 0.000000e+00, %entry ], [ %v188, %bb19 ]
  ret float %v19.lcssa

bb21:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable

bb22:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb23:                                             ; preds = %bb7
  tail call void @llvm.trap() #19
  unreachable

bb24:                                             ; preds = %bb8
  tail call void @llvm.trap() #19
  unreachable

bb25:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb26:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb27:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb28:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb29:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb30:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb31:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb32:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: read, inaccessiblemem: write)
define internal fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q4k(ptr readonly captures(none) %v0, i64 %v1, i64 %v2, ptr readonly captures(none) %v3, i64 %v4, i64 %v5, i32 range(i32 0, 16777216) %v6) unnamed_addr #12 {
entry:
  %v18 = alloca [8 x i8], align 4
  %v19 = alloca [8 x i8], align 4
  %v22.not81.not = icmp eq i32 %v6, 0
  br i1 %v22.not81.not, label %bb41, label %bb2.lr.ph

bb2.lr.ph:                                        ; preds = %entry
  %v120.fca.4.gep = getelementptr inbounds nuw i8, ptr %v18, i64 4
  %v121.fca.4.gep = getelementptr inbounds nuw i8, ptr %v19, i64 4
  %0 = add i64 %v2, 16
  %1 = add i64 %v5, 32
  br label %bb2

bb2:                                              ; preds = %bb2.lr.ph, %bb40
  %v2183 = phi i32 [ 0, %bb2.lr.ph ], [ %v213, %bb40 ]
  %v2082 = phi float [ 0.000000e+00, %bb2.lr.ph ], [ %v211, %bb40 ]
  %2 = zext nneg i32 %v2183 to i64
  %3 = mul nuw nsw i64 %2, 144
  %4 = add i64 %0, %3
  %5 = add i64 %v2, %3
  %6 = sub i64 -16, %5
  %7 = shl nuw nsw i64 %2, 8
  %8 = add i64 %1, %7
  %9 = add i64 %v5, %7
  %10 = sub i64 -32, %9
  %11 = add i64 %v5, %7
  %12 = add i64 %v5, %7
  %13 = sub i64 0, %12
  %v26 = add i64 %3, %v2
  %v28 = icmp ult i64 %v26, %v1
  br i1 %v28, label %bb3, label %bb42

bb3:                                              ; preds = %bb2
  %v32 = add nuw i64 %v26, 1
  %v33 = icmp ult i64 %v32, %v1
  br i1 %v33, label %bb4, label %bb43

bb4:                                              ; preds = %bb3
  %v30 = getelementptr inbounds i8, ptr %v0, i64 %v26
  %v31 = load i8, ptr %v30, align 1
  %v35 = getelementptr inbounds i8, ptr %v0, i64 %v32
  %v36 = load i8, ptr %v35, align 1
  %v40 = alloca [2 x i8], align 2
  store i8 %v31, ptr %v40, align 2
  %v40.repack1 = getelementptr inbounds nuw i8, ptr %v40, i64 1
  store i8 %v36, ptr %v40.repack1, align 1
  %v41 = load i16, ptr %v40, align 2
  %v4.i = lshr i16 %v41, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v41, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v41, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb4
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %14 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %14
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb4
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb4
  %v44.i = shl nuw i32 %v6.i, 31
  %15 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %15 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v43 = add nuw i64 %v26, 2
  %v44 = icmp ult i64 %v43, %v1
  br i1 %v44, label %bb6, label %bb44

bb6:                                              ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  %v48 = add nuw i64 %v26, 3
  %v49 = icmp ult i64 %v48, %v1
  br i1 %v49, label %bb7, label %bb45

bb7:                                              ; preds = %bb6
  %v46 = getelementptr inbounds i8, ptr %v0, i64 %v43
  %v47 = load i8, ptr %v46, align 1
  %v51 = getelementptr inbounds i8, ptr %v0, i64 %v48
  %v52 = load i8, ptr %v51, align 1
  %v56 = alloca [2 x i8], align 2
  store i8 %v47, ptr %v56, align 2
  %v56.repack3 = getelementptr inbounds nuw i8, ptr %v56, i64 1
  store i8 %v52, ptr %v56.repack3, align 1
  %v57 = load i16, ptr %v56, align 2
  %v4.i5 = lshr i16 %v57, 15
  %v6.i6 = zext nneg i16 %v4.i5 to i32
  %v9.i7 = lshr i16 %v57, 10
  %v10.i8 = and i16 %v9.i7, 31
  %v12.i9 = and i16 %v57, 1023
  %v13.i10 = zext nneg i16 %v12.i9 to i32
  switch i16 %v10.i8, label %bb10.i33 [
    i16 0, label %bb1.i18
    i16 31, label %bb9.i11
  ]

bb1.i18:                                          ; preds = %bb7
  %v15.i19 = icmp eq i16 %v12.i9, 0
  br i1 %v15.i19, label %bb2.i31, label %bb6.i20

bb2.i31:                                          ; preds = %bb1.i18
  %v17.i32 = shl nuw i32 %v6.i6, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40

bb6.i20:                                          ; preds = %bb1.i18
  %v13.masked.numleadingzeros.i21 = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i10, i1 true)
  %v13.masked.leadingonepos.i22 = xor i32 %v13.masked.numleadingzeros.i21, 31
  %bb5.tripcount.i23 = sub nuw nsw i32 10, %v13.masked.leadingonepos.i22
  %v23.i24 = shl nuw nsw i32 %v13.i10, %bb5.tripcount.i23
  %v27.i25 = shl nuw i32 %v6.i6, 31
  %16 = shl nuw nsw i32 %v13.masked.numleadingzeros.i21, 23
  %reass.sub84 = sub i32 %v27.i25, %16
  %v31.i27 = add i32 %reass.sub84, 1124073472
  %v25.i28 = shl i32 %v23.i24, 13
  %v33.i29 = and i32 %v25.i28, 8380416
  %v34.i30 = or disjoint i32 %v33.i29, %v31.i27
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40

bb9.i11:                                          ; preds = %bb7
  %v38.i12 = shl nuw i32 %v6.i6, 31
  %v41.i13 = shl nuw nsw i32 %v13.i10, 13
  %v39.i14 = or disjoint i32 %v38.i12, %v41.i13
  %v42.i15 = or disjoint i32 %v39.i14, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40

bb10.i33:                                         ; preds = %bb7
  %v44.i34 = shl nuw i32 %v6.i6, 31
  %17 = add nuw nsw i16 %v10.i8, 112
  %v46.i35 = zext nneg i16 %17 to i32
  %v48.i36 = shl nuw nsw i32 %v46.i35, 23
  %v49.i37 = or disjoint i32 %v48.i36, %v44.i34
  %v51.i38 = shl nuw nsw i32 %v13.i10, 13
  %v52.i39 = or disjoint i32 %v49.i37, %v51.i38
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit40

cuda_kernels__oxide_kernels__f16_to_f32.exit40:   ; preds = %bb2.i31, %bb6.i20, %bb9.i11, %bb10.i33
  %v54.i16 = phi i32 [ %v34.i30, %bb6.i20 ], [ %v17.i32, %bb2.i31 ], [ %v42.i15, %bb9.i11 ], [ %v52.i39, %bb10.i33 ]
  %v55.i17 = bitcast i32 %v54.i16 to float
  %v59 = add nuw i64 %v26, 4
  %v60 = icmp ult i64 %v59, %v1
  br i1 %v60, label %bb9, label %bb46

bb9:                                              ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40
  %v62 = getelementptr inbounds i8, ptr %v0, i64 %v59
  %v63 = load i8, ptr %v62, align 1
  %v64 = add nuw i64 %v26, 5
  %v65 = icmp ult i64 %v64, %v1
  br i1 %v65, label %bb10, label %bb47

bb10:                                             ; preds = %bb9
  %v67 = getelementptr inbounds i8, ptr %v0, i64 %v64
  %v68 = load i8, ptr %v67, align 1
  %v69 = add nuw i64 %v26, 6
  %v70 = icmp ult i64 %v69, %v1
  br i1 %v70, label %bb11, label %bb48

bb11:                                             ; preds = %bb10
  %v72 = getelementptr inbounds i8, ptr %v0, i64 %v69
  %v73 = load i8, ptr %v72, align 1
  %v74 = add nuw i64 %v26, 7
  %v75 = icmp ult i64 %v74, %v1
  br i1 %v75, label %bb12, label %bb49

bb12:                                             ; preds = %bb11
  %v77 = getelementptr inbounds i8, ptr %v0, i64 %v74
  %v78 = load i8, ptr %v77, align 1
  %v79 = add nuw i64 %v26, 8
  %v80 = icmp ult i64 %v79, %v1
  br i1 %v80, label %bb13, label %bb50

bb13:                                             ; preds = %bb12
  %v82 = getelementptr inbounds i8, ptr %v0, i64 %v79
  %v83 = load i8, ptr %v82, align 1
  %v84 = add nuw i64 %v26, 9
  %v85 = icmp ult i64 %v84, %v1
  br i1 %v85, label %bb14, label %bb51

bb14:                                             ; preds = %bb13
  %v87 = getelementptr inbounds i8, ptr %v0, i64 %v84
  %v88 = load i8, ptr %v87, align 1
  %v89 = add nuw i64 %v26, 10
  %v90 = icmp ult i64 %v89, %v1
  br i1 %v90, label %bb15, label %bb52

bb15:                                             ; preds = %bb14
  %v92 = getelementptr inbounds i8, ptr %v0, i64 %v89
  %v93 = load i8, ptr %v92, align 1
  %v94 = add nuw i64 %v26, 11
  %v95 = icmp ult i64 %v94, %v1
  br i1 %v95, label %bb16, label %bb53

bb16:                                             ; preds = %bb15
  %v97 = getelementptr inbounds i8, ptr %v0, i64 %v94
  %v98 = load i8, ptr %v97, align 1
  %v99 = add nuw i64 %v26, 12
  %v100 = icmp ult i64 %v99, %v1
  br i1 %v100, label %bb17, label %bb54

bb17:                                             ; preds = %bb16
  %v102 = getelementptr inbounds i8, ptr %v0, i64 %v99
  %v103 = load i8, ptr %v102, align 1
  %v104 = add nuw i64 %v26, 13
  %v105 = icmp ult i64 %v104, %v1
  br i1 %v105, label %bb18, label %bb55

bb18:                                             ; preds = %bb17
  %v107 = getelementptr inbounds i8, ptr %v0, i64 %v104
  %v108 = load i8, ptr %v107, align 1
  %v109 = add nuw i64 %v26, 14
  %v110 = icmp ult i64 %v109, %v1
  br i1 %v110, label %bb19, label %bb56

bb19:                                             ; preds = %bb18
  %v114 = add nuw i64 %v26, 15
  %v115 = icmp ult i64 %v114, %v1
  br i1 %v115, label %bb20, label %bb57

bb20:                                             ; preds = %bb19
  %v112 = getelementptr inbounds i8, ptr %v0, i64 %v109
  %v113 = load i8, ptr %v112, align 1
  %v117 = getelementptr inbounds i8, ptr %v0, i64 %v114
  %v118 = load i8, ptr %v117, align 1
  %v43.sroa.4.0.insert.ext.i = zext i8 %v78 to i32
  %v43.sroa.4.0.insert.shift.i = shl nuw i32 %v43.sroa.4.0.insert.ext.i, 24
  %v43.sroa.3.0.insert.ext.i = zext i8 %v73 to i32
  %v43.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.3.0.insert.ext.i, 16
  %v43.sroa.2.0.insert.ext.i = zext i8 %v68 to i32
  %v43.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v43.sroa.2.0.insert.ext.i, 8
  %v43.sroa.0.0.insert.ext.i = zext i8 %v63 to i32
  %v43.sroa.3.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.shift.i, %v43.sroa.0.0.insert.ext.i
  %v43.sroa.2.0.insert.insert.i = or disjoint i32 %v43.sroa.3.0.insert.insert.i, %v43.sroa.3.0.insert.shift.i
  %v43.sroa.0.0.insert.insert.i = or disjoint i32 %v43.sroa.2.0.insert.insert.i, %v43.sroa.4.0.insert.shift.i
  %v51.sroa.4.0.insert.ext.i = zext i8 %v98 to i32
  %v51.sroa.4.0.insert.shift.i = shl nuw i32 %v51.sroa.4.0.insert.ext.i, 24
  %v51.sroa.3.0.insert.ext.i = zext i8 %v93 to i32
  %v51.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.3.0.insert.ext.i, 16
  %v51.sroa.2.0.insert.ext.i = zext i8 %v88 to i32
  %v51.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v51.sroa.2.0.insert.ext.i, 8
  %v51.sroa.0.0.insert.ext.i = zext i8 %v83 to i32
  %v51.sroa.3.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.shift.i, %v51.sroa.0.0.insert.ext.i
  %v51.sroa.2.0.insert.insert.i = or disjoint i32 %v51.sroa.3.0.insert.insert.i, %v51.sroa.3.0.insert.shift.i
  %v51.sroa.0.0.insert.insert.i = or disjoint i32 %v51.sroa.2.0.insert.insert.i, %v51.sroa.4.0.insert.shift.i
  %v59.sroa.4.0.insert.ext.i = zext i8 %v118 to i32
  %v59.sroa.4.0.insert.shift.i = shl nuw i32 %v59.sroa.4.0.insert.ext.i, 24
  %v59.sroa.3.0.insert.ext.i = zext i8 %v113 to i32
  %v59.sroa.3.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.3.0.insert.ext.i, 16
  %v59.sroa.2.0.insert.ext.i = zext i8 %v108 to i32
  %v59.sroa.2.0.insert.shift.i = shl nuw nsw i32 %v59.sroa.2.0.insert.ext.i, 8
  %v59.sroa.0.0.insert.ext.i = zext i8 %v103 to i32
  %v59.sroa.3.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.shift.i, %v59.sroa.0.0.insert.ext.i
  %v59.sroa.2.0.insert.insert.i = or disjoint i32 %v59.sroa.3.0.insert.insert.i, %v59.sroa.3.0.insert.shift.i
  %v59.sroa.0.0.insert.insert.i = or disjoint i32 %v59.sroa.2.0.insert.insert.i, %v59.sroa.4.0.insert.shift.i
  %v65.i = lshr i32 %v59.sroa.0.0.insert.insert.i, 4
  %v66.i = and i32 %v65.i, 252645135
  %18 = lshr i32 %v51.sroa.0.0.insert.insert.i, 2
  %v73.i = and i32 %18, 808464432
  %v81.i = and i32 %v59.sroa.0.0.insert.insert.i, 252645135
  %19 = lshr i32 %v43.sroa.0.0.insert.insert.i, 2
  %v88.i = and i32 %19, 808464432
  %v94.i = and i32 %v43.sroa.0.0.insert.insert.i, 1061109567
  %v98.sroa.2.0.extract.shift.i = lshr i32 %v94.i, 8
  %v98.sroa.4.0.extract.shift.i = lshr i32 %v94.i, 24
  %v98.sroa.3.0.extract.shift.i = lshr i32 %v94.i, 16
  %v98.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v98.sroa.4.0.extract.shift.i to i8
  %v98.sroa.3.0.extract.trunc.i = trunc i32 %v98.sroa.3.0.extract.shift.i to i8
  %20 = insertelement <4 x i32> poison, i32 %v94.i, i64 0
  %21 = insertelement <4 x i32> %20, i32 %v98.sroa.2.0.extract.shift.i, i64 1
  %22 = trunc <4 x i32> %21 to <4 x i8>
  %23 = insertelement <4 x i8> %22, i8 %v98.sroa.3.0.extract.trunc.i, i64 2
  %24 = insertelement <4 x i8> %23, i8 %v98.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %24, ptr %v18, align 4
  %v89.i = or disjoint i32 %v81.i, %v88.i
  %v102.sroa.2.0.extract.shift.i = lshr i32 %v89.i, 8
  %v102.sroa.4.0.extract.shift.i = lshr i32 %v89.i, 24
  %v102.sroa.3.0.extract.shift.i = lshr i32 %v89.i, 16
  %v102.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v102.sroa.4.0.extract.shift.i to i8
  %v102.sroa.3.0.extract.trunc.i = trunc i32 %v102.sroa.3.0.extract.shift.i to i8
  %25 = insertelement <4 x i32> poison, i32 %v89.i, i64 0
  %26 = insertelement <4 x i32> %25, i32 %v102.sroa.2.0.extract.shift.i, i64 1
  %27 = trunc <4 x i32> %26 to <4 x i8>
  %28 = insertelement <4 x i8> %27, i8 %v102.sroa.3.0.extract.trunc.i, i64 2
  %29 = insertelement <4 x i8> %28, i8 %v102.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %29, ptr %v120.fca.4.gep, align 4
  %v78.i = and i32 %v51.sroa.0.0.insert.insert.i, 1061109567
  %v106.sroa.2.0.extract.shift.i = lshr i32 %v78.i, 8
  %v106.sroa.4.0.extract.shift.i = lshr i32 %v78.i, 24
  %v106.sroa.3.0.extract.shift.i = lshr i32 %v78.i, 16
  %v106.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v106.sroa.4.0.extract.shift.i to i8
  %v106.sroa.3.0.extract.trunc.i = trunc i32 %v106.sroa.3.0.extract.shift.i to i8
  %30 = insertelement <4 x i32> poison, i32 %v78.i, i64 0
  %31 = insertelement <4 x i32> %30, i32 %v106.sroa.2.0.extract.shift.i, i64 1
  %32 = trunc <4 x i32> %31 to <4 x i8>
  %33 = insertelement <4 x i8> %32, i8 %v106.sroa.3.0.extract.trunc.i, i64 2
  %34 = insertelement <4 x i8> %33, i8 %v106.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %34, ptr %v19, align 4
  %v74.i = or disjoint i32 %v66.i, %v73.i
  %v110.sroa.2.0.extract.shift.i = lshr i32 %v74.i, 8
  %v110.sroa.4.0.extract.shift.i = lshr i32 %v74.i, 24
  %v110.sroa.3.0.extract.shift.i = lshr i32 %v74.i, 16
  %v110.sroa.4.0.extract.trunc.i = trunc nuw nsw i32 %v110.sroa.4.0.extract.shift.i to i8
  %v110.sroa.3.0.extract.trunc.i = trunc i32 %v110.sroa.3.0.extract.shift.i to i8
  %35 = insertelement <4 x i32> poison, i32 %v74.i, i64 0
  %36 = insertelement <4 x i32> %35, i32 %v110.sroa.2.0.extract.shift.i, i64 1
  %37 = trunc <4 x i32> %36 to <4 x i8>
  %38 = insertelement <4 x i8> %37, i8 %v110.sroa.3.0.extract.trunc.i, i64 2
  %39 = insertelement <4 x i8> %38, i8 %v110.sroa.4.0.extract.trunc.i, i64 3
  store <4 x i8> %39, ptr %v121.fca.4.gep, align 4
  %invariant.gep = getelementptr float, ptr %v3, i64 %11
  br label %bb24.preheader

bb24.preheader:                                   ; preds = %bb20, %bb28
  %indvars.iv85 = phi i64 [ %13, %bb20 ], [ %indvars.iv.next86, %bb28 ]
  %indvars.iv = phi i64 [ %11, %bb20 ], [ %indvars.iv.next, %bb28 ]
  %v12675 = phi i64 [ 0, %bb20 ], [ %v150, %bb28 ]
  %v12574 = phi float [ %v2082, %bb20 ], [ %v149, %bb28 ]
  %umax = tail call i64 @llvm.umax.i64(i64 %v4, i64 %indvars.iv)
  %40 = add i64 %umax, %indvars.iv85
  %.not = icmp ult i64 %40, 32
  br i1 %.not, label %bb58, label %bb24.preheader.split

bb24.preheader.split:                             ; preds = %bb24.preheader
  %.idx = shl i64 %v12675, 7
  %gep = getelementptr i8, ptr %invariant.gep, i64 %.idx
  br label %bb25

bb25:                                             ; preds = %bb25, %bb24.preheader.split
  %v13073 = phi i64 [ 0, %bb24.preheader.split ], [ %v142.3, %bb25 ]
  %v12972 = phi float [ 0.000000e+00, %bb24.preheader.split ], [ %v141.3, %bb25 ]
  %v139 = getelementptr float, ptr %gep, i64 %v13073
  %v140 = load float, ptr %v139, align 4
  %v141 = fadd contract float %v12972, %v140
  %41 = getelementptr float, ptr %gep, i64 %v13073
  %v139.1 = getelementptr i8, ptr %41, i64 4
  %v140.1 = load float, ptr %v139.1, align 4
  %v141.1 = fadd contract float %v141, %v140.1
  %42 = getelementptr float, ptr %gep, i64 %v13073
  %v139.2 = getelementptr i8, ptr %42, i64 8
  %v140.2 = load float, ptr %v139.2, align 4
  %v141.2 = fadd contract float %v141.1, %v140.2
  %43 = getelementptr float, ptr %gep, i64 %v13073
  %v139.3 = getelementptr i8, ptr %43, i64 12
  %v140.3 = load float, ptr %v139.3, align 4
  %v141.3 = fadd contract float %v141.2, %v140.3
  %v142.3 = add nuw nsw i64 %v13073, 4
  %exitcond.3 = icmp eq i64 %v142.3, 32
  br i1 %exitcond.3, label %bb28, label %bb25

bb28:                                             ; preds = %bb25
  %v144 = getelementptr inbounds nuw i8, ptr %v19, i64 %v12675
  %v145 = load i8, ptr %v144, align 1
  %v146 = uitofp i8 %v145 to float
  %v147 = fmul contract float %v55.i17, %v146
  %v148 = fmul contract float %v141.3, %v147
  %v149 = fsub contract float %v12574, %v148
  %v150 = add nuw nsw i64 %v12675, 1
  %indvars.iv.next = add i64 %indvars.iv, 32
  %indvars.iv.next86 = add i64 %indvars.iv85, -32
  %exitcond87 = icmp eq i64 %v150, 8
  br i1 %exitcond87, label %bb29, label %bb24.preheader

bb29:                                             ; preds = %bb28
  %44 = getelementptr i8, ptr %v0, i64 %v26
  %45 = getelementptr i8, ptr %44, i64 16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb39
  %indvars.iv101 = phi i64 [ %13, %bb29 ], [ %indvars.iv.next102, %bb39 ]
  %indvars.iv98 = phi i64 [ %11, %bb29 ], [ %indvars.iv.next99, %bb39 ]
  %indvars.iv96 = phi i64 [ %10, %bb29 ], [ %indvars.iv.next97, %bb39 ]
  %indvars.iv93 = phi i64 [ %8, %bb29 ], [ %indvars.iv.next94, %bb39 ]
  %indvars.iv91 = phi i64 [ %6, %bb29 ], [ %indvars.iv.next92, %bb39 ]
  %indvars.iv88 = phi i64 [ %4, %bb29 ], [ %indvars.iv.next89, %bb39 ]
  %v15380 = phi i64 [ 0, %bb29 ], [ %v212, %bb39 ]
  %v15279 = phi float [ %v149, %bb29 ], [ %v211, %bb39 ]
  %umax90 = tail call i64 @llvm.umax.i64(i64 %v1, i64 %indvars.iv88)
  %46 = add i64 %umax90, %indvars.iv91
  %umax95 = tail call i64 @llvm.umax.i64(i64 %v4, i64 %indvars.iv93)
  %47 = add i64 %umax95, %indvars.iv96
  %umax100 = tail call i64 @llvm.umax.i64(i64 %v4, i64 %indvars.iv98)
  %48 = add i64 %umax100, %indvars.iv101
  %umin103 = tail call i64 @llvm.umin.i64(i64 %47, i64 %48)
  %umin103.fr = freeze i64 %umin103
  %umin104 = tail call i64 @llvm.umin.i64(i64 %umin103.fr, i64 %46)
  %umin105 = tail call i64 @llvm.umin.i64(i64 %umin104, i64 31)
  %v156 = shl nuw nsw i64 %v15380, 5
  %v170 = shl nuw nsw i64 %v15380, 6
  %v171 = add i64 %v170, %11
  %.not109 = icmp eq i64 %46, %umin105
  %.not111 = icmp eq i64 %47, %umin105
  br i1 %.not109, label %bb60, label %bb31.split

bb31.split:                                       ; preds = %bb31
  %.not110 = icmp eq i64 %48, %umin105
  br i1 %.not110, label %bb61, label %bb31.split.split

bb31.split.split:                                 ; preds = %bb31.split
  br i1 %.not111, label %bb62, label %bb31.split.split.split

bb31.split.split.split:                           ; preds = %bb31.split.split
  %49 = getelementptr i8, ptr %45, i64 %v156
  %invariant.gep112 = getelementptr float, ptr %v3, i64 %v171
  %50 = getelementptr float, ptr %v3, i64 %v171
  %51 = getelementptr i8, ptr %50, i64 128
  br label %bb33

bb33:                                             ; preds = %bb33, %bb31.split.split.split
  %v16078 = phi i64 [ 0, %bb31.split.split.split ], [ %v194.1, %bb33 ]
  %v15977 = phi float [ 0.000000e+00, %bb31.split.split.split ], [ %v193.1, %bb33 ]
  %v15876 = phi float [ 0.000000e+00, %bb31.split.split.split ], [ %v179.1, %bb33 ]
  %v166 = getelementptr i8, ptr %49, i64 %v16078
  %v167 = load i8, ptr %v166, align 1
  %v182 = lshr i8 %v167, 4
  %v183 = uitofp nneg i8 %v182 to float
  %gep113 = getelementptr float, ptr %invariant.gep112, i64 %v16078
  %v177 = load float, ptr %gep113, align 4
  %v168 = and i8 %v167, 15
  %v169 = uitofp nneg i8 %v168 to float
  %v178 = fmul contract float %v177, %v169
  %v179 = fadd contract float %v15876, %v178
  %v190 = getelementptr float, ptr %51, i64 %v16078
  %v191 = load float, ptr %v190, align 4
  %v192 = fmul contract float %v191, %v183
  %v193 = fadd contract float %v15977, %v192
  %v194 = or disjoint i64 %v16078, 1
  %v166.1 = getelementptr i8, ptr %49, i64 %v194
  %v167.1 = load i8, ptr %v166.1, align 1
  %v182.1 = lshr i8 %v167.1, 4
  %v183.1 = uitofp nneg i8 %v182.1 to float
  %gep113.1 = getelementptr float, ptr %invariant.gep112, i64 %v194
  %v177.1 = load float, ptr %gep113.1, align 4
  %v168.1 = and i8 %v167.1, 15
  %v169.1 = uitofp nneg i8 %v168.1 to float
  %v178.1 = fmul contract float %v177.1, %v169.1
  %v179.1 = fadd contract float %v179, %v178.1
  %v190.1 = getelementptr float, ptr %51, i64 %v194
  %v191.1 = load float, ptr %v190.1, align 4
  %v192.1 = fmul contract float %v191.1, %v183.1
  %v193.1 = fadd contract float %v193, %v192.1
  %v194.1 = add nuw nsw i64 %v16078, 2
  %exitcond106.1 = icmp eq i64 %v194.1, 32
  br i1 %exitcond106.1, label %bb39, label %bb33

bb39:                                             ; preds = %bb33
  %v195 = shl nuw nsw i64 %v15380, 1
  %v197 = getelementptr inbounds nuw i8, ptr %v18, i64 %v195
  %v198 = load i8, ptr %v197, align 2
  %v199 = uitofp i8 %v198 to float
  %v200 = fmul contract float %v55.i, %v199
  %v201 = fmul contract float %v179.1, %v200
  %v202 = fadd contract float %v15279, %v201
  %v206 = getelementptr inbounds nuw i8, ptr %v197, i64 1
  %v207 = load i8, ptr %v206, align 1
  %v208 = uitofp i8 %v207 to float
  %v209 = fmul contract float %v55.i, %v208
  %v210 = fmul contract float %v193.1, %v209
  %v211 = fadd contract float %v202, %v210
  %v212 = add nuw nsw i64 %v15380, 1
  %indvars.iv.next89 = add i64 %indvars.iv88, 32
  %indvars.iv.next92 = add i64 %indvars.iv91, -32
  %indvars.iv.next94 = add i64 %indvars.iv93, 64
  %indvars.iv.next97 = add i64 %indvars.iv96, -64
  %indvars.iv.next99 = add i64 %indvars.iv98, 64
  %indvars.iv.next102 = add i64 %indvars.iv101, -64
  %exitcond107 = icmp eq i64 %v212, 4
  br i1 %exitcond107, label %bb40, label %bb31

bb40:                                             ; preds = %bb39
  %v213 = add nuw nsw i32 %v2183, 1
  %exitcond108.not = icmp eq i32 %v213, %v6
  br i1 %exitcond108.not, label %bb41, label %bb2

bb41:                                             ; preds = %bb40, %entry
  %v20.lcssa = phi float [ 0.000000e+00, %entry ], [ %v211, %bb40 ]
  ret float %v20.lcssa

bb42:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb44:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb6
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit40
  tail call void @llvm.trap() #19
  unreachable

bb47:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb48:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb49:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb50:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb51:                                             ; preds = %bb13
  tail call void @llvm.trap() #19
  unreachable

bb52:                                             ; preds = %bb14
  tail call void @llvm.trap() #19
  unreachable

bb53:                                             ; preds = %bb15
  tail call void @llvm.trap() #19
  unreachable

bb54:                                             ; preds = %bb16
  tail call void @llvm.trap() #19
  unreachable

bb55:                                             ; preds = %bb17
  tail call void @llvm.trap() #19
  unreachable

bb56:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb57:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb58:                                             ; preds = %bb24.preheader
  tail call void @llvm.trap() #19
  unreachable

bb60:                                             ; preds = %bb31
  tail call void @llvm.trap() #19
  unreachable

bb61:                                             ; preds = %bb31.split
  tail call void @llvm.trap() #19
  unreachable

bb62:                                             ; preds = %bb31.split.split
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: read, inaccessiblemem: write)
define internal fastcc float @cuda_kernels__oxide_kernels__kernels__dot_q6k(ptr readonly captures(none) %v0, i64 %v1, i64 %v2, ptr readonly captures(none) %v3, i64 %v4, i64 %v5, i32 range(i32 0, 16777216) %v6) unnamed_addr #12 {
entry:
  %v19.not47.not = icmp eq i32 %v6, 0
  br i1 %v19.not47.not, label %bb28, label %bb2

bb2:                                              ; preds = %entry, %bb27
  %v1849 = phi i32 [ %v224, %bb27 ], [ 0, %entry ]
  %v1748 = phi float [ %v221, %bb27 ], [ 0.000000e+00, %entry ]
  %v21 = zext nneg i32 %v1849 to i64
  %v22 = mul nuw nsw i64 %v21, 210
  %v23 = add i64 %v22, %v2
  %v24 = add i64 %v23, 208
  %v26 = icmp ult i64 %v24, %v1
  br i1 %v26, label %bb3, label %bb29

bb3:                                              ; preds = %bb2
  %v30 = add i64 %v23, 209
  %v31 = icmp ult i64 %v30, %v1
  br i1 %v31, label %bb4, label %bb30

bb4:                                              ; preds = %bb3
  %v28 = getelementptr inbounds i8, ptr %v0, i64 %v24
  %v29 = load i8, ptr %v28, align 1
  %v33 = getelementptr inbounds i8, ptr %v0, i64 %v30
  %v34 = load i8, ptr %v33, align 1
  %v38 = alloca [2 x i8], align 2
  store i8 %v29, ptr %v38, align 2
  %v38.repack1 = getelementptr inbounds nuw i8, ptr %v38, i64 1
  store i8 %v34, ptr %v38.repack1, align 1
  %v39 = load i16, ptr %v38, align 2
  %v4.i = lshr i16 %v39, 15
  %v6.i = zext nneg i16 %v4.i to i32
  %v9.i = lshr i16 %v39, 10
  %v10.i = and i16 %v9.i, 31
  %v12.i = and i16 %v39, 1023
  %v13.i = zext nneg i16 %v12.i to i32
  switch i16 %v10.i, label %bb10.i [
    i16 0, label %bb1.i
    i16 31, label %bb9.i
  ]

bb1.i:                                            ; preds = %bb4
  %v15.i = icmp eq i16 %v12.i, 0
  br i1 %v15.i, label %bb2.i, label %bb6.i

bb2.i:                                            ; preds = %bb1.i
  %v17.i = shl nuw i32 %v6.i, 31
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb6.i:                                            ; preds = %bb1.i
  %v13.masked.numleadingzeros.i = tail call range(i32 22, 33) i32 @llvm.ctlz.i32(i32 %v13.i, i1 true)
  %v13.masked.leadingonepos.i = xor i32 %v13.masked.numleadingzeros.i, 31
  %bb5.tripcount.i = sub nuw nsw i32 10, %v13.masked.leadingonepos.i
  %v23.i = shl nuw nsw i32 %v13.i, %bb5.tripcount.i
  %v27.i = shl nuw i32 %v6.i, 31
  %0 = shl nuw nsw i32 %v13.masked.numleadingzeros.i, 23
  %reass.sub = sub i32 %v27.i, %0
  %v31.i = add i32 %reass.sub, 1124073472
  %v25.i = shl i32 %v23.i, 13
  %v33.i = and i32 %v25.i, 8380416
  %v34.i = or disjoint i32 %v33.i, %v31.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb9.i:                                            ; preds = %bb4
  %v38.i = shl nuw i32 %v6.i, 31
  %v41.i = shl nuw nsw i32 %v13.i, 13
  %v39.i = or disjoint i32 %v38.i, %v41.i
  %v42.i = or disjoint i32 %v39.i, 2139095040
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

bb10.i:                                           ; preds = %bb4
  %v44.i = shl nuw i32 %v6.i, 31
  %1 = add nuw nsw i16 %v10.i, 112
  %v46.i = zext nneg i16 %1 to i32
  %v48.i = shl nuw nsw i32 %v46.i, 23
  %v49.i = or disjoint i32 %v48.i, %v44.i
  %v51.i = shl nuw nsw i32 %v13.i, 13
  %v52.i = or disjoint i32 %v49.i, %v51.i
  br label %cuda_kernels__oxide_kernels__f16_to_f32.exit

cuda_kernels__oxide_kernels__f16_to_f32.exit:     ; preds = %bb2.i, %bb6.i, %bb9.i, %bb10.i
  %v54.i = phi i32 [ %v34.i, %bb6.i ], [ %v17.i, %bb2.i ], [ %v42.i, %bb9.i ], [ %v52.i, %bb10.i ]
  %v55.i = bitcast i32 %v54.i to float
  %v42 = shl nuw nsw i64 %v21, 8
  %v43 = add i64 %v42, %v5
  %v50 = add i64 %v23, 128
  %v53 = add i64 %v23, 192
  br label %bb7

bb7:                                              ; preds = %cuda_kernels__oxide_kernels__f16_to_f32.exit, %bb26
  %v46.not = phi i1 [ true, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ false, %bb26 ]
  %v4546 = phi i64 [ 0, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ 1, %bb26 ]
  %v4445 = phi float [ %v1748, %cuda_kernels__oxide_kernels__f16_to_f32.exit ], [ %v221, %bb26 ]
  %v48 = shl nuw nsw i64 %v4546, 6
  %v49 = add i64 %v48, %v23
  %v51 = shl nuw nsw i64 %v4546, 5
  %v52 = add i64 %v50, %v51
  %v54 = shl nuw nsw i64 %v4546, 3
  %v55 = add i64 %v53, %v54
  %v56 = shl nuw nsw i64 %v4546, 7
  %v57 = add i64 %v43, %v56
  br label %bb9

bb9:                                              ; preds = %bb7, %bb25
  %v5944 = phi i64 [ 0, %bb7 ], [ %v222, %bb25 ]
  %v5843 = phi float [ %v4445, %bb7 ], [ %v221, %bb25 ]
  %v623 = lshr i64 %v5944, 4
  %v63 = add nuw i64 %v49, %v5944
  %v64 = icmp ult i64 %v63, %v1
  br i1 %v64, label %bb10, label %bb31

bb10:                                             ; preds = %bb9
  %v66 = getelementptr inbounds i8, ptr %v0, i64 %v63
  %v67 = load i8, ptr %v66, align 1
  %v70 = add nuw i64 %v52, %v5944
  %v71 = icmp ult i64 %v70, %v1
  br i1 %v71, label %bb11, label %bb32

bb11:                                             ; preds = %bb10
  %v68 = and i8 %v67, 15
  %v73 = getelementptr inbounds i8, ptr %v0, i64 %v70
  %v74 = load i8, ptr %v73, align 1
  %v75 = shl i8 %v74, 4
  %2 = and i8 %v75, 48
  %v794 = or disjoint i8 %2, %v68
  %v79 = zext nneg i8 %v794 to i32
  %v80 = add nsw i32 %v79, -32
  %v82 = add i64 %v63, 32
  %v83 = icmp ult i64 %v82, %v1
  br i1 %v83, label %bb12, label %bb33

bb12:                                             ; preds = %bb11
  %v85 = getelementptr inbounds i8, ptr %v0, i64 %v82
  %v86 = load i8, ptr %v85, align 1
  %v87 = and i8 %v86, 15
  %3 = shl i8 %v74, 2
  %4 = and i8 %3, 48
  %v1015 = or disjoint i8 %v87, %4
  %v101 = zext nneg i8 %v1015 to i32
  %v102 = add nsw i32 %v101, -32
  %v110 = lshr i8 %v67, 4
  %v120 = and i8 %v74, 48
  %v1246 = or disjoint i8 %v120, %v110
  %v124 = zext nneg i8 %v1246 to i32
  %v125 = add nsw i32 %v124, -32
  %v134 = lshr i8 %v86, 4
  %5 = lshr i8 %v74, 2
  %6 = and i8 %5, 48
  %v1487 = or disjoint i8 %v134, %6
  %v148 = zext nneg i8 %v1487 to i32
  %v149 = add nsw i32 %v148, -32
  %v150 = add i64 %v55, %v623
  %v151 = icmp ult i64 %v150, %v1
  br i1 %v151, label %bb18, label %bb39

bb18:                                             ; preds = %bb12
  %v160 = add nuw i64 %v57, %v5944
  %v162 = icmp ult i64 %v160, %v4
  br i1 %v162, label %bb19, label %bb40

bb19:                                             ; preds = %bb18
  %v153 = getelementptr inbounds i8, ptr %v0, i64 %v150
  %v154 = load i8, ptr %v153, align 1
  %v156 = sitofp i8 %v154 to float
  %v157 = fmul contract float %v55.i, %v156
  %v158 = sitofp i32 %v80 to float
  %v159 = fmul contract float %v157, %v158
  %v164 = getelementptr inbounds float, ptr %v3, i64 %v160
  %v165 = load float, ptr %v164, align 4
  %v166 = fmul contract float %v165, %v159
  %v167 = fadd contract float %v5843, %v166
  %v168 = add i64 %v150, 2
  %v169 = icmp ult i64 %v168, %v1
  br i1 %v169, label %bb20, label %bb41

bb20:                                             ; preds = %bb19
  %v179 = add i64 %v160, 32
  %v180 = icmp ult i64 %v179, %v4
  br i1 %v180, label %bb21, label %bb42

bb21:                                             ; preds = %bb20
  %v171 = getelementptr inbounds i8, ptr %v0, i64 %v168
  %v172 = load i8, ptr %v171, align 1
  %v174 = sitofp i8 %v172 to float
  %v175 = fmul contract float %v55.i, %v174
  %v176 = sitofp i32 %v102 to float
  %v177 = fmul contract float %v175, %v176
  %v182 = getelementptr inbounds float, ptr %v3, i64 %v179
  %v183 = load float, ptr %v182, align 4
  %v184 = fmul contract float %v183, %v177
  %v185 = fadd contract float %v167, %v184
  %v186 = add i64 %v150, 4
  %v187 = icmp ult i64 %v186, %v1
  br i1 %v187, label %bb22, label %bb43

bb22:                                             ; preds = %bb21
  %v197 = add i64 %v160, 64
  %v198 = icmp ult i64 %v197, %v4
  br i1 %v198, label %bb23, label %bb44

bb23:                                             ; preds = %bb22
  %v189 = getelementptr inbounds i8, ptr %v0, i64 %v186
  %v190 = load i8, ptr %v189, align 1
  %v192 = sitofp i8 %v190 to float
  %v193 = fmul contract float %v55.i, %v192
  %v194 = sitofp i32 %v125 to float
  %v195 = fmul contract float %v193, %v194
  %v200 = getelementptr inbounds float, ptr %v3, i64 %v197
  %v201 = load float, ptr %v200, align 4
  %v202 = fmul contract float %v201, %v195
  %v203 = fadd contract float %v185, %v202
  %v204 = add i64 %v150, 6
  %v205 = icmp ult i64 %v204, %v1
  br i1 %v205, label %bb24, label %bb45

bb24:                                             ; preds = %bb23
  %v215 = add i64 %v160, 96
  %v216 = icmp ult i64 %v215, %v4
  br i1 %v216, label %bb25, label %bb46

bb25:                                             ; preds = %bb24
  %v207 = getelementptr inbounds i8, ptr %v0, i64 %v204
  %v208 = load i8, ptr %v207, align 1
  %v210 = sitofp i8 %v208 to float
  %v211 = fmul contract float %v55.i, %v210
  %v212 = sitofp i32 %v149 to float
  %v213 = fmul contract float %v211, %v212
  %v218 = getelementptr inbounds float, ptr %v3, i64 %v215
  %v219 = load float, ptr %v218, align 4
  %v220 = fmul contract float %v219, %v213
  %v221 = fadd contract float %v203, %v220
  %v222 = add nuw nsw i64 %v5944, 1
  %exitcond = icmp eq i64 %v222, 32
  br i1 %exitcond, label %bb26, label %bb9

bb26:                                             ; preds = %bb25
  br i1 %v46.not, label %bb7, label %bb27

bb27:                                             ; preds = %bb26
  %v224 = add nuw nsw i32 %v1849, 1
  %exitcond50.not = icmp eq i32 %v224, %v6
  br i1 %exitcond50.not, label %bb28, label %bb2

bb28:                                             ; preds = %bb27, %entry
  %v17.lcssa = phi float [ 0.000000e+00, %entry ], [ %v221, %bb27 ]
  ret float %v17.lcssa

bb29:                                             ; preds = %bb2
  tail call void @llvm.trap() #19
  unreachable

bb30:                                             ; preds = %bb3
  tail call void @llvm.trap() #19
  unreachable

bb31:                                             ; preds = %bb9
  tail call void @llvm.trap() #19
  unreachable

bb32:                                             ; preds = %bb10
  tail call void @llvm.trap() #19
  unreachable

bb33:                                             ; preds = %bb11
  tail call void @llvm.trap() #19
  unreachable

bb39:                                             ; preds = %bb12
  tail call void @llvm.trap() #19
  unreachable

bb40:                                             ; preds = %bb18
  tail call void @llvm.trap() #19
  unreachable

bb41:                                             ; preds = %bb19
  tail call void @llvm.trap() #19
  unreachable

bb42:                                             ; preds = %bb20
  tail call void @llvm.trap() #19
  unreachable

bb43:                                             ; preds = %bb21
  tail call void @llvm.trap() #19
  unreachable

bb44:                                             ; preds = %bb22
  tail call void @llvm.trap() #19
  unreachable

bb45:                                             ; preds = %bb23
  tail call void @llvm.trap() #19
  unreachable

bb46:                                             ; preds = %bb24
  tail call void @llvm.trap() #19
  unreachable
}

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #11

; Function Attrs: nofree nosync nounwind memory(none)
declare noundef i32 @__nvvm_reflect(ptr noundef) local_unnamed_addr #13

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #14

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #14

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #14

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #14

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.f2i.rn.ftz(float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.nvvm.f2i.rn(float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.ftz.f32(float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fabs.f32(float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.mul.rn.ftz.f(float, float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.mul.rn.f(float, float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.saturate.ftz.f(float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.saturate.f(float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rm.ftz.f(float, float, float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.nvvm.fma.rm.f(float, float, float) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.ex2.approx.ftz.f32(float) #14

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #15

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #15

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #16

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #16

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #15

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #15

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #17

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #15

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #15

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #18

attributes #0 = { convergent nounwind memory(argmem: readwrite, inaccessiblemem: write) }
attributes #1 = { convergent norecurse nounwind }
attributes #2 = { convergent nounwind memory(read, argmem: readwrite, inaccessiblemem: write, target_mem0: none, target_mem1: none) }
attributes #3 = { convergent nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #4 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) }
attributes #5 = { convergent nofree norecurse nosync nounwind memory(argmem: readwrite) }
attributes #6 = { convergent nounwind }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #9 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { convergent nocallback nounwind }
attributes #11 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #12 = { convergent nounwind memory(argmem: read, inaccessiblemem: write) }
attributes #13 = { nofree nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #14 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #15 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #16 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #17 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #18 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #19 = { convergent }
attributes #20 = { nounwind }
attributes #21 = { nounwind memory(none) }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.disable"}
!4 = distinct !{!4, !3}
!5 = distinct !{!5, !3}
!6 = distinct !{!6, !3}
!7 = distinct !{!7, !3}
!8 = distinct !{!8, !3}
!9 = distinct !{!9, !3}
!10 = distinct !{!10, !3}
!11 = distinct !{!11, !3}
!12 = distinct !{!12, !3}
!13 = distinct !{!13, !3}
!14 = !{i32 33119, i32 33123, i32 33168, i32 33213}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.unroll.count", i32 1}
