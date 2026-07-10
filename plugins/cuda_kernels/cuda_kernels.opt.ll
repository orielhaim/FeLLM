; ModuleID = '/mnt/c/Users/oriel/Documents/Projects/FeLLM/plugins/cuda_kernels/cuda_kernels.ll'
source_filename = "cuda_kernels"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @scale_f32(float %v0, ptr readonly captures(none) %v1, i64 %v2, ptr writeonly captures(none) %v3, i64 %v4) local_unnamed_addr #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v16.not = icmp ult i64 %v9.i, %v4
  %v20 = icmp ult i64 %v9.i, %v2
  br i1 %v16.not, label %bb2, label %bb5

bb2:                                              ; preds = %entry
  %v22 = getelementptr inbounds nuw float, ptr %v1, i64 %v9.i
  %v26 = getelementptr inbounds nuw float, ptr %v3, i64 %v9.i
  tail call void @llvm.assume(i1 %v20)
  %v23 = load float, ptr %v22, align 4
  %v24 = fmul contract float %v0, %v23
  store float %v24, ptr %v26, align 4
  br label %bb5

bb5:                                              ; preds = %entry, %bb2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define range(i64 0, 2199023254528) i64 @cuda_device____internal__index_1d(ptr readnone captures(none) %v0) local_unnamed_addr #2 {
entry:
  %v2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
  %v3 = zext nneg i32 %v2 to i64
  %v4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4
  %v5 = zext nneg i32 %v4 to i64
  %v6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4
  %v7 = zext nneg i32 %v6 to i64
  %v8 = mul nuw nsw i64 %v5, %v7
  %v9 = add nuw nsw i64 %v8, %v3
  ret i64 %v9
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #3

attributes #0 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: write) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #4 = { convergent }
