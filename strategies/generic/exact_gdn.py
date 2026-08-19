"""Decode-only partial fusion, retaining native normalization and reduction order.

No changed approximation or accumulation dtype. Unsupported cases use HF.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

BLOCK = 1024
WARPS = 4
COUNTS={"fused_row_major":0,"fused_col_major":0,"fused_permuted":0,"fallback":0}
LAYOUTS={}
NATIVE = m.torch_recurrent_gated_delta_rule

@triton.jit
def first(H, K, G, D, P, N:tl.constexpr, KV:tl.constexpr, V:tl.constexpr, KD:tl.constexpr, NH:tl.constexpr, BS:tl.constexpr, SB:tl.constexpr, SH:tl.constexpr, SK:tl.constexpr, SV:tl.constexpr, BK:tl.constexpr):
    i=tl.program_id(0)*BK+tl.arange(0,BK)
    h=tl.load(H+i,i<N,0)
    bh=(i//SB%BS)*NH+(i//SH%NH)
    g=tl.load(G+bh,i<N,0)
    ki=bh*KD+(i//SK%KD)
    k=tl.load(K+ki,i<N,0)
    d=libdevice.mul_rn(h,g)
    p=libdevice.mul_rn(d,k)
    tl.store(D+i,d,i<N);tl.store(P+i,p,i<N)

@triton.jit
def second(D, K, Q, VV, B, MEM, H, P, N:tl.constexpr, KV:tl.constexpr, V:tl.constexpr, KD:tl.constexpr, NH:tl.constexpr, BS:tl.constexpr, SB:tl.constexpr, SH:tl.constexpr, SK:tl.constexpr, SV:tl.constexpr, BK:tl.constexpr):
    i=tl.program_id(0)*BK+tl.arange(0,BK)
    bh=(i//SB%BS)*NH+(i//SH%NH)
    hv=bh*V+(i//SV%V)
    d=tl.load(D+i,i<N,0)
    ki=bh*KD+(i//SK%KD)
    k=tl.load(K+ki,i<N,0)
    q=tl.load(Q+ki,i<N,0)
    v=tl.load(VV+hv,i<N,0)
    b=tl.load(B+bh,i<N,0)
    mem=tl.load(MEM+hv,i<N,0)
    delta=libdevice.mul_rn(libdevice.sub_rn(v,mem),b)
    h=libdevice.add_rn(d,libdevice.mul_rn(k,delta))
    p=libdevice.mul_rn(h,q)
    tl.store(H+i,h,i<N);tl.store(P+i,p,i<N)

def fused(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False):
    if query.shape[1]!=1 or initial_state is None:
        COUNTS["fallback"]+=1
        return NATIVE(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel)
    original=(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel)
    dtype=query.dtype
    if use_qk_l2norm_in_kernel:
        query=m.l2norm(query,dim=-1,eps=1e-6)
        key=m.l2norm(key,dim=-1,eps=1e-6)
    query,key,value,beta,g=[x.transpose(1,2).contiguous().to(torch.float32).contiguous() for x in (query,key,value,beta,g)]
    query=query*(1/(query.shape[-1]**0.5))
    bs,nh,_,kd=key.shape;vd=value.shape[-1]
    state=initial_state.to(value)
    layout=str(tuple(state.stride()));LAYOUTS[layout]=LAYOUTS.get(layout,0)+1
    # Cover every dense non-overlapping axis permutation, not only K/V transpose.
    dense=True;expect=1
    for stride,size in sorted((st,sz) for st,sz in zip(state.stride(),state.shape) if sz>1):
        if stride!=expect:dense=False
        expect*=size
    if not dense:
        COUNTS['fallback']+=1
        return NATIVE(*original)
    category='fused_row_major' if state.is_contiguous() else ('fused_col_major' if state.stride(-2)==1 else 'fused_permuted')
    COUNTS[category]+=1
    sb,sh,sk,sv=state.stride()
    dec=torch.empty_like(state);prod=torch.empty_like(state)
    gd=g[:,:,0].exp().contiguous()
    first[(triton.cdiv(state.numel(),BLOCK),)](state,key,gd,dec,prod,state.numel(),kd*vd,vd,kd,nh,bs,sb,sh,sk,sv,BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    mem=prod.sum(dim=-2).contiguous()
    outstate=torch.empty_like(state)
    second[(triton.cdiv(state.numel(),BLOCK),)](dec,key,query,value,beta,mem,outstate,prod,state.numel(),kd*vd,vd,kd,nh,bs,sb,sh,sk,sv,BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    out=prod.sum(dim=-2).unsqueeze(1).contiguous().to(dtype)
    return out,outstate if output_final_state else None

def configure(model,enabled):
    for layer in model.model.language_model.layers:
        if hasattr(layer,'linear_attn'):
            layer.linear_attn.recurrent_gated_delta_rule=fused if enabled else NATIVE
