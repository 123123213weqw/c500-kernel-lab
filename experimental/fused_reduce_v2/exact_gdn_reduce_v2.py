"""Exact-order row-major recurrence for the observed pinned MetaX reduction dispatch.

Observed torch sum:8 row lanes, serial16 additions/lane, tree offsets4,2,1.
Native normalization/casts retained; no mutation of input state.
"""
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
BLOCK=32
WARPS=4
COUNTS={"fused_row_major":0,"fallback":0}
LAYOUTS={}
NATIVE=m.torch_recurrent_gated_delta_rule

@triton.jit
def memory(H,K,G,MEM, BV:tl.constexpr):
    bh=tl.program_id(0)
    v=tl.program_id(1)*BV+tl.arange(0,BV)
    g=tl.load(G+bh)
    a0=tl.full((BV,),0,tl.float32)
    a1=tl.full((BV,),0,tl.float32)
    a2=tl.full((BV,),0,tl.float32)
    a3=tl.full((BV,),0,tl.float32)
    a4=tl.full((BV,),0,tl.float32)
    a5=tl.full((BV,),0,tl.float32)
    a6=tl.full((BV,),0,tl.float32)
    a7=tl.full((BV,),0,tl.float32)
    for j in tl.static_range(16):
        k0=j*8+0
        hv0=tl.load(H+bh*16384+k0*128+v)
        kv0=tl.load(K+bh*128+k0)
        d0=libdevice.mul_rn(hv0,g)
        p0=libdevice.mul_rn(d0,kv0)
        a0=libdevice.add_rn(a0,p0)
        k1=j*8+1
        hv1=tl.load(H+bh*16384+k1*128+v)
        kv1=tl.load(K+bh*128+k1)
        d1=libdevice.mul_rn(hv1,g)
        p1=libdevice.mul_rn(d1,kv1)
        a1=libdevice.add_rn(a1,p1)
        k2=j*8+2
        hv2=tl.load(H+bh*16384+k2*128+v)
        kv2=tl.load(K+bh*128+k2)
        d2=libdevice.mul_rn(hv2,g)
        p2=libdevice.mul_rn(d2,kv2)
        a2=libdevice.add_rn(a2,p2)
        k3=j*8+3
        hv3=tl.load(H+bh*16384+k3*128+v)
        kv3=tl.load(K+bh*128+k3)
        d3=libdevice.mul_rn(hv3,g)
        p3=libdevice.mul_rn(d3,kv3)
        a3=libdevice.add_rn(a3,p3)
        k4=j*8+4
        hv4=tl.load(H+bh*16384+k4*128+v)
        kv4=tl.load(K+bh*128+k4)
        d4=libdevice.mul_rn(hv4,g)
        p4=libdevice.mul_rn(d4,kv4)
        a4=libdevice.add_rn(a4,p4)
        k5=j*8+5
        hv5=tl.load(H+bh*16384+k5*128+v)
        kv5=tl.load(K+bh*128+k5)
        d5=libdevice.mul_rn(hv5,g)
        p5=libdevice.mul_rn(d5,kv5)
        a5=libdevice.add_rn(a5,p5)
        k6=j*8+6
        hv6=tl.load(H+bh*16384+k6*128+v)
        kv6=tl.load(K+bh*128+k6)
        d6=libdevice.mul_rn(hv6,g)
        p6=libdevice.mul_rn(d6,kv6)
        a6=libdevice.add_rn(a6,p6)
        k7=j*8+7
        hv7=tl.load(H+bh*16384+k7*128+v)
        kv7=tl.load(K+bh*128+k7)
        d7=libdevice.mul_rn(hv7,g)
        p7=libdevice.mul_rn(d7,kv7)
        a7=libdevice.add_rn(a7,p7)
    total=libdevice.add_rn(libdevice.add_rn(libdevice.add_rn(a0,a4),libdevice.add_rn(a2,a6)),libdevice.add_rn(libdevice.add_rn(a1,a5),libdevice.add_rn(a3,a7)))
    tl.store(MEM+bh*128+v,total)

@triton.jit
def state_out(H,K,Q,VV,B,G,MEM,HO,OUT, BV:tl.constexpr):
    bh=tl.program_id(0)
    v=tl.program_id(1)*BV+tl.arange(0,BV)
    g=tl.load(G+bh)
    vv=tl.load(VV+bh*128+v)
    mem=tl.load(MEM+bh*128+v)
    beta=tl.load(B+bh)
    delta=libdevice.mul_rn(libdevice.sub_rn(vv,mem),beta)
    a0=tl.full((BV,),0,tl.float32)
    a1=tl.full((BV,),0,tl.float32)
    a2=tl.full((BV,),0,tl.float32)
    a3=tl.full((BV,),0,tl.float32)
    a4=tl.full((BV,),0,tl.float32)
    a5=tl.full((BV,),0,tl.float32)
    a6=tl.full((BV,),0,tl.float32)
    a7=tl.full((BV,),0,tl.float32)
    for j in tl.static_range(16):
        k0=j*8+0
        hv0=tl.load(H+bh*16384+k0*128+v)
        kv0=tl.load(K+bh*128+k0)
        d0=libdevice.mul_rn(hv0,g)
        hh0=libdevice.add_rn(d0,libdevice.mul_rn(kv0,delta))
        tl.store(HO+bh*16384+k0*128+v,hh0)
        q0=tl.load(Q+bh*128+k0)
        p0=libdevice.mul_rn(hh0,q0)
        a0=libdevice.add_rn(a0,p0)
        k1=j*8+1
        hv1=tl.load(H+bh*16384+k1*128+v)
        kv1=tl.load(K+bh*128+k1)
        d1=libdevice.mul_rn(hv1,g)
        hh1=libdevice.add_rn(d1,libdevice.mul_rn(kv1,delta))
        tl.store(HO+bh*16384+k1*128+v,hh1)
        q1=tl.load(Q+bh*128+k1)
        p1=libdevice.mul_rn(hh1,q1)
        a1=libdevice.add_rn(a1,p1)
        k2=j*8+2
        hv2=tl.load(H+bh*16384+k2*128+v)
        kv2=tl.load(K+bh*128+k2)
        d2=libdevice.mul_rn(hv2,g)
        hh2=libdevice.add_rn(d2,libdevice.mul_rn(kv2,delta))
        tl.store(HO+bh*16384+k2*128+v,hh2)
        q2=tl.load(Q+bh*128+k2)
        p2=libdevice.mul_rn(hh2,q2)
        a2=libdevice.add_rn(a2,p2)
        k3=j*8+3
        hv3=tl.load(H+bh*16384+k3*128+v)
        kv3=tl.load(K+bh*128+k3)
        d3=libdevice.mul_rn(hv3,g)
        hh3=libdevice.add_rn(d3,libdevice.mul_rn(kv3,delta))
        tl.store(HO+bh*16384+k3*128+v,hh3)
        q3=tl.load(Q+bh*128+k3)
        p3=libdevice.mul_rn(hh3,q3)
        a3=libdevice.add_rn(a3,p3)
        k4=j*8+4
        hv4=tl.load(H+bh*16384+k4*128+v)
        kv4=tl.load(K+bh*128+k4)
        d4=libdevice.mul_rn(hv4,g)
        hh4=libdevice.add_rn(d4,libdevice.mul_rn(kv4,delta))
        tl.store(HO+bh*16384+k4*128+v,hh4)
        q4=tl.load(Q+bh*128+k4)
        p4=libdevice.mul_rn(hh4,q4)
        a4=libdevice.add_rn(a4,p4)
        k5=j*8+5
        hv5=tl.load(H+bh*16384+k5*128+v)
        kv5=tl.load(K+bh*128+k5)
        d5=libdevice.mul_rn(hv5,g)
        hh5=libdevice.add_rn(d5,libdevice.mul_rn(kv5,delta))
        tl.store(HO+bh*16384+k5*128+v,hh5)
        q5=tl.load(Q+bh*128+k5)
        p5=libdevice.mul_rn(hh5,q5)
        a5=libdevice.add_rn(a5,p5)
        k6=j*8+6
        hv6=tl.load(H+bh*16384+k6*128+v)
        kv6=tl.load(K+bh*128+k6)
        d6=libdevice.mul_rn(hv6,g)
        hh6=libdevice.add_rn(d6,libdevice.mul_rn(kv6,delta))
        tl.store(HO+bh*16384+k6*128+v,hh6)
        q6=tl.load(Q+bh*128+k6)
        p6=libdevice.mul_rn(hh6,q6)
        a6=libdevice.add_rn(a6,p6)
        k7=j*8+7
        hv7=tl.load(H+bh*16384+k7*128+v)
        kv7=tl.load(K+bh*128+k7)
        d7=libdevice.mul_rn(hv7,g)
        hh7=libdevice.add_rn(d7,libdevice.mul_rn(kv7,delta))
        tl.store(HO+bh*16384+k7*128+v,hh7)
        q7=tl.load(Q+bh*128+k7)
        p7=libdevice.mul_rn(hh7,q7)
        a7=libdevice.add_rn(a7,p7)
    total=libdevice.add_rn(libdevice.add_rn(libdevice.add_rn(a0,a4),libdevice.add_rn(a2,a6)),libdevice.add_rn(libdevice.add_rn(a1,a5),libdevice.add_rn(a3,a7)))
    tl.store(OUT+bh*128+v,total)

def fused(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False):
    if query.shape[1]!=1 or initial_state is None or not initial_state.is_contiguous() or key.shape[-1]!=128 or value.shape[-1]!=128:
        COUNTS['fallback']+=1
        return NATIVE(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel)
    dtype=query.dtype
    if use_qk_l2norm_in_kernel:
        query=m.l2norm(query,dim=-1,eps=1e-6)
        key=m.l2norm(key,dim=-1,eps=1e-6)
    query,key,value,beta,g=[x.transpose(1,2).contiguous().to(torch.float32).contiguous() for x in (query,key,value,beta,g)]
    query=query*(1/(query.shape[-1]**0.5))
    state=initial_state.to(value);bs,nh,_,kd=key.shape
    COUNTS['fused_row_major']+=1
    layout=str(tuple(state.stride()));LAYOUTS[layout]=LAYOUTS.get(layout,0)+1
    gd=g[:,:,0].exp().contiguous()
    mem=torch.empty((bs,nh,128),device=value.device,dtype=torch.float32)
    outstate=torch.empty_like(state)
    out=torch.empty((bs,nh,128),device=value.device,dtype=torch.float32)
    grid=(bs*nh,triton.cdiv(128,BLOCK))
    memory[grid](state,key,gd,mem,BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    state_out[grid](state,key,query,value,beta,gd,mem,outstate,out,BLOCK,num_warps=WARPS,enable_fp_fusion=False)
    return out.unsqueeze(1).contiguous().to(dtype),outstate if output_final_state else None

def configure(model,enabled):
    for layer in model.model.language_model.layers:
        if hasattr(layer,'linear_attn'):
            layer.linear_attn.recurrent_gated_delta_rule=fused if enabled else NATIVE
