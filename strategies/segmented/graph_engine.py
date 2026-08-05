"""Qwen3.5 hybrid-cache native-prefix unrolled decode CUDA graph.

Only unpadded text, positive contiguous decode positions, fixed batch/L/N.
No attention, RoPE, normalization, or reduction math changes. Fresh prefill and
all state copies are paid on every request. Graph startup is excluded/logged.
"""
import contextlib,time,torch
import exact_gdn
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

FIELDS=('key_cache','value_cache','conv_states','recurrent_states')

def inplace_recurrent(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False):
    out,state=exact_gdn.fused(query,key,value,g,beta,initial_state,True,use_qk_l2norm_in_kernel)
    if initial_state is not None and query.shape[1]==1:
        initial_state.copy_(state);state=initial_state
    return out,state if output_final_state else None

def enable_graph(model):
    for layer in model.model.language_model.layers:
        if hasattr(layer,'linear_attn'):layer.linear_attn.recurrent_gated_delta_rule=inplace_recurrent

class CopyList(list):
    def __init__(self,targets):super().__init__([None]*len(targets));self.targets=targets
    def __setitem__(self,i,value):
        target=self.targets[i]
        if target is not None and value is not None:
            assert target.shape==value.shape and target.dtype==value.dtype
            target.copy_(value);value=target
        super().__setitem__(i,value)

class SeedCache(m.Qwen3_5DynamicCache):
    def __init__(self,cfg,targets):
        super().__init__(cfg)
        for f in FIELDS:setattr(self,f,CopyList(targets[f]))

class Engine:
    def __init__(self,model,cfg,eos_mask,x,n=128,keep_logits=False):
        self.model=model;self.cfg=cfg;self.eos_mask=eos_mask
        self.b,self.l=x.shape;self.n=n;self.keep_logits=keep_logits
        self.out=torch.empty(self.b,self.l+n,device=x.device,dtype=torch.long)
        self.mask=torch.ones_like(self.out)
        self.finite=torch.empty(n,device=x.device,dtype=torch.bool)
        self.raw=[]
        self.positions=[torch.tensor([self.l+i-1],device=x.device) for i in range(1,n)]
        self.pps=[torch.full((4,self.b,1),self.l+i-1,device=x.device,dtype=torch.long) for i in range(1,n)]
        self.prepare_masks(x.device)
        # Warm all exact shapes without retaining an old cache during capture.
        cache,raw=self.prefill(x)
        self.seed(x,raw)
        with self.decode_guard():self.decode(cache,False)
        del cache,raw
        torch.cuda.synchronize()
        cache,raw=self.prefill(x)
        self.initial={f:list(getattr(cache,f)) for f in FIELDS}
        self.seed(x,raw)
        self.cache=cache
        self.graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with self.decode_guard(),torch.cuda.graph(self.graph,stream=stream):
            self.decode(cache,keep_logits)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    @contextlib.contextmanager
    def decode_guard(self):
        lm=self.model.model.language_model;original=lm._update_linear_attn_mask
        # This branch's result is already known on host for our strictly
        # positive one-token cached decode; avoid its CUDA scalar bool sync.
        def known_cached(attention_mask,cache_position):
            assert cache_position.numel()==1 and attention_mask.ndim in (2,4)
            return None
        lm._update_linear_attn_mask=known_cached
        try:yield
        finally:lm._update_linear_attn_mask=original

    def prefill(self,x):
        cache=SeedCache(self.cfg,self.initial) if hasattr(self,'initial') else m.Qwen3_5DynamicCache(self.cfg)
        mask=torch.ones_like(x)
        pp=self.model._prepare_position_ids_for_generation(x,{'attention_mask':mask,'past_key_values':cache})
        pos=torch.arange(self.l,device=x.device)
        inp=self.model.prepare_inputs_for_generation(x,past_key_values=cache,attention_mask=mask,cache_position=pos,position_ids=pp,use_cache=True,logits_to_keep=1,is_first_iteration=True)
        r=self.model(**inp)
        return cache,r.logits[:,-1]

    def seed(self,x,raw):
        self.out[:,:self.l].copy_(x)
        self.finite[0].copy_(torch.isfinite(raw).all())
        self.out[:,self.l].copy_(torch.where(self.eos_mask,-float('inf'),raw.float()).argmax(-1))

    def prepare_masks(self,device):
        # Same official eager-mask factory, exact prefix length per step.
        # All inputs are unpadded; these masks depend only on fixed shape.
        self.decode_masks=[m.create_causal_mask(config=self.cfg,inputs_embeds=torch.empty(self.b,1,1,device=device,dtype=self.model.dtype),attention_mask=self.mask[:,:self.l+i],cache_position=self.positions[i-1],past_key_values=None,position_ids=self.pps[i-1][0]) for i in range(1,self.n)]

    def decode(self,cache,keep_logits):
        for i in range(1,self.n):
            inp=self.model.prepare_inputs_for_generation(self.out[:,self.l+i-1:self.l+i],past_key_values=cache,attention_mask=self.decode_masks[i-1],cache_position=self.positions[i-1],position_ids=self.pps[i-1],use_cache=True,logits_to_keep=1,is_first_iteration=False)
            r=self.model(**inp);raw=r.logits[:,-1]
            self.finite[i].copy_(torch.isfinite(raw).all())
            self.out[:,self.l+i].copy_(torch.where(self.eos_mask,-float('inf'),raw.float()).argmax(-1))
            if keep_logits:self.raw.append(raw.clone())

    def __call__(self,x):
        assert tuple(x.shape)==(self.b,self.l)
        cache,raw=self.prefill(x)
        if self.keep_logits:self.first_raw=raw.clone()
        self.seed(x,raw)
        # SeedCache performs fresh native prefill copies one layer at a time,
        # rather than retaining two whole 27B recurrent caches concurrently.
        del cache
        self.graph.replay()
        return self.out
