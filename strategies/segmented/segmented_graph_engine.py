"""Entire native prefill plus unrolled decode capture; fixed unpadded text.

Each replay recomputes prefix state from captured zero-initialized cache math.
No request-to-request KV/recurrent-state reuse. A/B/A correctness probes are
mandatory in the study, in addition to all full generation checks.
"""
import contextlib,torch,time
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from graph_engine import Engine as DecodeEngine,enable_graph as enable_decode_graph,FIELDS

def chunk_device_zero(query,key,value,g,beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,chunk_size=64):
    # Native Torch constructs this exact FP32 zero on CPU and copies to CUDA.
    # Allocate the identical +0 tensor on device to avoid capture-time H2D sync.
    if initial_state is None:
        initial_state=torch.zeros(query.shape[0],key.shape[2],key.shape[-1],value.shape[-1],device=query.device,dtype=torch.float32)
    return m.torch_chunk_gated_delta_rule(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

def enable_graph(model):
    enable_decode_graph(model)
    for layer in model.model.language_model.layers:
        if hasattr(layer,'linear_attn'):layer.linear_attn.chunk_gated_delta_rule=chunk_device_zero

class Engine(DecodeEngine):
    def __init__(self,model,cfg,eos_mask,x,n=128,keep_logits=False):
        self.model=model;self.cfg=cfg;self.eos_mask=eos_mask
        self.b,self.l=x.shape;self.n=n;self.keep_logits=keep_logits
        self.input=x.clone()
        self.out=torch.empty(self.b,self.l+n,device=x.device,dtype=torch.long)
        self.mask=torch.ones_like(self.out)
        self.finite=torch.empty(n,device=x.device,dtype=torch.bool)
        self.positions=[torch.tensor([self.l+i-1],device=x.device) for i in range(1,n)]
        self.pps=[torch.full((4,self.b,1),self.l+i-1,device=x.device,dtype=torch.long) for i in range(1,n)]
        self.pos0=torch.arange(self.l,device=x.device)
        self.pp0=self.model._prepare_position_ids_for_generation(self.input,{'attention_mask':self.mask[:,:self.l],'past_key_values':m.Qwen3_5DynamicCache(cfg)})
        self.prepare_masks(x.device)
        self.prefill_mask=m.create_causal_mask(config=cfg,inputs_embeds=torch.empty(self.b,self.l,1,device=x.device,dtype=model.dtype),attention_mask=self.mask[:,:self.l],cache_position=self.pos0,past_key_values=None,position_ids=self.pp0[0])
        self.raw=[]
        with self.decode_guard():self.run_all(False)
        del self.cache
        torch.cuda.synchronize()
        self.graphs=[];pool=torch.cuda.graph_pool_handle();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with self.decode_guard():
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=pool,stream=stream):self.prefix(keep_logits)
            self.graphs.append(graph);print('GRAPH_CAPTURE_PREFIX_DONE',self.b,time.time(),flush=True)
            for begin in range(1,n,8):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=pool,stream=stream):
                    for i in range(begin,min(begin+8,n)):self.step(self.cache,i,keep_logits)
                self.graphs.append(graph);print('GRAPH_CAPTURE_SEGMENT_DONE',self.b,begin,time.time(),flush=True)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()

    @contextlib.contextmanager
    def decode_guard(self):
        lm=self.model.model.language_model;original=lm._update_linear_attn_mask
        def known_unpadded(attention_mask,cache_position):
            assert attention_mask.ndim in (2,4) and cache_position.numel() in (1,self.l)
            return None
        lm._update_linear_attn_mask=known_unpadded
        try:yield
        finally:lm._update_linear_attn_mask=original

    def prefix(self,keep_logits):
        cache=m.Qwen3_5DynamicCache(self.cfg)
        inp=self.model.prepare_inputs_for_generation(self.input,past_key_values=cache,attention_mask=self.prefill_mask,cache_position=self.pos0,position_ids=self.pp0,use_cache=True,logits_to_keep=1,is_first_iteration=True)
        r=self.model(**inp);raw=r.logits[:,-1]
        self.seed(self.input,raw)
        if keep_logits:self.first_raw=raw.clone()
        self.cache=cache

    def step(self,cache,i,keep_logits):
        inp=self.model.prepare_inputs_for_generation(self.out[:,self.l+i-1:self.l+i],past_key_values=cache,attention_mask=self.decode_masks[i-1],cache_position=self.positions[i-1],position_ids=self.pps[i-1],use_cache=True,logits_to_keep=1,is_first_iteration=False)
        r=self.model(**inp);raw=r.logits[:,-1]
        self.finite[i].copy_(torch.isfinite(raw).all())
        self.out[:,self.l+i].copy_(torch.where(self.eos_mask,-float('inf'),raw.float()).argmax(-1))
        if keep_logits:self.raw.append(raw.clone())

    def run_all(self,keep_logits):
        self.prefix(keep_logits)
        for i in range(1,self.n):self.step(self.cache,i,keep_logits)

    def __call__(self,x):
        assert tuple(x.shape)==(self.b,self.l)
        self.input.copy_(x)
        for graph in self.graphs:graph.replay()
        return self.out
