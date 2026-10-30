"""Experimental reduced-traffic variant; original state-copy lifetime retained."""
import exact_gdn_reduce_v2 as e
from segmented_graph_engine import Engine,FIELDS,chunk_device_zero

def recurrent(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False):
    out,state=e.fused(query,key,value,g,beta,initial_state,True,use_qk_l2norm_in_kernel)
    if initial_state is not None and query.shape[1]==1:
        initial_state.copy_(state);state=initial_state
    return out,state if output_final_state else None

def enable_graph(model):
    for layer in model.model.language_model.layers:
        if hasattr(layer,'linear_attn'):
            layer.linear_attn.recurrent_gated_delta_rule=recurrent
            layer.linear_attn.chunk_gated_delta_rule=chunk_device_zero
