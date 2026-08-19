"""Direct state-write graph variant: eliminate separate state-buffer copy."""
import exact_gdn_inplace as e
from segmented_graph_engine import Engine,FIELDS,chunk_device_zero

def recurrent(query,key,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel=False):
    out,state=e.fused(query,key,value,g,beta,initial_state,True,use_qk_l2norm_in_kernel,mutate_state=True)
    if initial_state is not None and state.data_ptr()!=initial_state.data_ptr():
        initial_state.copy_(state);state=initial_state
    return out,state if output_final_state else None

def enable_graph(model):
    for layer in model.model.language_model.layers:
        if hasattr(layer,'linear_attn'):
            layer.linear_attn.recurrent_gated_delta_rule=recurrent
            layer.linear_attn.chunk_gated_delta_rule=chunk_device_zero
