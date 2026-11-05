"""Read-only post-training directional-attention summaries for frozen R4 comparisons."""
from pathlib import Path
import json
import numpy as np
import torch
from .train_main import evaluate_multistep
from .artifacts import write_json_atomic

def summarize_attention(attention, alignment):
    a = np.asarray(attention, dtype=np.float64)
    d = np.asarray(alignment, dtype=np.float64)
    if a.shape != d.shape or a.ndim != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError('Attention/alignment need matching B,N,N arrays')
    if not np.isfinite(a).all() or not np.isfinite(d).all() or (a < 0).any():
        raise ValueError('Nonfinite or negative actual attention')
    if not np.allclose(a.sum(-1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError('Attention rows are not actual probabilities')
    up, down = d > .5, d < -.5
    au, ad = (a * up).sum(-1), (a * down).sum(-1)
    denominator = au + ad
    dsi = np.divide(au, denominator, out=np.full_like(au, np.nan), where=denominator > 0)
    entropy = -(a * np.log(np.maximum(a, 1e-300))).sum(-1) / np.log(a.shape[-1])
    return dict(dsi=dsi, normalized_entropy=entropy, maximum_weight=a.max(-1),
                attention_up_mass=au, attention_down_mass=ad,
                up_neighbor_count=up.sum(-1), down_neighbor_count=down.sum(-1))

class AttentionRecorder:
    def __init__(self, model):
        self.model, self.parts = model, []
    def hook(self, graph, args, output):
        weights = graph.last_attn_weights
        alignment = getattr(graph, 'last_direction_alignment', None)
        if alignment is None:
            # The original direction-off control still has a physically defined
            # direction reference, although it did not add its bias to attention.
            rel, wind = args[4], args[5]
            if wind.ndim == 2:
                wind = torch.nn.functional.normalize(wind[:, None, None, :], dim=-1)
            else:
                unit = torch.nn.functional.normalize(wind, dim=-1)
                wind = torch.nn.functional.normalize(unit[:, :, None, :] + unit[:, None, :, :], dim=-1)
            alignment = (wind * torch.nn.functional.normalize(rel, dim=-1)[None]).sum(-1)
        if weights is None:
            raise ValueError('Actual graph attention was not retained')
        self.parts.append(summarize_attention(weights.detach().cpu().numpy(), alignment.detach().cpu().numpy()))
    def __enter__(self):
        self.handle = self.model.graph.register_forward_hook(self.hook)
        return self
    def __exit__(self, *args):
        self.handle.remove()
    def save(self, directory, bundle, split):
        if not self.parts:
            raise ValueError('No actual post-training attention batches')
        values = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        n = len(values['dsi']); metadata = bundle.handler.metadata[split]
        if n != len(metadata['timestamp']):
            raise ValueError('Attention cannot be reported from an incomplete evaluation split')
        values.update(turbine_ids=np.asarray(bundle.handler.turbine_ids, dtype=str),
                      timestamp=np.asarray(metadata['timestamp'], dtype=str),
                      origin_timestamp=np.asarray(metadata['origin_timestamp'], dtype=str))
        path = Path(directory) / (split + '_attention_diagnostics.npz')
        np.savez_compressed(path, **values)
        valid = np.isfinite(values['dsi'])
        summary = dict(split=split, sample_count=n, turbine_count=len(bundle.handler.turbine_ids),
                       dsi_mean=float(values['dsi'][valid].mean()) if valid.any() else None,
                       dsi_supported_query_count=int(valid.sum()), dsi_undefined_query_count=int((~valid).sum()),
                       normalized_entropy_mean=float(values['normalized_entropy'].mean()),
                       maximum_weight_mean=float(values['maximum_weight'].mean()),
                       definition='DSI=A_up/(A_up+A_down), alignment>0.5/<-0.5; rel[i,j]=r_i-r_j; positive flow alignment means j upstream of i',
                       entropy_definition='All actual N graph neighbors including self, normalized by log(N)',
                       weighting='Equal query/sample/turbine; undefined DSI separately counted',
                       no_significance_test=True, path=str(path))
        write_json_atomic(Path(directory) / (split + '_attention_summary.json'), summary)
        return summary

def evaluate_and_record(model, loader, device, bundle, split, directory, *, max_test_batches=None):
    with AttentionRecorder(model) as recorder:
        result = evaluate_multistep(model, loader, device, bundle.capacity_mw, max_test_batches=max_test_batches)
    if max_test_batches is None:
        recorder.save(directory, bundle, split)
    return result
