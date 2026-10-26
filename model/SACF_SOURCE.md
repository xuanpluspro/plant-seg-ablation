# SACF branch adaptation

DRM/CBM reference: https://github.com/Pollob001/HyperSeg-DG
Pinned revision: 4bad7bce0184a9c2367d3724f9cc05c9e68a93ac
Source: network/model.py, LightweightDynamicRelationModule and LightweightContextBridge.
Attribution: Pollob001/HyperSeg-DG contributors. Cite HyperSeg-DG when reporting this adaptation.

DRM retains ordinary 3x3 dilated convolutions (d=1,2), per-image two-scale
softmax gating from the original input, concatenation/1x1 fusion and a
zero-initialized learnable residual scale. CBM retains DWConv/PWConv/GELU,
channel gating of the original input, local+global addition and projected
residual. The attention bottleneck uses max(1,C//8) to avoid zero channels;
this is identical at the project's supported widths C>=8.

This adapts only the two submodules into SACF's two different-resolution
inputs: DRM on deep features before upsampling; CBM on shallow features.
No complete HFCB, mode branches, outer input BN, averaging or outer gamma
is imported. The existing MODA-derived spatial dual-sigmoid fusion and
Conv/BN/ReLU projection are preserved, including align_corners=True.
This is not a reproduction of the full HyperSeg-DG model or its paper equations.
The original MODA licensing is retained in LICENSE.MODA; attribution here
is not a new license grant for third-party code.
