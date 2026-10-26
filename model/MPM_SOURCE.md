# MPM-5531 source and adaptation

Upstream: https://github.com/Wangtao-Bao/MPCNet
Pinned commit: 5024ea51e8abdf7bc40e36489041143c468b1fdf
Source file: MPCNet.py
Attribution: Wangtao-Bao/MPCNet contributors.

The autopad, activation, CBR, h_sigmoid, h_swish and CoordAttiton helpers
retain upstream behavior. The MPM is adapted to fixed 5x5, 5x5, 3x3, 1x1
ordinary convolution branches with BN/ReLU. Two 5x5 branches share one
convolution and BN. Branch results concatenate with the original input,
then CBR fuses 2C to C before coordinate attention. The upstream CBR's
trainable 7x7 depthwise activation and coordinate attention's h-swish and
max(8,C//32) bottleneck remain. The fusion output is now also used for Q and for an output residual; CAM
feeds LayerNorm followed by an independent KV projection. See below.

Only required helpers are included; no thop dependency. This is a derived
module, not a new original module or a reproduction of the full network.
No upstream LICENSE file was present at the pinned revision; this document
does not assign this project's Apache license to the upstream code.


## MSCN sparse-attention adaptation

Upstream: https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/MSCN
Pinned commit: d4c5aa895b8c45164534bbd393911061683b1ff4
Source: MSCN/network/MSCN.py, class MSC.
Attribution: MSCN authors and TangXu-Group contributors.
Upstream Apache-2.0 license retained in ../LICENSE.MSCN.

The sparse-attention implementation in mpm.py is adapted and rewritten from
MSCN: default eight heads, two Top-k branches, scaled dot products, learned
unconstrained mixing scalars initialized to 0.5, and output projection.
The divisors are explicitly 2 and 6 (the upstream README training command),
rather than the upstream class defaults 2 and 3. Linear Q/KV/output maps
are implemented as mathematically equivalent pointwise Conv2d maps.
KV has LayerNorm BEFORE projection, following the code; the paper figure
and equation place LN after projection, which is a different operation.

Deliberate changes: no MSCN multiscale pooling (MPM already supplies 5531);
Q comes from fused Fm and KV from coordinate-attention output; residual
adds Fm after output projection. No attention/output dropout. Supports
rectangular maps, guarantees at least one retained key, uses boolean masks
and FP32 correlations/aggregation. No output residual existed in active
upstream MSC code. Dense Top-k masking does not provide sparse-kernel speedups.
Only the deepest projected feature is processed; DRM/CBM and dynamic fusion
are unchanged. This is not a reproduction of the full MSCN classifier.
