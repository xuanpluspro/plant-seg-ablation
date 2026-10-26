# Backbone pretrained weights

Default file: pretrained/resnet50.pth (official torchvision ResNet-50,
ImageNet-1K V2). Run `python download_pretrained.py` from the project.
The script verifies the SHA256 prefix published in torchvision's download URL,
reuses a verified file, and does not overwrite a mismatched existing file.
Binary weights and unfinished downloads are ignored by Git.

This file initializes only the ResNet-50 backbone. MPM sparse attention, DRM/CBM, dynamic fusion and
segmentation heads are trained on your plant masks; it is not a full
segmentation checkpoint. Other compatible local torchvision RGB ResNet-50
weights can be selected explicitly with --backbone-weights /path/to/file.pth.


Optional full U-Net checkpoint: pretrained/unet_resnet_voc.pth.
Select with --backbone-weights pretrained/unet_resnet_voc.pth.
The loader extracts resnet.* (or backbone.*) only, supports module. prefixes
and common state-dict wrappers, and strictly validates every backbone key
and shape before loading. Decoder tensors and fc classifier tensors are
ignored. This does not replace the segmentation architecture. The default
resnet50.pth is unchanged. Upload binary weights yourself; do not commit them.


Before training: python check_pretrained.py --weights pretrained/unet_resnet_voc.pth --device cuda
Success prints CHECK PASSED. Default batch1, size512, width256; add
--batch-size 32 to exercise the intended training batch on an idle GPU.
The check validates loading/numerics and synthetic train/eval execution,
without an optimizer step or any checkpoint write.
