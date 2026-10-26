# Local datasets

Put datasets here on AutoDL; images and masks are excluded from Git.
Example: data/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation}.
Lists are train.txt, val.txt and optional test.txt, with disjoint image IDs.
Use --data-path data/VOC2012; images and masks must be 512x512.

Masks: 0 background; binary 0/255 (--mask-encoding binary255), binary 0/1
(--mask-encoding index), or class IDs 0..K-1 (--mask-encoding index).
--num-classes excludes background: use 1 for binary or 4 for IDs 0..4.
Default auto recognizes binary 0/1 or 0/255. Palette PNG indices are preserved.
RGB visualization colors require conversion to indices. No ignored label is supported.
