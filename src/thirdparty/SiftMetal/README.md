# SiftMetal provenance

This directory contains a locally modified port of [SIFTMetal by Luke Van In](https://github.com/lukevanin/SIFTMetal). The Swift host implementation was rewritten in Objective-C++ for COLMAP; the Metal shaders have also been modified locally.

[LICENSE](LICENSE) preserves the upstream MIT notice, copyright 2023 Luke Van In, verbatim from [upstream commit 4abdeb8cd92b3bc3afe72643360cc45ea0c50cc1](https://github.com/lukevanin/SIFTMetal/blob/4abdeb8cd92b3bc3afe72643360cc45ea0c50cc1/LICENSE). Include this notice when distributing the derived code or compiled Metal library.

The local import commit `bf01a458b958fbe31fcb67643c44e873e6ec2dd0` credits SIFTMetal but does not record an exact upstream source revision. The pinned revision above identifies the verified license source; it does not assert that the current directory is an unchanged copy of that revision. COLMAP's root license does not replace this donor notice.
