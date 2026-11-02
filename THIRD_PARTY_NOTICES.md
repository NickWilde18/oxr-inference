# Third-party software and licenses

This repository contains only this project's own code, licensed under [LICENSE](LICENSE) (Apache 2.0 with OXR's
additional terms). Images built with this repository's Dockerfile also contain the following; the license texts are in
the image's `/licenses` and in the files each component ships.

| Component | Location in the image | License | License text |
|---|---|---|---|
| This project | `/opt/runtime` | Apache 2.0 + OXR additional terms | `/licenses/oxr-inference/` |
| [OXR](https://github.com/SoMarkAI/OXR) source (SoMark Team, pinned at commit `c1a1dcb`) | `/opt/oxr` | Apache 2.0 + OXR additional terms. The adapted XY-Cut++ and D-FINE parts are Apache 2.0; KaTeX (MIT), the KaTeX fonts (OFL 1.1), highlight.js (BSD 3-Clause) and others used by the demo page are listed in OXR's NOTICE | `/licenses/OXR-LICENSE`, `/licenses/OXR-NOTICE`, `/licenses/layout-analysis-THIRD_PARTY_NOTICES.md` |
| OXR-1.0 weights (including the layout-detection weights), downloaded from ModelScope or Hugging Face at build time | `/models/OXR-1.0` | Same as the OXR source (the model repository's LICENSE is identical to the source's) | `/licenses/OXR-1.0-weights-LICENSE` |
| Python packages: PyTorch, torchvision, Triton, flash-attn 4, FastAPI, OpenCV, pypdfium2 and others (versions in `docker/requirements.lock`) | `/usr/local/lib/python3.12/site-packages` | Mostly open-source licenses: BSD, MIT, Apache 2.0, MPL 2.0, ISC, PSF | Listed one by one in `/licenses/python-packages.md` (generated from package metadata at build time); the texts are in each package's own files |
| NVIDIA CUDA 13 runtime libraries: cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, NCCL, NVSHMEM and others (installed with PyTorch) | `site-packages/nvidia` | NVIDIA proprietary licenses (the CUDA EULA and each library's own license agreement) | `License.txt` in each package's dist-info |
| NVIDIA CUTLASS DSL (`nvidia-cutlass-dsl*`, used by flash-attn 4 to compile kernels) | `site-packages` | NVIDIA Software License Agreement (proprietary, see below) | `LICENSE` in each package's dist-info |
| Python 3.12 and the Debian 12 base system (`python:3.12-slim-bookworm`), plus gcc, libc6-dev, libgl1 and libglib2.0-0 | system directories | PSF, GPL, LGPL, MIT and others | `/usr/local/lib/python3.12/LICENSE.txt`, `/usr/share/doc/*/copyright` |

## Things to note

- **OXR's additional terms** apply to OXR and its derivative works, including this project. In short: if you and your
  affiliates, on a consolidated basis, exceed USD 2,000,000 in total funding or annual revenue, commercial use needs a
  commercial license from SoMark Team first; when providing an online service to third parties, you must prominently
  identify SoMark OXR in its user interface or public documentation; you may not offer third parties a hosted API
  service that competes with SoMark's (document parsing, OCR, layout analysis, content extraction), while use solely
  within your own organization isn't restricted by this term. The original text in [LICENSE](LICENSE) prevails.
- **NVIDIA CUTLASS DSL**: its license (the NVIDIA Software License Agreement shipped with the package) allows
  installing and using it; it permits redistributing only the Python source files in the package, as part of an
  application (Section 1.1(d)), and no other part (Section 2.2); employees and contractors of your organization — and,
  for educational institutions, enrolled students — may use it from your secure network (Section 3). The image
  contains its binary libraries (about 460 MB of `.so` files in `nvidia-cutlass-dsl-libs-*`). The public images of
  vLLM and SGLang install the same package, and so does ours; review NVIDIA's terms before using or redistributing the
  image.
- **opencv-python** bundles FFmpeg, Qt and other libraries, see `site-packages/cv2/LICENSE-3RD-PARTY.txt`;
  **pypdfium2** bundles PDFium and its dependencies, with the licenses under `licenses/` in its dist-info.
- The tokenizers package ships no license text (its metadata says Apache 2.0); the Apache 2.0 terms are in this
  project's LICENSE.
