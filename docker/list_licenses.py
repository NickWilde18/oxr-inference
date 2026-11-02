"""列出镜像里装的每个 Python 包和它的许可证，输出 Markdown（构建镜像时写到 /licenses/python-packages.md）。

许可证取包元数据：先看 ``License-Expression``，没有就看一行以内的 ``License`` 字段，再没有就看 ``License ::`` 分类器。
许可证原文是各包自带的文件（文件名以 LICENSE、LICENCE、COPYING、NOTICE 开头，或放在 dist-info 的 ``licenses/`` 下），
表里写出它们在镜像里的路径；包里没带原文的，标出来。
"""

from __future__ import annotations

import importlib.metadata as metadata

_MARKERS = ("LICENSE", "LICENCE", "COPYING", "NOTICE")


def _license(md: metadata.PackageMetadata) -> str:
    if md.get("License-Expression"):
        return md["License-Expression"]
    text = (md.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 80:
        return text
    classifiers = [c.removeprefix("License :: ").removeprefix("OSI Approved :: ")
                   for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    return "; ".join(classifiers) or "not stated in the metadata"


def _files(dist: metadata.Distribution) -> list[str]:
    found = []
    for f in dist.files or []:
        parts = f.parts
        in_licenses_dir = any(p.endswith(".dist-info") for p in parts) and "licenses" in parts
        if in_licenses_dir or f.name.upper().startswith(_MARKERS):
            found.append(str(dist.locate_file(f)))
    return found


def main() -> None:
    dists = sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
    print("# Python packages in this image\n")
    print("Generated from package metadata by `docker/list_licenses.py` when the image was built. "
          "Each package's own license text prevails.\n")
    print("| Package | Version | License | License text |")
    print("|---|---|---|---|")
    for dist in dists:
        md = dist.metadata
        files = _files(dist)
        where = "<br>".join(f"`{f}`" for f in files) if files else "not included in the package"
        print(f"| {md['Name']} | {md['Version']} | {_license(md)} | {where} |")


if __name__ == "__main__":
    main()
