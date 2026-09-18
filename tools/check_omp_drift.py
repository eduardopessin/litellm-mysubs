"""Verifica as âncoras `# omp:` contra o pacote real do OMP.

O wiring é do ``@oh-my-pi/pi-ai``; este pacote é uma porta para Python. Sem verificação, um
rename do lado deles deixa as âncoras a apontar para nada e ninguém dá por isso até
alguém tentar seguir uma.

    python tools/check_omp_drift.py            # verifica a versão fixada
    python tools/check_omp_drift.py --update   # só reporta a versão mais recente

Sai com 1 se algum símbolo anotado tiver desaparecido. Uma versão nova no npm é aviso, não
erro: actualizar é uma decisão, não uma obrigação.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Final, NamedTuple

#: Versão do OMP contra a qual as âncoras foram escritas.
OMP_VERSION = "18.2.6"

#: O wiring está repartido por dois pacotes: `pi-ai` tem a lógica, `pi-catalog` tem as
#: constantes de fio (valores de headers, versões de cliente fixadas). Uma âncora pode
#: apontar para qualquer um deles.
PACKAGES: Final = ("@oh-my-pi/pi-ai", "@oh-my-pi/pi-catalog")
REGISTRY = "https://registry.npmjs.org"

#: `# omp: providers/anthropic.ts :: symbolA, symbolB`
ANCHOR_RE = re.compile(r"^\s*#\s*omp:\s*(?P<file>[\w./-]+)\s*::\s*(?P<symbols>[\w.,\s]+?)\s*$")

SRC = Path(__file__).resolve().parent.parent / "src"


class Anchor(NamedTuple):
    path: Path
    line: int
    file: str
    symbol: str

    def __str__(self) -> str:
        return f"{self.path.name}:{self.line} -> {self.file} :: {self.symbol}"


def collect_anchors() -> list[Anchor]:
    """Uma âncora por símbolo: várias no mesmo comentário contam separadamente."""
    anchors: list[Anchor] = []
    for source in sorted(SRC.rglob("*.py")):
        for number, text in enumerate(source.read_text("utf-8").splitlines(), start=1):
            match = ANCHOR_RE.match(text)
            if match is None:
                continue
            file = match.group("file")
            for symbol in match.group("symbols").split(","):
                if stripped := symbol.strip():
                    anchors.append(Anchor(source, number, file, stripped))
    return anchors


def fetch_sources(version: str) -> dict[str, str]:
    """Ficheiros ``src/`` dos tarballs, indexados por caminho relativo.

    Os dois pacotes partilham o espaço de nomes: um caminho só existe num deles.
    """
    sources: dict[str, str] = {}
    for package in PACKAGES:
        sources.update(_fetch_package_sources(package, version))
    return sources


def _fetch_package_sources(package: str, version: str) -> dict[str, str]:
    name = package.split("/")[-1]
    url = f"{REGISTRY}/{package}/-/{name}-{version}.tgz"
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()

    sources: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or "/src/" not in member.name:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            relative = member.name.split("/src/", 1)[1]
            sources[relative] = handle.read().decode("utf-8", "replace")
    return sources


def latest_version() -> str:
    with urllib.request.urlopen(f"{REGISTRY}/{PACKAGES[0]}", timeout=60) as response:
        return str(json.load(response)["dist-tags"]["latest"])


def main() -> int:
    latest = latest_version()
    if "--update" in sys.argv:
        print(f"fixada: {OMP_VERSION}\nmais recente: {latest}")
        return 0

    anchors = collect_anchors()
    if not anchors:
        print("nenhuma âncora `# omp:` encontrada")
        return 0

    sources = fetch_sources(OMP_VERSION)
    names = " + ".join(p.split("/")[-1] for p in PACKAGES)
    print(f"{names}@{OMP_VERSION}: {len(sources)} ficheiros, {len(anchors)} âncoras\n")

    broken: list[tuple[Anchor, str]] = []
    for anchor in anchors:
        content = sources.get(anchor.file)
        if content is None:
            broken.append((anchor, "ficheiro inexistente"))
        elif anchor.symbol not in content:
            broken.append((anchor, "símbolo ausente"))
        else:
            print(f"  ok    {anchor}")

    for anchor, reason in broken:
        print(f"  FALHA {anchor}  ({reason})")

    if latest != OMP_VERSION:
        print(f"\naviso: {PACKAGES[0]}@{latest} disponível (fixada: {OMP_VERSION})")

    if broken:
        print(f"\n{len(broken)} âncora(s) sem correspondência")
        return 1
    print(f"\n{len(anchors)} âncoras verificadas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
