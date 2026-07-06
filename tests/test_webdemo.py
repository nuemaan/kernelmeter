"""The web demo embeds a copy of the gpu database. This keeps the copy
honest: if gpus.py changes and docs/index.html isn't regenerated, ci fails."""

import json
import pathlib
import re

import pytest

from kernelmeter import gpus

DOCS = pathlib.Path(__file__).parent.parent / "docs" / "index.html"


def _embedded_db():
    page = DOCS.read_text()
    match = re.search(
        r'<script type="application/json" id="db">(.*?)</script>', page, re.S
    )
    assert match, "db json block missing from docs/index.html"
    return {entry["id"]: entry for entry in json.loads(match.group(1))}


def test_web_db_matches_python_db():
    web = _embedded_db()
    assert set(web) == {spec.id for spec in gpus.DATABASE}
    for spec in gpus.DATABASE:
        p = spec.peaks()
        entry = web[spec.id]
        assert entry["name"] == spec.name
        assert entry["theoretical_mem_bandwidth_gb_s"] == pytest.approx(
            p.mem_bandwidth_gbs, rel=1e-6
        )
        assert entry["theoretical_fp32_tflops"] == pytest.approx(p.fp32_tflops, rel=1e-6)
        if p.fp16_tensor_tflops is None:
            assert entry["theoretical_fp16_tensor_tflops"] is None
        else:
            assert entry["theoretical_fp16_tensor_tflops"] == pytest.approx(
                p.fp16_tensor_tflops, rel=1e-6
            )


def test_page_links_back_to_repo():
    page = DOCS.read_text()
    assert "github.com/nuemaan/kernelmeter" in page
    assert "uvx kernelmeter" in page


def test_web_quants_match_python_quants():
    from kernelmeter import llm

    page = DOCS.read_text()
    match = re.search(r"const QUANTS = (\{[^}]*\});", page)
    assert match, "QUANTS object missing from docs/index.html"
    web_quants = json.loads(match.group(1))
    assert web_quants == llm.QUANTS


def test_web_db_carries_vram():
    from kernelmeter import gpus

    web = _embedded_db()
    for spec in gpus.DATABASE:
        assert web[spec.id]["vram_gb"] == spec.vram_gb


def test_llm_tab_present():
    page = DOCS.read_text()
    assert 'id="llmView"' in page
    assert 'id="tabLlm"' in page
    assert "roofline ceilings, not predictions" in page
