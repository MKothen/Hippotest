
from src.data.hippocampome import HippocampomeCache, load_priors


def test_priors_fallback(tmp_path, monkeypatch):
    cache = HippocampomeCache(enabled=True, json_endpoint=None, cache_filename="test_priors.json")
    # monkeypatch cache_root to tmp? easiest: disable by setting cache_path method
    priors = load_priors(cache)
    assert "DG_Granule" in priors
    assert "PV_Basket" in priors
