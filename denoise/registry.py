"""
denoise.registry — local model registry for Noise2Inverse trained models.

Registry location (in order of precedence):
  1. DENOISE_REGISTRY environment variable
  2. ~/.denoise/registry/

Each entry is a sub-directory containing:
  config.yaml            — full denoise config (with metadata: block)
  best_val_model.pth
  best_lcl_model.pth
  best_edge_model.pth
"""

import os
import re
import shutil
import pathlib
import datetime

# Fields used to decide whether two configs share the same noise fingerprint.
# Exact string match.  'temperature' is optional (only scored when present in both).
_MATCH_KEYS = [
    'beamline',
    'mode',
    'energy',
    'type',              # scintillator type
    'active_thickness',
    'serial_number',     # detector serial — most specific identifier
    'exposure_time',
    'binning_x',
    'binning_y',
]
_OPTIONAL_MATCH_KEYS = ['temperature']

REGISTRY_DIR = pathlib.Path(
    os.environ.get('DENOISE_REGISTRY', os.path.expanduser('~/.denoise/registry'))
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path):
    import yaml
    with open(config_path) as fh:
        content = fh.read()
    try:
        return yaml.safe_load(content)
    except yaml.constructor.ConstructorError:
        # Older configs may contain numpy-tagged scalars; fall back to full loader.
        return yaml.load(content, Loader=yaml.UnsafeLoader)


def _slug(meta):
    """Generate a short human-readable name from metadata."""
    parts = []
    for key in ('beamline', 'mode', 'energy', 'serial_number', 'exposure_time'):
        val = meta.get(key, '')
        if val:
            # strip units, keep alphanumeric
            parts.append(re.sub(r'[^A-Za-z0-9.]+', '', str(val)))
    return '_'.join(p for p in parts if p) or 'model'


def _score(query_meta, entry_meta):
    """Return (n_matched, n_total) for noise-relevant keys."""
    matched = 0
    total = 0
    for key in _MATCH_KEYS:
        if key in query_meta:
            total += 1
            if str(query_meta[key]) == str(entry_meta.get(key, '')):
                matched += 1
    # optional keys: only count when present in *both*
    for key in _OPTIONAL_MATCH_KEYS:
        if key in query_meta and key in entry_meta:
            total += 1
            if str(query_meta[key]) == str(entry_meta[key]):
                matched += 1
    return matched, total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(config_path):
    """
    Search the registry for models matching the noise fingerprint of *config_path*.

    Returns a list of dicts, sorted by score descending::

        [{'dir': Path, 'config': dict, 'matched': int, 'total': int,
          'score': float, 'registered': str}, ...]

    Returns an empty list when the registry is empty or the config has no
    ``metadata:`` block.
    """
    cfg = _load_config(config_path)
    query_meta = cfg.get('metadata', {})
    if not query_meta:
        return []

    if not REGISTRY_DIR.is_dir():
        return []

    results = []
    for entry_dir in sorted(REGISTRY_DIR.iterdir()):
        entry_cfg_path = entry_dir / 'config.yaml'
        if not entry_cfg_path.exists():
            continue
        entry_cfg = _load_config(entry_cfg_path)
        entry_meta = entry_cfg.get('metadata', {})
        matched, total = _score(query_meta, entry_meta)
        if total == 0:
            continue
        results.append({
            'dir':        entry_dir,
            'config':     entry_cfg,
            'matched':    matched,
            'total':      total,
            'score':      matched / total,
            'registered': entry_meta.get('start_date', 'unknown'),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def register(config_path, model_dir, name=None):
    """
    Copy *model_dir*/{best_val,best_lcl,best_edge}_model.pth and *config_path*
    into the registry under an auto-generated (or user-supplied) *name*.

    Returns the Path of the registry entry directory.
    """
    cfg = _load_config(config_path)
    meta = cfg.get('metadata', {})

    if name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = _slug(meta) + '_' + timestamp

    entry_dir = REGISTRY_DIR / name
    entry_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(config_path, entry_dir / 'config.yaml')

    # Copy checkpoints that exist
    model_dir = pathlib.Path(model_dir)
    copied = []
    for pth in ('best_val_model.pth', 'best_lcl_model.pth', 'best_edge_model.pth'):
        src = model_dir / pth
        if src.exists():
            shutil.copy2(src, entry_dir / pth)
            copied.append(pth)

    return entry_dir, copied


def list_registry():
    """Return all registry entries as a list of (name, config) tuples."""
    if not REGISTRY_DIR.is_dir():
        return []
    entries = []
    for entry_dir in sorted(REGISTRY_DIR.iterdir()):
        cfg_path = entry_dir / 'config.yaml'
        if cfg_path.exists():
            entries.append((entry_dir.name, _load_config(cfg_path)))
    return entries
