"""Regressed-mass histogram metrics for the Upsilon fine-tuning.

The metrics here answer two questions that are evaluated over two *different*
sample sets:

  - on the nominal-mass Y->ggg samples (Y at its physical mass), how does the
    regressed mass behave as a function of Y pT?
  - on the smeared-mass Y->ggg samples, how does it behave as a function of the
    generated Y mass?

Rather than hard-coding that pairing, each metric is described by a
`MassHistBlock`: a selection over (sample of origin, truth class) plus an axis
along which that selection is split into individual histograms.  Blocks are
built from plain dicts, so which metric runs over which sample is a matter of
configuration (`eval_kw` in the network config) rather than code.

Histograms are accumulated in numpy and only converted to ROOT objects when
written out, which keeps the per-epoch cost negligible, lets the state be
checkpointed to an .npz (so a resumed or post-hoc run can extend the epoch
axis), and keeps everything except the final write independent of PyROOT.
"""

import os
import numpy as np

from utils.logger import _logger

# `sample_kind` is injected per-file by the filepath rules in utils/data/fileio.py
# and must be declared as an observer in the data config to reach the eval loop.
SAMPLE_KINDS = {
    'qcd': 0,
    'nominal': 1,    # SingleUpsilon / SingleUpsilonToTauHTauH -- Y at its physical mass
    'modified': 2,   # Upsilon_modified_mass -- Y mass smeared over the training range
}


def _as_code_list(spec, table, what):
    """Normalise a selection spec to a list of integer codes, or None for 'any'."""
    if spec is None:
        return None
    if isinstance(spec, (str, int, np.integer)):
        spec = [spec]
    out = []
    for s in spec:
        if isinstance(s, str):
            if s not in table:
                raise ValueError('Unknown %s %r; known: %s' % (what, s, sorted(table)))
            out.append(table[s])
        else:
            out.append(int(s))
    return out


class Split(object):
    """Subdivides a selection into named sub-bins along one variable.

    Two styles, matching the two things we want to look at:
      - `edges`:  contiguous bins, e.g. [50, 100, 200, inf] on the Y pT
      - `points`: windows of +/- `tol` around values, e.g. 6, 15, 24 on the Y mass
    """

    def __init__(self, var, edges=None, points=None, tol=0.5, unit='GeV', label=None):
        if (edges is None) == (points is None):
            raise ValueError("Split needs exactly one of `edges` or `points`")
        self.var = var
        self.unit = unit
        self.label = label or var
        self.tol = float(tol)
        # `None` as an edge means open-ended; the config is passed through
        # ast.literal_eval, which has no spelling for infinity.
        if edges is not None:
            edges = [np.inf if e is None else float(e) for e in edges]
        self.edges = None if edges is None else np.asarray(edges, dtype=float)
        self.points = None if points is None else np.asarray(points, dtype=float)

    @property
    def n_bins(self):
        return len(self.edges) - 1 if self.edges is not None else len(self.points)

    def bin_name(self, i):
        if self.edges is not None:
            lo, hi = self.edges[i], self.edges[i + 1]
            return '%s%g_%s' % (self.var, lo, 'inf' if not np.isfinite(hi) else '%g' % hi)
        return '%s%g' % (self.var, self.points[i])

    def bin_title(self, i):
        if self.edges is not None:
            lo, hi = self.edges[i], self.edges[i + 1]
            if not np.isfinite(hi):
                return '%s > %g %s' % (self.label, lo, self.unit)
            return '%g < %s < %g %s' % (lo, self.label, hi, self.unit)
        return '%s = %g #pm %g %s' % (self.label, self.points[i], self.tol, self.unit)

    def ref_value(self, i):
        """The value a histogram in this sub-bin should ideally peak at, if defined."""
        return None if self.points is None else float(self.points[i])

    def masks(self, values):
        values = np.asarray(values)
        if self.edges is not None:
            return [(values >= self.edges[i]) & (values < self.edges[i + 1])
                    for i in range(self.n_bins)]
        return [np.abs(values - p) < self.tol for p in self.points]


class MassHistBlock(object):
    """One metric: a (sample, class) selection, split into histograms along an axis."""

    def __init__(self, name, split, sample=None, cls=None, title=None):
        self.name = name
        self.split = split if isinstance(split, Split) else Split(**split)
        self.sample = _as_code_list(sample, SAMPLE_KINDS, 'sample')
        self.cls = _as_code_list(cls, {}, 'class')
        self.title = title or name

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(name=d.pop('name'), split=d.pop('split'), sample=d.pop('sample', None),
                   cls=d.pop('cls', None), title=d.pop('title', None))

    def select(self, arrays):
        """Boolean mask over the event arrays for the jets this block covers."""
        n = len(arrays['pred_mass'])
        sel = np.isfinite(arrays['pred_mass'])
        if self.sample is not None:
            if 'sample_kind' not in arrays:
                _logger.warning(
                    "MassHistBlock %r selects on `sample_kind`, which was not collected; "
                    "declare it as an observer in the data config. Block skipped.", self.name)
                return np.zeros(n, dtype=bool)
            sel &= np.isin(arrays['sample_kind'].astype(int), self.sample)
        if self.cls is not None:
            sel &= np.isin(arrays['cls'].astype(int), self.cls)
        if self.split.var not in arrays:
            _logger.warning("MassHistBlock %r splits on %r, which was not collected. "
                            "Block skipped.", self.name, self.split.var)
            return np.zeros(n, dtype=bool)
        sel &= np.isfinite(arrays[self.split.var])
        return sel


def ks_statistic(counts_a, counts_b):
    """Two-sample KS statistic and p-value from two binned distributions.

    Equivalent to ROOT's TH1::KolmogorovTest on the same binning.
    Returns (D, prob); (nan, nan) if either histogram is empty.
    """
    na, nb = counts_a.sum(), counts_b.sum()
    if na <= 0 or nb <= 0:
        return float('nan'), float('nan')
    cdf_a = np.cumsum(counts_a) / na
    cdf_b = np.cumsum(counts_b) / nb
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    ne = np.sqrt(na * nb / (na + nb))
    # Kolmogorov distribution, same series TMath::KolmogorovProb uses
    z = (ne + 0.12 + 0.11 / ne) * d
    if z < 1e-8:
        return d, 1.0
    j = np.arange(1, 101)
    prob = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j ** 2 * z ** 2))
    return d, float(np.clip(prob, 0.0, 1.0))


def _quantile_stats(counts, centers):
    """Median and (half 16-84 range)/median, read off a binned distribution."""
    total = counts.sum()
    if total <= 0:
        return float('nan'), float('nan')
    cdf = np.cumsum(counts) / total
    q16, q50, q84 = np.interp([0.16, 0.50, 0.84], cdf, centers)
    res = 0.5 * (q84 - q16) / q50 if q50 > 0 else float('nan')
    return float(q50), float(res)


class MassHistBook(object):
    """Accumulates per-epoch regressed-mass histograms and writes ROOT/PDF output.

    Counts live in `self.counts[block][sub_bin][epoch]` as numpy arrays over a
    fixed mass binning, so epochs stay directly comparable and the whole state
    round-trips through a small .npz.
    """

    def __init__(self, blocks, mass_bins=(100, 0.0, 50.0), outdir='.', tag='massreg',
                 baseline_epoch=-1, min_entries=50, resp_bins=(120, 0.0, 3.0),
                 selection_metric=None):
        self.blocks = [b if isinstance(b, MassHistBlock) else MassHistBlock.from_dict(b)
                       for b in blocks]
        n, lo, hi = mass_bins
        self.n_mass, self.mass_lo, self.mass_hi = int(n), float(lo), float(hi)
        self.mass_edges = np.linspace(self.mass_lo, self.mass_hi, self.n_mass + 1)
        self.mass_centers = 0.5 * (self.mass_edges[1:] + self.mass_edges[:-1])
        # response = predicted / true mass. Histogrammed alongside the mass itself
        # because it is truth-referenced, so the bias is well defined for both
        # split styles -- a pT-split block has no single "true mass" to compare
        # its mass histogram against, but every jet in it has a response.
        n, lo, hi = resp_bins
        self.n_resp, self.resp_lo, self.resp_hi = int(n), float(lo), float(hi)
        self.resp_edges = np.linspace(self.resp_lo, self.resp_hi, self.n_resp + 1)
        self.resp_centers = 0.5 * (self.resp_edges[1:] + self.resp_edges[:-1])
        self.outdir = outdir
        self.tag = tag
        self.baseline_epoch = baseline_epoch
        self.min_entries = int(min_entries)
        self.selection_metric = selection_metric
        # {block_name: {sub_name: {epoch: counts}}}
        self.counts = {b.name: {} for b in self.blocks}
        self.resp_counts = {b.name: {} for b in self.blocks}

    def _axis(self, which):
        if which == 'resp':
            return self.resp_counts, self.resp_edges, self.resp_centers, self.resp_lo, self.resp_hi
        return self.counts, self.mass_edges, self.mass_centers, self.mass_lo, self.mass_hi

    # -- state -------------------------------------------------------------
    @property
    def state_path(self):
        return os.path.join(self.outdir, '%s_hists.npz' % self.tag)

    def save_state(self):
        os.makedirs(self.outdir, exist_ok=True)
        flat = {}
        for which, store in (('mass', self.counts), ('resp', self.resp_counts)):
            for b, subs in store.items():
                for s, eps in subs.items():
                    for e, c in eps.items():
                        flat['%s|%s|%s|%d' % (which, b, s, e)] = c
        np.savez_compressed(self.state_path, mass_edges=self.mass_edges,
                            resp_edges=self.resp_edges, **flat)
        _logger.info('Saved mass-histogram state (%d histograms) to %s', len(flat), self.state_path)

    def load_state(self):
        """Restore previously filled epochs, so a resumed run extends the epoch axis."""
        if not os.path.exists(self.state_path):
            return False
        with np.load(self.state_path) as f:
            if not np.allclose(f['mass_edges'], self.mass_edges):
                _logger.warning('Mass binning in %s differs from the current config; '
                                'ignoring the stored histograms.', self.state_path)
                return False
            if 'resp_edges' in f.files and not np.allclose(f['resp_edges'], self.resp_edges):
                _logger.warning('Response binning in %s differs from the current config; '
                                'ignoring the stored histograms.', self.state_path)
                return False
            for key in f.files:
                if key in ('mass_edges', 'resp_edges'):
                    continue
                parts = key.split('|')
                if len(parts) != 4:
                    _logger.warning('Ignoring unrecognised key %r in %s (written by an '
                                    'older version?)', key, self.state_path)
                    continue
                which, b, s, e = parts
                store = self.resp_counts if which == 'resp' else self.counts
                if b not in store:
                    continue
                store[b].setdefault(s, {})[int(e)] = f[key]
        _logger.info('Loaded mass-histogram state from %s', self.state_path)
        return True

    # -- filling -----------------------------------------------------------
    def fill(self, epoch, arrays):
        """Histogram one evaluation pass.

        `arrays` maps names to equal-length numpy arrays. `pred_mass` and `cls`
        are required; `sample_kind` and whatever variables the blocks split on
        (e.g. `gen_pt`, `gen_mass`) must be present for those blocks to run.
        """
        have_true = 'true_mass' in arrays
        if not have_true:
            _logger.warning('MassHistBook: `true_mass` not supplied; response histograms '
                            '(and the bias metrics read off them) will be skipped.')
        for block in self.blocks:
            sel = block.select(arrays)
            if not sel.any():
                continue
            pred = arrays['pred_mass'][sel]
            split_vals = arrays[block.split.var][sel]
            resp = None
            if have_true:
                true = arrays['true_mass'][sel]
                with np.errstate(divide='ignore', invalid='ignore'):
                    resp = np.where(true > 0, pred / np.maximum(true, 1e-9), np.nan)
            for i, mask in enumerate(block.split.masks(split_vals)):
                if mask.sum() < self.min_entries:
                    continue
                sub = block.split.bin_name(i)
                self._fill_one('mass', block, sub, i, epoch, pred[mask])
                if resp is not None:
                    r = resp[mask]
                    r = r[np.isfinite(r)]
                    if len(r) >= self.min_entries:
                        self._fill_one('resp', block, sub, i, epoch, r)
            store = self.counts.get(block.name, {})
            filled = sum(1 for s in store.values() if int(epoch) in s)
            _logger.info('MassHistBook: block %r filled %d/%d sub-bins from %d jets at epoch %d',
                         block.name, filled, block.split.n_bins, int(sel.sum()), int(epoch))

    def _fill_one(self, which, block, sub, i, epoch, vals):
        store, edges, _, lo, hi = self._axis(which)
        # Fold under/overflow into the edge bins rather than letting np.histogram
        # drop them: silently discarding the tails would make the normalization,
        # and hence the KS test, differ between epochs purely because the tails moved.
        n_out = int(np.count_nonzero((vals < lo) | (vals > hi)))
        if n_out > 0.01 * len(vals):
            _logger.warning(
                'MassHistBook: block %r sub-bin %r (%s axis) has %.1f%% of entries outside '
                '[%g, %g] at epoch %d; they are piled into the edge bins. Consider '
                'widening `%s_bins`.',
                block.name, sub, which, 100.0 * n_out / len(vals), lo, hi, int(epoch),
                'mass' if which == 'mass' else 'resp')
        counts, _ = np.histogram(np.clip(vals, lo, hi), bins=edges)
        store.setdefault(block.name, {}).setdefault(sub, {})[int(epoch)] = counts

    # -- derived metrics ---------------------------------------------------
    # Every metric below is scoped to one block, so it is computed over exactly one
    # sample by construction -- nominal-mass and smeared-mass numbers are never
    # averaged together. Lower is better for `resolution`, `abs_median_bias` and
    # `ks_vs_baseline`, which is what makes them usable as a selection metric.
    METRICS = ('resolution', 'abs_median_bias', 'ks_vs_baseline')

    def sub_metrics(self, block_name, sub, epoch):
        """The metrics for one sub-bin at one epoch, as a dict (nan where undefined)."""
        out = {}
        eps = self.counts.get(block_name, {}).get(sub, {})
        counts = eps.get(int(epoch))
        if counts is not None:
            med, res = _quantile_stats(counts, self.mass_centers)
            out['median'] = med
            out['resolution'] = res
            ref = eps.get(self.baseline_epoch)
            out['ks_vs_baseline'] = (ks_statistic(counts, ref)[0]
                                     if ref is not None and int(epoch) != self.baseline_epoch
                                     else float('nan'))
            out['ks_prob_vs_baseline'] = (ks_statistic(counts, ref)[1]
                                          if ref is not None and int(epoch) != self.baseline_epoch
                                          else float('nan'))
        # response-based: truth-referenced, so these work for pT-split blocks too
        r_eps = self.resp_counts.get(block_name, {}).get(sub, {})
        r_counts = r_eps.get(int(epoch))
        if r_counts is not None:
            r_med, r_res = _quantile_stats(r_counts, self.resp_centers)
            out['resp_median'] = r_med
            out['resp_resolution'] = r_res
            out['abs_median_bias'] = abs(r_med - 1.0)
        return out

    def scalars(self, epoch, tb_mode='eval'):
        """Per-sub-bin scalars for TensorBoard, all block-scoped."""
        out = []
        for block in self.blocks:
            subs = set(self.counts.get(block.name, {})) | set(self.resp_counts.get(block.name, {}))
            for sub in sorted(subs):
                base = 'MassHist/%s/%s' % (block.name, sub)
                for name, val in self.sub_metrics(block.name, sub, epoch).items():
                    out.append(('%s/%s (%s)' % (base, name, tb_mode), val, epoch))
        sel = self.selection_value(epoch)
        if sel is not None and np.isfinite(sel):
            out.append(('MassHist/selection_metric (%s)' % tb_mode, sel, epoch))
        return out

    def selection_value(self, epoch):
        """The single number this book offers for best-epoch selection, or None.

        Configured as e.g.
            'selection_metric': {'block': 'smeared_by_mass',
                                 'metric': 'abs_median_bias', 'reduce': 'mean'}
        `block` names exactly one block, so the value is computed over exactly one
        sample -- combining nominal and smeared numbers into one score is
        deliberately not expressible here. `sub` optionally restricts it to a
        single sub-bin; otherwise the sub-bins are combined with `reduce`
        ('mean', 'max' or 'worst', where worst == max since lower is better).
        Returns None when unconfigured, and nan when the metric has no value yet.
        """
        cfg = self.selection_metric
        if not cfg:
            return None
        block_name = cfg.get('block')
        metric = cfg.get('metric', 'abs_median_bias')
        if metric not in self.METRICS:
            raise ValueError('selection_metric.metric must be one of %s, got %r'
                             % (list(self.METRICS), metric))
        known = {b.name for b in self.blocks}
        if block_name not in known:
            raise ValueError('selection_metric.block %r is not one of the configured '
                             'blocks %s' % (block_name, sorted(known)))
        subs = [cfg['sub']] if cfg.get('sub') else sorted(
            set(self.counts.get(block_name, {})) | set(self.resp_counts.get(block_name, {})))
        vals = [self.sub_metrics(block_name, s, epoch).get(metric, float('nan')) for s in subs]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return float('nan')
        reduce = cfg.get('reduce', 'mean')
        if reduce == 'mean':
            return float(np.mean(vals))
        if reduce in ('max', 'worst'):
            return float(np.max(vals))
        raise ValueError("selection_metric.reduce must be 'mean', 'max' or 'worst', got %r"
                         % reduce)


_BOOKS = {}


def get_book(kw):
    """Fetch (creating and restoring on first use) the book for this configuration.

    Cached at module level so the epoch axis accumulates across calls within a
    training job, and restored from the .npz so it also survives a resumed one.
    """
    kw = dict(kw)
    kw.pop('write_plots', None)
    tag = kw.get('tag', 'massreg')
    if tag not in _BOOKS:
        book = MassHistBook(**kw)
        book.load_state()
        _BOOKS[tag] = book
    return _BOOKS[tag]


def run_mass_hist(kw, epoch, arrays, tb_helper=None, tb_mode='eval'):
    """Fill one evaluation pass, persist the state, and render if ROOT is available.

    Rendering failures are logged rather than raised: the .npz is the thing worth
    protecting, and it can always be rendered later.
    """
    book = get_book(kw)

    kinds, counts = np.unique(arrays['sample_kind'].astype(int), return_counts=True) \
        if 'sample_kind' in arrays else (np.array([]), np.array([]))
    if len(kinds):
        inv = {v: k for k, v in SAMPLE_KINDS.items()}
        _logger.info('MassHistBook: %s sample composition: %s', tb_mode,
                     ', '.join('%s=%d' % (inv.get(int(k), 'kind%d' % k), c)
                               for k, c in zip(kinds, counts)))

    book.fill(epoch, arrays)
    book.save_state()

    if tb_helper is not None:
        tb_helper.write_scalars(book.scalars(epoch, tb_mode=tb_mode))

    if dict(kw).get('write_plots', True):
        try:
            from utils.nn.mass_hist_root import write_root_output
            write_root_output(book)
        except ImportError as e:
            _logger.warning('Skipping ROOT/PDF output: %s', e)
        except Exception:
            import traceback
            _logger.error('Failed to write ROOT/PDF output:\n%s', traceback.format_exc())
    return book
