"""PyROOT rendering for the regressed-mass histograms.

Kept separate from `mass_hist` on purpose: filling and the derived scalars are
pure numpy, so a training job can accumulate the histograms (and push scalars to
TensorBoard) in an environment without ROOT, and the ROOT/PDF output can be
produced either in-process or afterwards from the saved .npz by
`scripts/mass_reg_posthoc.py` running under an environment that does have ROOT.
"""

import os
import numpy as np

from utils.logger import _logger
from utils.nn.mass_hist import _quantile_stats

_PALETTE = [1, 632, 600, 418, 800, 880, 616, 434]  # kBlack, kRed, kBlue, kGreen+2, kOrange, kViolet, kMagenta, kCyan+2


def _root():
    try:
        import ROOT
    except ImportError as e:
        raise ImportError(
            'PyROOT is required to write the mass-histogram plots. Either run in an '
            'environment with ROOT, or leave the .npz state to be rendered later by '
            'scripts/mass_reg_posthoc.py --from-state.') from e
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.TH1.AddDirectory(False)
    return ROOT


def _make_th1(ROOT, name, title, counts, book, normalize=False):
    h = ROOT.TH1F(name, title, book.n_mass, book.mass_lo, book.mass_hi)
    for i, c in enumerate(counts):
        h.SetBinContent(i + 1, float(c))
        h.SetBinError(i + 1, float(np.sqrt(max(c, 0.0))))
    h.GetXaxis().SetTitle('Regressed mass [GeV]')
    h.GetYaxis().SetTitle('Entries')
    if normalize and h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
        h.GetYaxis().SetTitle('Normalized entries')
    return h


def _make_th2(ROOT, name, title, eps, counts_by_epoch, book):
    """Mass (x) vs. epoch (y); one of these per sub-bin, since 2D overlays don't work."""
    e_lo, e_hi = min(eps), max(eps)
    n_e = e_hi - e_lo + 1
    h = ROOT.TH2F(name, title, book.n_mass, book.mass_lo, book.mass_hi,
                  n_e, e_lo - 0.5, e_hi + 0.5)
    for e in eps:
        counts = counts_by_epoch[e]
        total = counts.sum()
        for i, c in enumerate(counts):
            # normalize each epoch row so the shape evolution is visible regardless
            # of how many jets happened to land in this sub-bin that epoch
            h.SetBinContent(i + 1, e - e_lo + 1, float(c) / total if total > 0 else 0.0)
    h.GetXaxis().SetTitle('Regressed mass [GeV]')
    h.GetYaxis().SetTitle('Epoch')
    h.GetZaxis().SetTitle('Normalized entries')
    return h


def _make_graph(ROOT, name, eps, values, y_title):
    xs = np.asarray(eps, dtype=float)
    ys = np.asarray(values, dtype=float)
    good = np.isfinite(ys)
    if not good.any():
        return None
    g = ROOT.TGraph(int(good.sum()), xs[good].astype(np.float64), ys[good].astype(np.float64))
    g.SetName(name)
    g.GetXaxis().SetTitle('Epoch')
    g.GetYaxis().SetTitle(y_title)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    return g


def write_root_output(book, outdir=None, tag=None):
    """Write every histogram to a TFile and a multi-page PDF per block.

    Returns (root_path, [pdf paths]).
    """
    ROOT = _root()
    outdir = outdir or book.outdir
    tag = tag or book.tag
    os.makedirs(outdir, exist_ok=True)

    root_path = os.path.join(outdir, '%s.root' % tag)
    fout = ROOT.TFile(root_path, 'RECREATE')
    pdfs = []

    for block in book.blocks:
        subs = book.counts.get(block.name, {})
        if not subs:
            continue
        d = fout.mkdir(block.name)
        sub_names = [block.split.bin_name(i) for i in range(block.split.n_bins)
                     if block.split.bin_name(i) in subs]
        all_eps = sorted({e for s in sub_names for e in subs[s]})
        if not all_eps:
            continue
        last_ep = all_eps[-1]

        pdf = os.path.join(outdir, '%s_%s.pdf' % (tag, block.name))
        c = ROOT.TCanvas('c_%s' % block.name, block.title, 900, 700)
        c.Print(pdf + '[')

        # --- page 1: the sub-bin histograms overlaid, at the latest epoch ---
        keep = []
        leg = ROOT.TLegend(0.58, 0.68, 0.89, 0.89)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        first = True
        for k, s in enumerate(sub_names):
            if last_ep not in subs[s]:
                continue
            h = _make_th1(ROOT, '%s_%s_ep%d' % (block.name, s, last_ep),
                          '%s, epoch %d;Regressed mass [GeV];Normalized entries'
                          % (block.title, last_ep),
                          subs[s][last_ep], book, normalize=True)
            h.SetLineColor(_PALETTE[k % len(_PALETTE)])
            h.SetLineWidth(2)
            med, _ = _quantile_stats(subs[s][last_ep], book.mass_centers)
            i_sub = [block.split.bin_name(i) for i in range(block.split.n_bins)].index(s)
            leg.AddEntry(h, '%s (med %.1f)' % (block.split.bin_title(i_sub), med), 'l')
            h.Draw('HIST' if first else 'HIST SAME')
            first = False
            keep.append(h)
        if keep:
            keep[0].GetYaxis().SetRangeUser(0, 1.35 * max(h.GetMaximum() for h in keep))
            leg.Draw()
            c.Print(pdf)

        # --- one page per sub-bin: mass vs. epoch, plus the trend graphs ---
        for i in range(block.split.n_bins):
            s = block.split.bin_name(i)
            if s not in subs:
                continue
            eps = sorted(subs[s])
            h2 = _make_th2(ROOT, '%s_%s_vs_epoch' % (block.name, s),
                           '%s, %s' % (block.title, block.split.bin_title(i)),
                           eps, subs[s], book)
            c.Clear()
            h2.Draw('COLZ')
            ref = block.split.ref_value(i)
            line = None
            if ref is not None and book.mass_lo < ref < book.mass_hi:
                line = ROOT.TLine(ref, h2.GetYaxis().GetXmin(), ref, h2.GetYaxis().GetXmax())
                line.SetLineColor(2)
                line.SetLineStyle(2)
                line.SetLineWidth(2)
                line.Draw()
            c.Print(pdf)
            d.cd()
            h2.Write()

            # trends read straight off the book, so the plots and the TensorBoard
            # scalars can never disagree about what a metric means
            series = {k: [] for k in ('median', 'resolution', 'ks_vs_baseline',
                                      'abs_median_bias')}
            for e in eps:
                m = book.sub_metrics(block.name, s, e)
                for k in series:
                    series[k].append(m.get(k, np.nan))
                h1 = _make_th1(ROOT, '%s_%s_ep%d' % (block.name, s, e),
                               '%s, %s, epoch %d' % (block.title, block.split.bin_title(i), e),
                               subs[s][e], book)
                d.cd()
                h1.Write()

            c.Clear()
            c.Divide(2, 2)
            graphs = [
                _make_graph(ROOT, '%s_%s_median' % (block.name, s), eps, series['median'],
                            'Median regressed mass [GeV]'),
                _make_graph(ROOT, '%s_%s_resolution' % (block.name, s), eps, series['resolution'],
                            'Resolution (half 16-84)/median'),
                _make_graph(ROOT, '%s_%s_ks' % (block.name, s), eps, series['ks_vs_baseline'],
                            'KS vs. epoch %d' % book.baseline_epoch),
                _make_graph(ROOT, '%s_%s_absbias' % (block.name, s), eps,
                            series['abs_median_bias'], '|median(pred/true) - 1|'),
            ]
            keepalive = []
            for gi, g in enumerate(graphs):
                if g is None:
                    continue
                c.cd(gi + 1)
                g.SetTitle('%s, %s' % (block.title, block.split.bin_title(i)))
                g.Draw('ALP')
                if gi == 0 and ref is not None:
                    ln = ROOT.TLine(min(eps), ref, max(eps), ref)
                    ln.SetLineColor(2)
                    ln.SetLineStyle(2)
                    ln.Draw()
                    keepalive.append(ln)  # must outlive the Print below
                d.cd()
                g.Write()
            c.Print(pdf)
            c.Clear()

        c.Print(pdf + ']')
        pdfs.append(pdf)
        _logger.info('Wrote %s', pdf)

    fout.Close()
    _logger.info('Wrote %s', root_path)
    return root_path, pdfs
