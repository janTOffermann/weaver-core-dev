#!/usr/bin/env python
"""Build the regressed-mass histogram metrics after the fact, from saved checkpoints.

Two things this gives you that the in-training path does not:

  - metrics over samples the training job never validated on (in particular the
    nominal-mass Y->ggg samples, which only appear in `--data-test`), without
    changing what the validation loss is averaged over;
  - a post-mortem of a training that finished before these metrics existed.

It takes the same weaver arguments as `train.py` (data config, network config,
`--data-test`, `-o` network options including `eval_kw`), plus the options below.

Typical use, in three steps:

  # 1. the pre-finetune reference, stored at the book's baseline epoch (-1)
  python weaver/scripts/mass_reg_posthoc.py ... --baseline

  # 2. every epoch checkpoint the training wrote
  python weaver/scripts/mass_reg_posthoc.py ... --epochs 0-29

  # 3. render, under an environment that has PyROOT
  conda activate weaver-root
  python weaver/scripts/mass_reg_posthoc.py -o eval_kw "{...}" --from-state

Steps 1 and 2 need torch and the data but not ROOT; step 3 needs ROOT and the
.npz the first two wrote, but neither torch nor the data -- it does not import
`train` at all. That is why ROOT need not be installed next to the training
stack: `weaver` fills the histograms, `weaver-root` draws them.
"""

import os
import sys
import ast
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.logger import _logger  # noqa: E402
from utils.nn.mass_hist import MassHistBook, get_book  # noqa: E402


def add_extra_args(p):
    p.add_argument('--epochs', type=str, default=None,
                   help='epoch checkpoints to process, e.g. "0-29" or "0,5,10". '
                        'Default: every `<model-prefix>_epoch-*_state.pt` found.')
    p.add_argument('--baseline', action='store_true', default=False,
                   help='evaluate the pre-finetune starting point (--load-model-weights as '
                        'given, with no epoch checkpoint on top) and store it at the '
                        "book's baseline epoch, for the KS comparison.")
    p.add_argument('--from-state', action='store_true', default=False,
                   help='skip inference: load the saved .npz and only render the ROOT/PDF '
                        'output. Needs PyROOT but neither torch nor the data.')
    p.add_argument('--test-group', type=str, default=None,
                   help='which --data-test file group to run on (default: all of them)')
    return p


def parse_epochs(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part.lstrip('-'):
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def mass_hist_kw(network_option):
    opts = {k: ast.literal_eval(v) for k, v in network_option}
    kw = opts.get('eval_kw', {}).get('mass_hist_kw')
    if kw is None:
        raise SystemExit(
            "No `mass_hist_kw` found. Pass it the same way the training does, e.g.\n"
            "  -o eval_kw \"{'mass_hist_kw': {'outdir': ..., 'blocks': [...]}}\"")
    return kw


def book_kwargs(kw):
    return {k: v for k, v in kw.items() if k != 'write_plots'}


def render_from_state(network_option):
    book = MassHistBook(**book_kwargs(mass_hist_kw(network_option)))
    if not book.load_state():
        raise SystemExit('No saved state at %s; run the inference steps first.'
                         % book.state_path)
    from utils.nn.mass_hist_root import write_root_output
    root_path, pdfs = write_root_output(book)
    print('Wrote:\n  %s' % '\n  '.join([root_path] + pdfs))


def run_inference(args, kw, epoch, weights, test_loaders, data_config, model):
    import torch
    import numpy as np
    from utils.nn.mass_hist import run_mass_hist

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        _logger.info('Loaded %s', weights)
    model = model.to(dev).eval()
    n_cls = model.num_cls_nodes

    # the regression label, so the response (pred/true) histograms can be filled
    # here exactly as they are in the in-training path
    reg_label = data_config.label_names[1] if len(data_config.label_names) > 1 else None
    if reg_label is None:
        raise SystemExit('The data config has no regression label; nothing to histogram.')

    names = [args.test_group] if args.test_group else list(test_loaders)
    acc = {k: [] for k in ['pred_factor', 'true_factor', 'cls', 'jet_mass',
                           'gen_pt', 'gen_mass', 'sample_kind']}
    obs_of = {'jet_mass': 'scoutfj_mass', 'gen_pt': 'scoutfj_gen_pt',
              'gen_mass': 'scoutfj_gen_mass', 'sample_kind': 'sample_kind'}
    missing = set()

    with torch.no_grad():
        for name in names:
            _logger.info('Epoch %s: running test group %s', epoch, name)
            for X, y, Z in test_loaders[name]():
                inputs = [X[k].to(dev) for k in data_config.input_names]
                out = model(*inputs)
                acc['pred_factor'].append(out[:, n_cls:].float().cpu().numpy()[:, 0])
                acc['true_factor'].append(y[reg_label].float().cpu().numpy().reshape(-1))
                acc['cls'].append(y['truth_label'].long().cpu().numpy())
                for key, obs in obs_of.items():
                    if obs in Z:
                        acc[key].append(Z[obs].cpu().numpy().reshape(-1))
                    else:
                        missing.add(obs)

    for obs in sorted(missing):
        _logger.warning('Observer %r not present; metrics needing it will be skipped. '
                        'Declare it in the data config.', obs)

    n = len(acc['cls'])
    if n == 0:
        _logger.warning('Epoch %s: no jets selected, nothing filled', epoch)
        return
    arrays = {k: np.concatenate(v) for k, v in acc.items() if len(v) == n}
    if 'jet_mass' not in arrays:
        raise SystemExit('`scoutfj_mass` is required to turn the regressed factor into a mass.')
    # the network regresses gen_mass / scoutfj_mass, so undo that here
    arrays['pred_mass'] = arrays.pop('pred_factor') * arrays['jet_mass']
    arrays['true_mass'] = arrays.pop('true_factor') * arrays['jet_mass']
    _logger.info('Epoch %s: collected %d jets', epoch, len(arrays['cls']))

    run_mass_hist(dict(kw, write_plots=False), epoch, arrays)


def main():
    # --from-state must not import `train` (and therefore torch), so dispatch on it
    # before building the full weaver parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('-o', '--network-option', nargs=2, action='append', default=[])
    pre.add_argument('--from-state', action='store_true', default=False)
    known, _ = pre.parse_known_args()
    if known.from_state:
        render_from_state(known.network_option)
        return

    from train import parser, test_load, model_setup
    args = add_extra_args(parser).parse_args()
    kw = mass_hist_kw(args.network_option)

    test_loaders, data_config = test_load(args)
    # model_setup returns (model, model_info, loss_func, train, evaluate, save_fn)
    # -- its docstring says otherwise, don't trust it.
    model = model_setup(args, data_config)[0]

    if args.baseline:
        book = get_book(dict(kw, write_plots=False))
        _logger.info('Evaluating the pre-finetune starting point as epoch %d',
                     book.baseline_epoch)
        run_inference(args, kw, book.baseline_epoch, None, test_loaders, data_config, model)
        return

    if args.epochs:
        epochs = parse_epochs(args.epochs)
    else:
        found = glob.glob(args.model_prefix + '_epoch-*_state.pt')
        epochs = sorted(int(os.path.basename(f).split('_epoch-')[1].split('_')[0])
                        for f in found)
        if not epochs:
            raise SystemExit('No checkpoints matching %s_epoch-*_state.pt' % args.model_prefix)
    _logger.info('Processing epochs: %s', epochs)

    for epoch in epochs:
        weights = args.model_prefix + '_epoch-%d_state.pt' % epoch
        if not os.path.exists(weights):
            _logger.warning('Skipping epoch %d: %s not found', epoch, weights)
            continue
        run_inference(args, kw, epoch, weights, test_loaders, data_config, model)

    print('Done. Render the plots with --from-state under an environment that has PyROOT.')


if __name__ == '__main__':
    main()
