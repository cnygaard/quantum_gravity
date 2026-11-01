"""
Large-D_int volume-chaos statistics (about 10 min for all nodes; run in the background).

Referee-grade statistics for the volume-operator level statistics of scrambling.py, on every
node quoted in the papers (all spins pairwise distinct, so there is no S_n symmetry to resolve):
  * valence 6: D_int = 49, 75, 95,
  * valence 7: D_int = 231, 276 (anchor), 306, 336, 391,
  * valence 8: D_int = 553, 673, 937.
The singlet space is built in the Clebsch-Gordan recoupling basis with sparse operators
(scrambling.intertwiner_G), so no dense N0 x N0 matrix is formed (N0 <= 64074).
For each node: the default embedding (sign (-1)^(p+q+r), realized by the alternating moment
curve) with a moving-block-bootstrap error (adjacent ratios share a spacing, so the iid SEM
understates the error), and the ensemble over 64 random REALIZABLE embeddings (sigma = sign det
of random tangents; arbitrary +/-1 tables need not be realizable), split into generic and
degenerate embeddings (extra zero modes / exactly degenerate levels).  Per embedding: r~ on the
full positive spectrum and on the bulk window (ranks 20-90%), Porter-Thomas KS in the
recoupling basis, min |det| of the unit tangents.  Against the matched-D antisymmetric
reference (200 draws): two-sample KS and Levene tests of the per-embedding r~, the small-s
exponent beta of the pooled unfolded bulk spacings (same estimator on both), and the
Spearman correlation of r~ with min |det|.

Writes qfim/report/volume_anchor.json, which PhaseScramble logs (the suite itself does not
rebuild the large nodes).  `python3 scrambling_big.py --plot OUT.pdf` draws the per-embedding
r~ histograms from the JSON without recomputing.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scrambling as S            # noqa: E402

S.MAXN0 = 80000                   # allow the N0 ~ 6.4e4 eight-valent build (sparse)

NODES = [[1, 1.5, 2, 2.5, 3, 4], [1, 2, 2.5, 3, 3.5, 4], [1.5, 2, 2.5, 3, 3.5, 4.5],
         [1, 1.5, 2, 2.5, 3, 3.5, 4.5], [0.5, 1.5, 2, 3, 3.5, 4, 4.5],
         [1, 1.5, 2, 2.5, 3.5, 4, 4.5], [1, 1.5, 2, 3, 3.5, 4, 5], [1, 1.5, 2.5, 3, 3.5, 4, 4.5],
         [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], [0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5],
         [0.5, 1, 2, 2.5, 3, 3.5, 4, 4.5]]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sage', 'qfim', 'report', 'volume_anchor.json')


def main(nodes=NODES, nsign=64):
    results = []
    for spins in nodes:
        t = time.time()
        print(f"node {spins} ...", flush=True)
        res = S.anchor_analysis(spins, nsign=nsign)
        res['seconds'] = round(time.time() - t, 1)
        res['j2_residual'] = S.j0_residual(spins)
        res['basis_defect'] = S.basis_defect(spins)
        d, e, ref, ts = res['default'], res['ensemble'], res['reference'], res.get('tests', {})
        print(f"  D_int={res['D_int']} N0={res['N0']} levels={d['levels']}  [{res['seconds']:.0f}s]  "
              f"J2 res {res['j2_residual']:.1e}, basis defect {res['basis_defect']:.1e}")
        print(f"  default embedding: r~ = {d['mean']:.4f}  iid SEM {d['sem_iid']:.4f}  block-bootstrap "
              f"SE {d['se_block']:.4f} (block {d['block']}, lag-1 corr {d['lag1']:+.3f}); "
              f"zero modes {d['zero_modes']}, degenerate {d['degenerate']}")
        for key in ('generic', 'degenerate', 'all', 'generic_bulk'):
            g = e[key]
            if g['count'] > 1:
                print(f"  {key:12s} embeddings ({g['count']:2d}/{e['nsign']}): r~ = {g['mean']:.4f} +/- "
                      f"{g['sem']:.4f} (SEM; spread {g['std']:.4f}); PT-KS mean {g['ks_mean']:.4f}; "
                      f"near-zero {g['near_zero_mean']:.3f}")
        print(f"  matched-D antisym reference ({ref['nrep']} draws): r~ = {ref['mean']:.4f} +/- "
              f"{ref['std']:.4f} (spread; SEM {ref['sem']:.4f}); bulk {ref['mean_bulk']:.4f}; "
              f"near-zero {ref['near_zero']:.3f}; zero-mode counts {e['zero_mode_counts']}")
        if ts:
            b = ts['beta']
            print(f"  tests: KS vs ref p={ts['ks_ref'][1]:.2g} (bulk p={ts['ks_ref_bulk'][1]:.2g}); "
                  f"Levene p={ts['levene'][1]:.2g} (bulk {ts['levene_bulk'][1]:.2g}); spread ratio "
                  f"{ts['spread_ratio']:.2f}; beta {b['beta']:.2f}+/-{b['se']:.2f} vs ref "
                  f"{b['ref']:.2f}+/-{b['ref_se']:.2f}; Spearman(r~, min|det|) "
                  f"{ts['det_corr'][0]:+.2f} (p={ts['det_corr'][1]:.2g})", flush=True)
        results.append(res)
        S._GCACHE.clear()                            # release the big build before the next node
    with open(OUT, 'w') as f:
        json.dump({'nodes': results, 'nsign': nsign}, f, indent=1)
    print(f"wrote {OUT}")


def plot(src=OUT, out='volume_rtilde.pdf'):
    """Two-panel figure from volume_anchor.json (no recomputation).
    (a) per-embedding r~ of every node (generic embeddings, dots; mean +/- SEM, bar) against the
        matched-D reference mean +/- one draw-to-draw spread; nodes grouped by valence.
    (b) at the D=276 anchor: histogram of per-embedding r~ against the per-draw reference r~."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
    VOL, REF = '#2a78d6', '#eb6834'                         # validated categorical slots 1-2
    nodes = sorted(json.load(open(src))['nodes'], key=lambda nd: (len(nd['spins']), nd['D_int']))
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={'width_ratios': [1.9, 1]})
    rng = np.random.default_rng(0)
    xs, labels = [], []
    x = 0.0
    for i, nd in enumerate(nodes):
        if i and len(nd['spins']) != len(nodes[i - 1]['spins']):
            x += 0.8                                        # gap between valence groups
        e, ref = nd['ensemble'], nd['reference']
        g = [v for v, dg in zip(e['values'], e['degenerate_flags']) if not dg]
        a.fill_between([x - 0.32, x + 0.32], ref['mean'] - ref['std'], ref['mean'] + ref['std'],
                       color=REF, alpha=0.18, lw=0)
        a.plot([x - 0.32, x + 0.32], [ref['mean']] * 2, color=REF, lw=1.5)
        a.scatter(x + rng.uniform(-0.22, 0.22, len(g)), g, s=4, color=VOL, alpha=0.35, lw=0)
        gm = e['generic']
        a.errorbar(x, gm['mean'], yerr=gm['sem'], fmt='D', ms=4.5, color=INK, mec='white',
                   mew=0.8, ecolor=INK, elinewidth=1.4, capsize=0, zorder=5)
        xs.append(x)
        labels.append(str(nd['D_int']))
        x += 1.0
    a.axhline(S.RMT['GOE'], color=INK2, lw=0.7, ls=(0, (3, 2)))
    a.text(xs[-1] + 0.45, S.RMT['GOE'], 'GOE', va='center', ha='left', color=INK2, fontsize=7)
    a.set_xticks(xs, labels)
    a.set_xlim(xs[0] - 0.6, xs[-1] + 1.1)
    a.set_xlabel('intertwiner dimension $D$ (valence 6 | 7 | 8)', color=INK)
    a.set_ylabel(r'$\tilde r$ (positive volume levels)', color=INK)
    a.grid(axis='y', color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.set_ylim(0.30, 0.70)
    a.scatter([], [], s=10, color=VOL, alpha=0.6, label='generic embeddings')
    a.errorbar([], [], yerr=[], fmt='D', ms=4.5, color=INK, mec='white', label='their mean $\\pm$ SEM')
    a.fill_between([], [], [], color=REF, alpha=0.35, lw=0, label='matched reference ($\\pm$1 spread)')
    a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=6.5, labelcolor=INK, ncol=3,
             handletextpad=0.3, columnspacing=0.9)
    nd = next(n for n in nodes if n['D_int'] == 276)
    g = [v for v, dg in zip(nd['ensemble']['values'], nd['ensemble']['degenerate_flags']) if not dg]
    bins = np.linspace(0.49, 0.67, 19)
    b.hist(nd['reference']['values'], bins=bins, density=True, color=REF, alpha=0.35,
           histtype='stepfilled', label=f"reference ({nd['reference']['nrep']} draws)")
    b.hist(g, bins=bins, density=True, color=VOL, histtype='step', lw=1.6,
           label=f'generic embeddings ({len(g)})')
    p = nd['tests']['ks_ref'][1]
    b.text(0.03, 0.97, f'$D=276$\nKS $p={p:.0e}$'.replace('e-0', 'e-'), transform=b.transAxes,
           va='top', color=INK, fontsize=7)
    b.set_xlabel(r'$\tilde r$', color=INK)
    b.set_ylabel('density', color=INK)
    b.legend(loc='lower right', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=6.5,
             labelcolor=INK, ncol=1)
    b.grid(axis='y', color=GRID, lw=0.5)
    b.set_axisbelow(True)
    b.set_title('(b)', loc='left', color=INK, fontsize=8, pad=24, x=-0.25)
    a.set_title('(a)', loc='left', color=INK, fontsize=8, pad=24)
    for ax in (a, b):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == '--plot':
        plot(OUT, sys.argv[2])
    else:
        main()
