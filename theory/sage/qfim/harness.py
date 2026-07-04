"""
qfim/harness.py -- record + tolerance checking + report emission.

Every test appends a record {id, phase, computed, target, tier, status, error, note}.
Tier semantics:
  T0  symbolic-exact (Sage)  -> pass is a boolean fed in as computed==target
  T1  sum vs closed form     -> abs error < 1e-30, HARD gate
  T2  hand anchors           -> RELATIVE error < 1e-5, LOG only (never gates)
  Z   structural zeros       -> |value| < 1e-33, HARD gate
"""
import json
import os
from mpmath import mp, mpf, fabs

from .core.dictionary import TOL

RECORDS = []


def _flt(x):
    try:
        return float(mpf(x))
    except Exception:
        try:
            z = complex(x)
            return z.real if abs(z.imag) < 1e-12 else [z.real, z.imag]
        except Exception:
            return None


def rec(id, phase, computed, target, tier, note="", kind=None):
    """
    computed/target may be scalars or lists; scalars may be mpc (complex kernel
    values). `kind` in {'abs','rel'} overrides the default (T2 -> rel, others ->
    abs). For Z/Znum, target is ignored (distance from 0).
    """
    tol = TOL.get(tier, mpf(0))
    comp = list(computed) if isinstance(computed, (list, tuple)) else [computed]
    if tier in ('Z', 'Znum'):
        errs = [abs(c) for c in comp]
    else:
        targ = list(target) if isinstance(target, (list, tuple)) else [target] * len(comp)
        use_rel = (kind == 'rel') or (kind is None and tier == 'T2')
        errs = []
        for c, t in zip(comp, targ):
            t = mpf(t)
            d = abs(c - t)                         # modulus if c is mpc
            if use_rel:
                errs.append(d / (fabs(t) if t != 0 else mpf(1)))
            else:
                errs.append(d)
    err = max(errs)
    passed = bool(err < tol) if tier != 'T0' else bool(computed == target)
    status = 'PASS' if passed else 'FAIL'
    if tier == 'T2':
        status = 'LOG-' + status          # advisory: never gates the run
    R = {
        'id': id, 'phase': phase, 'tier': tier, 'status': status,
        'computed': (_flt(comp[0]) if len(comp) == 1 else [_flt(c) for c in comp]),
        'target': (None if target is None else
                   (_flt(target) if not isinstance(target, (list, tuple))
                    else [_flt(t) for t in target])),
        'error': _flt(err), 'tol': _flt(tol), 'note': note,
    }
    RECORDS.append(R)
    return R


def boolrec(id, phase, ok, tier, note="", computed=None, target=None):
    """Record a boolean (used for T0 symbolic and structural pass/fail)."""
    R = {
        'id': id, 'phase': phase, 'tier': tier,
        'status': 'PASS' if ok else 'FAIL',
        'computed': computed, 'target': target, 'error': None,
        'tol': _flt(TOL.get(tier, 0)), 'note': note,
    }
    RECORDS.append(R)
    return R


HARD_TIERS = ('T0', 'T1', 'Z', 'NUM', 'Znum')   # T2 is advisory (log-don't-fail)


def phase_status(phase):
    rs = [r for r in RECORDS if r['phase'] == phase]
    hard = [r for r in rs if r['tier'] in HARD_TIERS]
    failed = [r for r in hard if r['status'] == 'FAIL']
    return ('GATE-FAIL' if failed else 'OK'), len(rs), len(failed)


def summarize(reset=False):
    from collections import OrderedDict
    phases = OrderedDict()
    for r in RECORDS:
        phases.setdefault(r['phase'], []).append(r)
    lines = []
    for ph, rs in phases.items():
        npass = sum(1 for r in rs if r['status'] in ('PASS', 'LOG-PASS'))
        nfail = sum(1 for r in rs if r['status'] in ('FAIL', 'LOG-FAIL'))
        gate = 'GATE-FAIL' if any(r['status'] == 'FAIL' and r['tier'] in HARD_TIERS for r in rs) else 'OK'
        lines.append(f"  {ph:26s} {npass:3d} pass  {nfail:3d} fail   [{gate}]")
    return "\n".join(lines)


def dump(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(RECORDS, f, indent=2)
    return path
