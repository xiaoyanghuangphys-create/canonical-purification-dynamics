"""Small-chain support gate for known Hamiltonian and jump operators."""
import numpy as np

from factor_tdvp.tangent_solver import matrix, matrix_to_mpo
from automatic_rank_opening import automatic_rank_opening as auto
from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError
import scalable_cp_witness as sw


def finite_start(previous, model, step, budget, *, root_bond, factor_bond, max_n=4):
    """Inspect the physical support; construct a factor from a finite CP bridge.

    The returned seed still requires the caller's independent endpoint certificate.
    No independent endpoint density or root is used to construct it.
    """
    factor, root, _ = previous
    if root.n > min(4, max_n):
        raise GalerkinError("unsupported_support_audit", "Mixed-MPO support could not be audited", n=root.n)
    x = matrix(root)
    audit = auto.classify(x, model, step, budget)
    if audit["mode"] not in ("first_order_opening", "delayed_opening", "near_boundary"):
        return previous, 0., audit
    seeded, used, details = auto.positive_bridge(x, model, step, budget)
    vals, vecs = np.linalg.eigh(seeded)
    roundoff = 64*np.finfo(float).eps*max(np.max(abs(vals)),1.)
    if np.min(vals) < -roundoff:
        raise GalerkinError("bridge_indefinite", "Finite channel produced an indefinite root", minimum=min(vals))
    # Positive branch of the computed Kraus seed. Only remove roundoff below zero.
    dense_factor = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.conj().T
    raw = matrix_to_mpo(dense_factor, root.n)
    new_factor = sw.hermitian_round(raw, factor_bond)
    state = sw.factor_to_root(new_factor, root_bond)
    state_matrix = matrix(state[1])
    post = auto.detect(state_matrix, model, step)
    seeded_rank = int(np.sum(vals > 1.e-12*vals[-1]))
    if post["rank"] < seeded_rank:
        raise GalerkinError("bridge_support_lost", "Retraction discarded finite-channel support",
                            seeded_rank=seeded_rank, post=post)
    return state, used, dict(audit, bridge=details, post=post,
                             seed_root_error=float(np.linalg.norm(state_matrix-seeded)),
                             factor_retraction_error=float(np.linalg.norm(matrix(state[0])-dense_factor)))
