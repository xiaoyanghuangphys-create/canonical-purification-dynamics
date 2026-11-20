"""Gauge-fixed complex frames for one-site and overlapping two-site MPS variations.

The local index s=2*i+k pairs the spin row i and reference column k. A bond
number k is the cut immediately before zero-indexed site k. Prefix and suffix
bases are independently left- and right-canonical. No global tangent columns
are stored. Use factor_tdvp.tangent_solver.matrix_to_mpo / matrix for small
dense conversions using this same interleaving.
"""
from __future__ import annotations

import copy
import numpy as np

from factor_tdvp.tangent_solver import tn


def tangent_dimension(ranks, mode="one", d=4):
    """Complex projector rank for active OBC ranks (1, chi_1, ..., 1)."""
    if mode not in ("one", "two"):
        raise ValueError("mode must be 'one' or 'two'")
    if (len(ranks) < 2 or ranks[0] != 1 or ranks[-1] != 1
            or any(k < 1 or int(k) != k for k in ranks)):
        raise ValueError("expected positive open-chain bond dimensions")
    if any(a > d*b or b > d*a for a, b in zip(ranks, ranks[1:])):
        raise ValueError("bond ranks violate a one-site Schmidt bound")
    n = len(ranks) - 1
    if mode == "one" or n == 1:
        return sum(d*a*b for a, b in zip(ranks, ranks[1:])) - sum(k*k for k in ranks[1:-1])
    return (sum(d*d*ranks[j]*ranks[j+2] for j in range(n-1))
            - sum(d*ranks[j]*ranks[j+1] for j in range(1, n-1)))


class TangentFrame:
    """Isometry B from flat complex coordinates into the doubled-site chain.

    Block j has shape (q_j, 4**(width-1)*chi_{j+width}). Its first
    dimension is the left complement size 4*chi_j-chi_{j+1}, except on
    the unrestricted last block where it is 4*chi_j. Coordinates are in
    NumPy C order, block by block. The frame includes the radial direction.
    """
    def __init__(self, state, mode="one", rank_rtol=1e-13):
        if not np.isfinite(rank_rtol) or not 0 <= rank_rtol < 1:
            raise ValueError("rank_rtol must lie in [0, 1)")
        if mode not in ("one", "two"):
            raise ValueError("mode must be 'one' or 'two'")
        if len(state.tensors) != state.n or state.n < 1:
            raise ValueError("expected a nonempty MPS")
        prev = 1
        for a in state.tensors:
            if a.ndim != 3 or a.shape[:2] != (prev, 4) or not np.isfinite(a).all():
                raise ValueError("invalid MPS site shape or entries")
            prev = a.shape[2]
        if prev != 1:
            raise ValueError("expected an open right boundary")

        self.n, self.mode = state.n, mode
        self.width = min(self.n, 1 if mode == "one" else 2)
        stored = (1, *(a.shape[-1] for a in state.tensors))
        # The right QR followed by a left SVD in compress exposes the actual
        # Schmidt sectors, even for padded tensors or an invalid center hint.
        self.left, info = tn.compress(state, max(stored), tolerance=rank_rtol)
        if info["norm_before"] == 0:
            raise ValueError("the zero MPS has no regular active-rank frame")
        self.right = copy.deepcopy(self.left)
        self.right.move_center(0)
        self.ranks = (1, *(a.shape[-1] for a in self.left.tensors))
        self.complements, self.shapes, self.offsets = [], [], [0]
        for j in range(self.n-self.width+1):
            rows = 4*self.ranks[j]
            if j == self.n-self.width:
                complement = np.eye(rows, dtype=complex)
            else:
                a = self.left.tensors[j].reshape(rows, self.ranks[j+1])
                q, _ = np.linalg.qr(a, mode="complete")
                complement = q[:, self.ranks[j+1]:]
            cols = 4**(self.width-1)*self.ranks[j+self.width]
            self.complements.append(complement)
            self.shapes.append((complement.shape[1], cols))
            self.offsets.append(self.offsets[-1] + complement.shape[1]*cols)

        self.dimension = self.offsets[-1]
        if self.dimension != tangent_dimension(self.ranks, mode):
            raise RuntimeError("inconsistent coordinate and projector dimensions")
        self.metadata = dict(mode=mode, width=self.width, dimension=self.dimension,
                             stored_ranks=stored, active_ranks=self.ranks,
                             block_shapes=self.shapes, offsets=self.offsets,
                             rank_rtol=rank_rtol, rank_discard_bound=info["discarded_bound"])
        # Three states at each cut: before a variation, inside a pair, after a
        # variation. Unreachable states have zero width, including boundaries.
        self._cuts = []
        for k, chi in enumerate(self.ranks):
            before = chi if k <= self.n-self.width else 0
            middle = min(self.shapes[k-1]) if self.width == 2 and 0 < k < self.n else 0
            after = chi if k >= self.width else 0
            self._cuts.append((slice(0, before), slice(before, before+middle),
                               slice(before+middle, before+middle+after)))

    def decode(self, v):
        """B v as a single MPO. Tensor-network contraction is linear in v.

        Each path inserts coordinates exactly once. A pair splits as Q times Z
        or (Q Z) times I, whichever uses the smaller fixed intermediate bond.
        Neither choice depends on v, and neither uses an SVD.
        """
        v = np.asarray(v)
        if v.shape != (self.dimension,) or not np.isfinite(v).all():
            raise ValueError("expected finite complex tangent coordinates of shape (dimension,)")
        tensors = []
        for j in range(self.n):
            left, middle, done = self._cuts[j]
            left1, middle1, done1 = self._cuts[j+1]
            t = np.zeros((done.stop, 4, done1.stop), complex)
            if left1.stop:
                t[left, :, left1] = self.left.tensors[j]
            if done.stop > done.start:
                t[done, :, done1] = self.right.tensors[j]
            tensors.append(t)

        for j, (q, shape) in enumerate(zip(self.complements, self.shapes)):
            z = v[self.offsets[j]:self.offsets[j+1]].reshape(shape)
            left = self._cuts[j][0]
            if self.width == 1:
                done = self._cuts[j+1][2]
                tensors[j][left, :, done] = (q @ z).reshape(self.ranks[j], 4, self.ranks[j+1])
            else:
                middle = self._cuts[j+1][1]
                done = self._cuts[j+2][2]
                first, second = (q, z) if shape[0] <= shape[1] else (q @ z, np.eye(shape[1]))
                tensors[j][left, :, middle] = first.reshape(self.ranks[j], 4, min(shape))
                tensors[j+1][middle, :, done] = second.reshape(min(shape), 4, self.ranks[j+2])
        out = tn.MPS(self.n, max_bond=max(t.shape[-1] for t in tensors))
        out.tensors, out.center = tensors, self.n-1
        return out

    def encode(self, value):
        """B†Y by canonical prefix and suffix/environment contractions."""
        if value.n != self.n:
            raise ValueError("mismatched chain sizes")
        right = [None]*(self.n+1)
        right[-1] = np.ones((1, 1), complex)
        for j in range(self.n-1, self.width-1, -1):
            right[j] = np.einsum("ru,asr,bsu->ab", right[j+1], self.right.tensors[j].conj(),
                                 value.tensors[j], optimize=True)
        left = np.ones((1, 1), complex)
        v = np.empty(self.dimension, complex)
        for j, q in enumerate(self.complements):
            block = np.tensordot(left, value.tensors[j], axes=(1, 0))
            suffix = right[j+self.width].T
            if self.width == 2:
                suffix = np.tensordot(value.tensors[j+1], suffix, axes=(-1, 0))
            local = np.tensordot(block, suffix, axes=(-1, 0)).reshape(q.shape[0], -1)
            v[self.offsets[j]:self.offsets[j+1]] = (q.conj().T @ local).ravel()
            if j < self.n-self.width:
                left = np.einsum("asr,asu->ru", self.left.tensors[j].conj(), block, optimize=True)
        return v

    def project(self, value):
        """Orthogonal Frobenius projector BB†."""
        return self.decode(self.encode(value))


def build_frame(state, mode="one", rank_rtol=1e-13):
    """Build a frame at the numerical active ranks; metadata records trimming."""
    return TangentFrame(state, mode, rank_rtol)
