"""Method A's fixed preparation congruence restricted to B's normalized tangent.

For B an isometric tangent frame, Q the orthogonal projector off its radial
direction, and W the positive MPO rho(0)^(-1/4), the action is

    (scale / 2) Q B* [ W (B Qv) W ].

The positive scalar compensates the division of the Galerkin matrix by
`scale` before PCG. This map is complex linear and positive definite on
the normalized tangent. All contractions use the input anchor and tangent
MPO without a vector-dependent truncation or a dense global operator.
"""
import numpy as np


def _congruence_core(core):
    """Local operator on vectorized matrices, with paired anchor bonds."""
    left, _, right = core.shape
    w = core.reshape(left, 2, 2, right)
    tensor = np.einsum('aikb,cljd->acijklbd', w, w, optimize=True)
    return tensor.reshape(left*left, 4, 4, right*right)


def _open_right(operator, value, suffix):
    """Leave one physical index and the right tangent bond open."""
    env = np.tensordot(suffix, value, axes=(2, 2))  # right, Kright, input-left, phys-in
    tile = np.tensordot(operator, env, axes=((2, 3), (3, 1)))
    return tile.transpose(0, 3, 1, 2)  # Kleft, input-left, phys-out, right


def encode_action(frame, operators, value):
    """Encode a local-superoperator MPO applied to a value using two sweeps.

    Each core has axes (left bond, output physical, input physical, right bond).
    The physical dimension is four for a vectorized single-spin matrix.
    """
    width = frame.width
    right_tiles = [None] * len(frame.complements)
    suffix = np.ones((1, 1, 1), complex)
    for j in range(frame.n - 1, width - 2, -1):
        if width == 1:
            right_tiles[j] = suffix.transpose(1, 2, 0).reshape(-1, suffix.shape[0])
        if width == 2 or j > 0:
            tile = _open_right(operators[j], value.tensors[j], suffix)
            if width == 2:
                right_tiles[j-1] = tile.reshape(-1, 4*tile.shape[-1])
            if j >= width:
                suffix = np.tensordot(frame.right.tensors[j].conj(), tile,
                                      axes=((1, 2), (2, 3)))

    coords = np.empty(frame.dimension, complex)
    prefix = np.ones((1, 1, 1), complex)
    for j, (complement, right_tile) in enumerate(zip(frame.complements, right_tiles)):
        operator, core = operators[j], value.tensors[j]
        env = np.tensordot(prefix, core, axes=(2, 0))  # left, Kleft, phys-in, input-right
        tile = np.tensordot(env, operator, axes=((1, 2), (0, 2)))
        left_tile = tile.transpose(0, 2, 3, 1).reshape(4*prefix.shape[0], -1)
        local = left_tile @ right_tile
        coords[frame.offsets[j]:frame.offsets[j+1]] = (complement.conj().T @ local).ravel()
        if j < len(frame.complements)-1:
            left_basis = frame.left.tensors[j].reshape(left_tile.shape[0], -1)
            prefix = (left_basis.conj().T @ left_tile).reshape(
                left_basis.shape[1], operator.shape[-1], core.shape[-1])
    return coords


def preparation_preconditioner(frame, anchor, normal, scale):
    """Return the scaled inverse action, with a fixed HPD preparation anchor.

    `anchor` comes from the same witness engine that supplies A's root
    predictor. Positivity of W follows from the full-rank D/E preparations.
    `normal` is the unit radial coordinate vector at this numerical stage.
    """
    if frame.n != anchor.n:
        raise ValueError('Preparation anchor and tangent chain lengths differ')
    normal = np.array(normal, dtype=complex, copy=True)
    if normal.shape != (frame.dimension,) or not np.isclose(np.linalg.norm(normal), 1.,
                                                            atol=1e-12, rtol=1e-12):
        raise ValueError('Expected a normalized radial coordinate vector')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Expected a finite positive operator scale')
    operators = [_congruence_core(core) for core in anchor.tensors]

    def project(v):
        return v - normal * np.vdot(normal, v)

    def apply(v):
        value = frame.decode(project(v))
        coordinates = encode_action(frame, operators, value)
        return (.5 * scale) * project(coordinates)

    return apply
