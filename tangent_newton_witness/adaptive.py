"""Bounded time bisection around any predictor + witness endpoint callback."""
from time import perf_counter

from galerkin_sylvester_tdvp.galerkin_solver import GalerkinError
from galerkin_sylvester_tdvp.evolution import EvolutionError


def advance(engine, previous, start, end, propose, *, max_halvings=3):
    """Return the last certified checkpoint and every attempted interval.

    propose(previous, left, right) returns (candidate, info) and must install
    immutable metric/cache objects by replacement, as WitnessAudit does.
    It must regenerate the independent target and predictor for the interval.
    Intermediate accepted half-steps count as physical checkpoints.
    """
    if not start < end or max_halvings < 0 or int(max_halvings)!=max_halvings:
        raise ValueError('Invalid time interval or halving budget')
    stack=[(start,end,0)]
    state,time=previous,start
    log=[]
    while stack:
        left,right,depth=stack.pop()
        if time!=left:
            raise RuntimeError('Time and witness checkpoint disagree')
        snapshot=vars(engine).copy()
        tic=perf_counter()
        accepted=False
        try:
            candidate,info=propose(state,left,right)
            accepted=bool(info['accepted'])
        except (ArithmeticError,GalerkinError,EvolutionError) as exc:
            info=dict(accepted=False,status=getattr(exc,'status','numerical_failure'),error=str(exc))
        finally:
            if not accepted:
                vars(engine).clear()
                vars(engine).update(snapshot)
        log.append(dict(start=left,end=right,depth=depth,seconds=perf_counter()-tic,info=info))
        if accepted:
            state,time=candidate,right
            continue
        mid=(left+right)/2
        if depth>=max_halvings or not left<mid<right:
            break
        stack.extend([(mid,right,depth+1),(left,mid,depth+1)])
    return state,dict(accepted=time==end,last_accepted_time=time,attempts=log)
