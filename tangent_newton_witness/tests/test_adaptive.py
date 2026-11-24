from types import SimpleNamespace
import pytest
from tangent_newton_witness.adaptive import advance


@pytest.mark.parametrize('stop', [False,True])
def test_bisection_restarts_every_object_at_last_accepted_time(stop):
    engine=SimpleNamespace(metric_time=0.,anchor='0.0')
    def propose(state,left,right):
        assert left==state==engine.metric_time and engine.anchor==str(left)
        engine.metric_time=right
        engine.anchor=str(right)
        return right,dict(accepted=right-left<=.25 and (not stop or right<=.5))
    state,info=advance(engine,0.,0.,1.,propose,max_halvings=2)
    assert state==engine.metric_time==info['last_accepted_time']==(.5 if stop else 1.)
    assert info['accepted'] is (not stop)
    assert len(info['attempts'])>=6
