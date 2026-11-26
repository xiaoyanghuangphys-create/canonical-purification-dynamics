import pytest
from parity_local_lift_comparison.benchmark import eligible,config


@pytest.mark.parametrize('update,expected',[
    ({},True),({'status':'timeout'},False),({'status':'memory_limit'},False),
    ({'last_accepted_time':.5},False),({'global_cp_error':.004},False),
    ({'global_cp_error':None},False),({'global_cp_error':float('nan')},False)])
def test_only_complete_raw_global_hits_enter_frontier(update,expected):
    row=dict(config('Half-Petz'),status='completed',last_accepted_time=1.,
             global_cp_error=1e-5,production_seconds=.01)
    row.update(update)
    assert eligible(row,1e-3) is expected
