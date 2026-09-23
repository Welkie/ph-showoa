from scripts.compare_gpu_reference import compare


def test_compare_does_not_treat_lower_td_with_more_vehicles_as_better():
    ref = {'instances':[{'instance':'Rdp103','nv':13,'td':1382.69}]}
    rows = [{'Dataset':'explicit_rdp103.vrpsdptw','Status':'Success','Best NV':'14','Best TD':'1262.56'}]
    result = compare(rows, ref)['results'][0]
    assert result['delta_td'] < 0
    assert result['delta_scalar_cost'] > 0
    assert not result['scalar_better']
    assert not result['same_nv']


def test_missing_target_is_visible():
    assert compare([], {'instances':[{'instance':'RCdp1001','nv':3,'td':348.98}]})['results'][0]['status'] == 'missing_or_failed'
