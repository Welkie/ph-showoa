p='src_obj_baseline/gpu_kernels.py'
t=open(p,encoding='utf-8',newline='').read()
nl='\r\n' if '\r\n' in t else '\n'
t=t.replace('\r\n','\n')
old_better='''        if nr_a < nr_b:
            return True
        if nr_a == nr_b and dist_a < dist_b - 1e-3:
            return True
        return False
'''
new_better='''        # BASELINE objective: scalar Cost = 2000*NV + TD
        return (2000.0 * nr_a + dist_a) < (2000.0 * nr_b + dist_b) - 1e-3
'''
assert old_better in t
t=t.replace(old_better,new_better,1)
old_acc='''        if new_nr < old_nr:
            accepted = True
        elif new_nr == old_nr:
            delta = new_dist - old_dist
            if delta <= 1e-3:
                accepted = True
            else:
                temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
                prob = math.exp(-delta / (1e-6 + temp * abs(old_dist)))
                if rand_u01(rng_states, s) < prob:
                    accepted = True
'''
new_acc='''        # BASELINE acceptance: scalar Cost, P = exp(-dCost / (1e-6 + T*|Cost|))
        old_c = 2000.0 * old_nr + old_dist
        new_c = 2000.0 * new_nr + new_dist
        delta = new_c - old_c
        if delta <= 1e-3:
            accepted = True
        else:
            temp = 1.0 - float(iteration) / float(max_iter) if max_iter > 0 else 0.0
            prob = math.exp(-delta / (1e-6 + temp * abs(old_c)))
            if rand_u01(rng_states, s) < prob:
                accepted = True
'''
assert old_acc in t
t=t.replace(old_acc,new_acc,1)
open(p,'w',encoding='utf-8',newline='').write(t.replace('\n',nl))
print("patched")
