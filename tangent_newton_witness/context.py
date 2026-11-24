"""Fresh small factor-TDVP controls on the two predecessor benchmark families."""
import argparse
from pathlib import Path
from galerkin_sylvester_tdvp import benchmarks as bm

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    for family in ('D','E'):
        for method in ('factor_explicit','factor_lsqr'):
            c=bm.config(family,3,method,2,4,16,.1 if family=='D' else 1.)
            c.update(memory_mib=768)
            out=args.out/(family+'_'+method)
            out.mkdir()
            bm.save(out/'config.json',c)
            bm.execute(c,out,45)
