"""Compare notebook summary rows to the user-provided P36 reference image.

Usage: python scripts/compare_gpu_reference.py summary.csv --output comparison.json
This compares reported results, not route feasibility or equivalent timing.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def key(value):
    return Path(value).stem.lower().removeprefix('explicit_')


def compare(rows, reference):
    actual = {key(row['Dataset']): row for row in rows}
    results = []
    for target in reference['instances']:
        row = actual.get(key(target['instance']))
        item = {'instance': target['instance'], 'reference_nv': target['nv'],
                'reference_td': target['td']}
        if row is None or row.get('Status', '').lower() != 'success':
            item['status'] = 'missing_or_failed'
        else:
            nv, td = float(row['Best NV']), float(row['Best TD'])
            if not math.isfinite(nv) or not math.isfinite(td) or nv < 1 or nv != int(nv) or td < 0:
                raise ValueError(f"Invalid reported result for {target['instance']}")
            delta_cost = 2000 * (nv - target['nv']) + td - target['td']
            item.update(status='compared', nv=int(nv), td=td,
                        delta_nv=int(nv)-target['nv'], delta_td=td-target['td'],
                        delta_scalar_cost=delta_cost,
                        same_nv=(int(nv) == target['nv']),
                        within_reference_rounding=(abs(delta_cost) <= .01),
                        scalar_better=(delta_cost < -.01))
        results.append(item)
    return {'reference_population':36, 'candidate_population_expected':32,
            'note':'Reported summary only; verify routes and run configuration separately. '
                   'Timing and best/mean aggregation may differ from the historical image.',
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[1]/'docs/gpu_d88da6e_targets.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding='utf-8-sig', newline='') as stream:
        result = compare(list(csv.DictReader(stream)), json.loads(args.reference.read_text(encoding='utf-8')))
    content = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(content + '\n', encoding='utf-8')
    else:
        print(content)


if __name__ == '__main__':
    main()
