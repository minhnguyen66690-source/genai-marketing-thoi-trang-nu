"""Tổng hợp phần chấm mù của người (E3) và đối chiếu với giám khảo mô hình trên cùng cặp.

Đầu vào: _workspace/08_human_eval/ratings/ratings/*.json (xuất từ trang chấm bằng ArtifactData list out_dir),
_workspace/08_human_eval/key.json (bảng giải mã A/B, không đưa lên trang chấm),
results_v2/judge/gemini_description.jsonl và gpt_description.jsonl.
Đầu ra: results_v2/human_eval_summary.json
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / '_workspace/08_human_eval'


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def sign_p(w, l):
    n = w + l
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else None


def main():
    key = json.loads((H / 'key.json').read_text(encoding='utf-8'))
    ratings = {}
    for f in (H / 'ratings').rglob('*.json'):
        d = json.loads(f.read_text(encoding='utf-8'))
        d = d.get('data', d)
        if d.get('pair'):
            ratings[d['pair']] = d
    # Nhãn cuối của giám khảo mô hình cho cặp (M2, SELLER) trên cùng sản phẩm.
    judge = {}
    for j in ('gemini', 'gpt'):
        passes = defaultdict(dict)
        for r in map(json.loads, open(ROOT / f'results_v2/judge/{j}_description.jsonl', encoding='utf-8')):
            if r['method'] in ('M2', 'M3'):
                passes[(r['item_id'], r['method'])][r['pass']] = r['result']['overall']
        judge[j] = {k: (k[1] if p.get(1) == p.get(2) == k[1] else 'SELLER' if p.get(1) == p.get(2) == 'SELLER' else 'tie')
                    for k, p in passes.items() if len(p) == 2}
    out = {'n_rated': len(ratings), 'comparisons': {}}
    for comp in ('M2 vs SELLER', 'M2 vs M1', 'M3 vs SELLER'):
        target = comp.split()[0]
        res = defaultdict(Counter); wrong = Counter(); agree = Counter()
        for pid, k in key.items():
            if k['comparison'] != comp or pid not in ratings:
                continue
            r = ratings[pid]
            for crit in ('overall', 'informative', 'trustworthy'):
                v = r.get(crit)
                who = k[v] if v in ('A', 'B') else 'tie'
                res[crit]['win' if who == target else 'tie' if who == 'tie' else 'loss'] += 1
            for side in ('A', 'B'):
                if r.get('wrong' + side):
                    wrong[k[side]] += 1
            if comp.endswith('SELLER'):
                v = r.get('overall'); human = k[v] if v in ('A', 'B') else 'tie'
                for j in ('gemini', 'gpt'):
                    if (k['item_id'], target) in judge[j]:
                        agree[j + '_n'] += 1; agree[j + '_same'] += human == judge[j][(k['item_id'], target)]
        summ = {}
        for crit, c in res.items():
            w, l, t = c['win'], c['loss'], c['tie']; n = w + l + t
            summ[crit] = {'n': n, 'win': w, 'tie': t, 'loss': l, 'win_non_ties': w / (w + l) if w + l else None,
                          'wilson95': wilson(w, w + l), 'p_sign': sign_p(w, l)}
        out['comparisons'][comp] = {'criteria': summ, 'flagged_wrong': dict(wrong), 'agreement_with_judges': dict(agree)}
    (ROOT / 'results_v2/human_eval_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
