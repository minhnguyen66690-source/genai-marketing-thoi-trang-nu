"""Aggregate preregistered v2 metrics, keeping denominators explicit."""
import argparse
from collections import defaultdict
import json
import math
import random
import re
import statistics
from pathlib import Path

from holdout_paths import judge_file, results_dir
from metrics_content import email_checks, matches, seo_checks
from objections_v2 import topics_of
from judge_pairwise_v2 import ROOT, METHODS, inputs, manifest, output, read_jsonl, s

def rate(k,n):
    if not n: return {'n':0,'successes':0,'rate':None,'wilson95':None}
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return {'n':n,'successes':k,'rate':p,'wilson95':[center-half,center+half]}

def stats(vals):
    return rate(sum(vals),len(vals))

def valid(out, method):
    fields=('title','description') if method in ('M4a','M4b') else ('title','meta_description','description','sales_script')
    if any(not isinstance(out.get(k),str) or not out[k].strip() for k in fields): return False
    if method in ('M4a','M4b'): return True
    e=out.get('email_subjects')
    return isinstance(e,list) and len(e)==3 and all(isinstance(x,str) and x.strip() for x in e) and len({x.strip() for x in e})==3

def checks(out,brief,method):
    kws=brief.get('keywords') if isinstance(brief.get('keywords'),list) else []
    key=kws[0] if kws and isinstance(kws[0],str) else ''
    title=s(out.get('title')); meta=s(out.get('meta_description')); desc=s(out.get('description'))
    seo=seo_checks(title,meta,desc,key,kws,'vi'); raw=seo['checks']
    c={k:raw[k] for k in ('title_len_50_60','desc_words_ge_200')}
    c['title_starts_keyphrase']=title.strip().casefold().startswith(key.strip().casefold()) if key else (False if not title.strip() else None)
    c['description_keyphrase_2_4']=2 <= seo['keyphrase_count'] <= 4 if key else (False if not desc.strip() else None)
    c['description_bullets_ge3']=len(re.findall(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+\S',desc))>=3
    if method not in ('SELLER','M4a','M4b'): c['meta_len_120_156']=bool(raw['meta_len_120_156'])
    if method not in ('SELLER','M4a','M4b'):
        c['meta_has_keyphrase']=bool(matches(meta,key)) if key else (False if not meta.strip() else None)
        script=s(out.get('sales_script'))
        c['script_words_120_200']=120 <= len(script.split()) <= 200
    first=re.split(r'[.!?\n]',desc,maxsplit=1)[0]
    c['keyphrase_in_first_sentence']=bool(matches(first,key)) if key else (False if not desc.strip() else None)
    if method not in ('SELLER','M4a','M4b'):
        emails=out.get('email_subjects')
        good=isinstance(emails,list) and len(emails)==3 and all(isinstance(x,str) and x.strip() for x in emails) and len({x.strip() for x in emails})==3
        c['email_subjects_valid']=good
        c['email_le9w_le60c']=bool(good and all(email_checks(x,kws,brief)['checks']['mailchimp_le9w_le60c'] for x in emails))
        for metric in ('punct_le3','emoji_le1','no_fake_re_fwd','keyword_relevance'):
            c['email_'+metric]=bool(good and all(email_checks(x,kws,brief)['checks'][metric] for x in emails))
    return c

def sign_p(w,l):
    n=w+l
    return min(1.,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else None

def holm(items):
    ordered=sorted(((v['p_sign'],k) for k,v in items.items() if v['p_sign'] is not None))
    previous=0
    for i,(p,k) in enumerate(ordered):
        previous=max(previous,min(1,p*(len(ordered)-i))); items[k]['p_holm']=previous

def pairwise(root,split,briefs,gens,products,kind):
    # Accept only rows whose content and rubric still match current inputs.
    from judge_pairwise_v2 import build
    all_final={}; report={}
    for judge in ('gemini','gpt'):
        model = __import__('llm_backends').JUDGE_MODEL if judge=='gemini' else 'gpt-5.6-luna'
        expected={x['key'] for x in build(root,split,kind,model)}
        rows={x['key']:x for x in read_jsonl(judge_file(root,split,judge,kind) if split == 'test2' else root/'results_v2/judge'/(judge+'_'+kind+'.jsonl')) if x.get('key') in expected}
        groups=defaultdict(dict)
        for r in rows.values(): groups[(r['item_id'],r['method'])][r['pass']]=r
        finals={}; agg=defaultdict(lambda:defaultdict(list)); disagreements=defaultdict(list)
        for (item,method),passes in groups.items():
            if set(passes)!={1,2}: continue
            a,b=passes[1],passes[2]; other=a['other']
            for criterion in a['result']:
                x,y=a['result'][criterion],b['result'].get(criterion)
                if y is None: continue
                final='win' if x==y==method else 'loss' if x==y==other else 'tie'
                agg[criterion][method].append(final)
                finals[(item,method,criterion)]=final
                disagreements[method].append(x!=y) if criterion=='overall' else None
        all_final[judge]=finals
        summary={}
        for criterion,methods in agg.items():
            summary[criterion]={}
            for method,vals in methods.items():
                w=vals.count('win'); l=vals.count('loss'); t=vals.count('tie')
                summary[criterion][method]={'n':len(vals),'win':w/len(vals),'tie':t/len(vals),'loss':l/len(vals),'win_non_ties':rate(w,w+l),'p_sign':sign_p(w,l)}
            holm(summary[criterion])
        method_count = len(METHODS)
        if split == 'test2':
            method_count = sum(bool(gens[m]) for m in ('M1','M2','M3'))
        expected_pairs=len(briefs)*(method_count if kind=='description' else 1)
        report[judge]={'expected_pairs':expected_pairs,'complete_pairs':sum(1 for p in groups.values() if set(p)=={1,2}),'criteria':summary,'position_disagreement':{m:stats(v) for m,v in disagreements.items()}}
    def agreement(keys):
        pairs=[(all_final['gemini'][k],all_final['gpt'][k]) for k in keys]
        n=len(pairs)
        if not n: return {'n':0,'agreement':None,'kappa':None}
        observed=sum(a==b for a,b in pairs)/n
        expected=sum(sum(a==x for a,b in pairs)*sum(b==x for a,b in pairs) for x in ('win','tie','loss'))/(n*n)
        return {'n':n,'agreement':observed,'kappa':(observed-expected)/(1-expected) if expected<1 else None}
    keys=set(all_final['gemini']) & set(all_final['gpt'])
    keys={k for k in keys if k[2]=='overall'}
    report['agreement']={'all':agreement(keys),'gpt_generated_methods':agreement({k for k in keys if k[1] in ('M1','M2','M2noObj','M2ex','M3')})}
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--split',choices=['dev','test','test2'],required=True); a=p.parse_args()
    root=a.root; manifest(root,__file__,'none',a.split)
    briefs,products,gens=inputs(root,a.split)
    e1={}; e5={}; economy={}
    methods = METHODS
    if a.split == 'test2':
        methods = tuple(method for method in ('M1', 'M2', 'M3') if gens[method])
        if any(set(gens[method]) != set(briefs) for method in (*methods, 'M2', 'M3')):
            p.error('Chưa đủ đầu ra test2.')
    for method in (*methods,'SELLER'):
        values=defaultdict(list); validity=[]; mentions=[]; lat=[]; it=[]; ot=[]
        for item,brief in briefs.items():
            if method=='SELLER':
                if item not in products: continue
                out={'title':s(products[item].get('name')),'description':s(products[item].get('description_text'))}
            else:
                row=gens[method].get(item,{})
                out=output(row); validity.append(valid(out,method))
                for name,target in [('latency_s',lat),('input_tokens',it),('output_tokens',ot)]:
                    if isinstance(row.get(name),(int,float)) and not isinstance(row[name],bool): target.append(row[name])
            for k,v in checks(out,brief,method).items():
                if isinstance(v,bool): values[k].append(v)
            if method in ('M1','M2','M2noObj','M2ex','M3'):
                targets=set(brief.get('objections') or [])
                # Đủ cả hai chủ đề lo ngại của danh mục được nhắc trong kịch bản (từ vựng trung tính topics_of).
                if targets: mentions.append(targets <= set(topics_of(s(out.get('sales_script')))))
        e1[method]={'criteria':{k:stats(v) for k,v in values.items()},'valid_output':stats(validity) if method!='SELLER' else None}
        if method in ('M1','M2','M2noObj','M2ex','M3'): e5[method]={'nhắc chủ đề':stats(mentions)}
        if method!='SELLER': economy[method]={'latency_s_median':statistics.median(lat) if lat else None,'input_tokens_total':sum(it),'output_tokens_total':sum(ot),'token_records':len(it),'latency_records':len(lat)}
    if a.split == 'test2':
        comparison = pairwise(root,a.split,briefs,gens,products,'description')
        for judge in ('gemini','gpt'):
            if comparison[judge]['complete_pairs'] != comparison[judge]['expected_pairs']:
                p.error(f'Chưa đủ cặp chấm test2 cho {judge}.')
        summary = {'split': a.split, 'sample_size': len(briefs), 'E1': e1, 'E2': comparison,
                   'E5': {'nhắc chủ đề': e5}, 'economics': economy}
        dest = results_dir(root,a.split)
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'eval_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'split': a.split, 'sample_size': len(briefs), 'E2': comparison},ensure_ascii=False))
        return
    fc={}; records=[]
    from factcheck_v2 import RUBRIC
    from judge_pairwise_v2 import digest
    for judge in ('gemini','gpt'):
        model=__import__('llm_backends').JUDGE_MODEL if judge=='gemini' else 'gpt-6.1-terra'
        for r in read_jsonl(root/'results_v2/factcheck'/(judge+'_'+a.split+'.jsonl')):
            item=str(r.get('item_id')); method=r.get('method')
            if item not in briefs: continue
            if method=='SELLER':
                if item not in products: continue
                out={'title':s(products[item].get('name')),'description':s(products[item].get('description_text'))}
            elif method in gens and item in gens[method]: out=output(gens[method][item])
            else: continue
            key=digest(json.dumps([model,digest(RUBRIC),method,item,briefs[item],out],sort_keys=True,ensure_ascii=False))
            if r.get('key')==key: records.append(r)
    for method in ('M1','M2','M2noObj','M2ex','M3','M4a','M4b'):
        rows=[r for r in records if r.get('method')==method and str(r.get('item_id')) in briefs]
        # Report each judge independently; missing calls are not counted as clean.
        fc[method]={}
        for judge in ('gemini','gpt'):
            selected={str(r['item_id']):r for r in rows if r.get('judge')==judge}
            counts={label:[sum(c['label']==label for c in r['claims']) for r in selected.values()] for label in ('UNSUPPORTED_ATTRIBUTE','UNSUPPORTED_POLICY')}
            fc[method][judge]={'expected':'mẫu con 40 sản phẩm (seed 42)','checked':len(selected),'discarded_spans':sum(r.get('discarded_spans',0) for r in selected.values()),'types':{k:{'has_claim':stats([n>0 for n in v]),'mean_count':sum(v)/len(v) if v else None} for k,v in counts.items()}}
    sample=[]; rng=random.Random(42)
    for method in ('M1','M2','M2noObj','M2ex','M4b'):
        checked={str(r['item_id']):r for r in records if r.get('method')==method}
        # Chỉ rà tay trong các sản phẩm bộ kiểm tra đã chấm (mẫu con 40 sản phẩm, sai khác #6).
        candidates=sorted(item for item in briefs if item in gens[method] and item in checked)
        for item in rng.sample(candidates,min(12,len(candidates))):
            sample.append({'item_id':item,'method':method,'brief':briefs[item],'output':output(gens[method][item]),'claims':checked[item]['claims'] if item in checked else None})
    dest=root/'results_v2'; dest.mkdir(parents=True,exist_ok=True)
    (dest/'hallucination_audit_sample.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in sample),encoding='utf-8')
    manual=read_jsonl(dest/'hallucination_audit_opus.jsonl')
    # So ở mức ĐẦU RA: mỗi đầu ra có ít nhất một thuộc tính bịa / chính sách bịa hay không.
    # Nhãn tay: {item_id, method, has_attr, has_policy, note}. Bộ kiểm tra chính là Gemini (sai khác #11).
    predictions={(str(r['item_id']),r['method']):r for r in records if r.get('judge')=='gpt'}
    def has(r,label): return any(c['label']==label for c in r['claims'])
    audit={'n_manual':len(manual),'level':'đầu ra','checker':'gpt-6.1-terra','types':{}}
    for label,field in (('UNSUPPORTED_ATTRIBUTE','has_attr'),('UNSUPPORTED_POLICY','has_policy')):
        tp=fp=fn=tn=0
        for row in manual:
            key=(str(row.get('item_id')),row.get('method'))
            if key not in predictions: continue
            g=bool(row.get(field)); p=has(predictions[key],label)
            tp+=g and p; fp+=(not g) and p; fn+=g and not p; tn+=(not g) and not p
        n=tp+fp+fn+tn
        audit['types'][label]={'n':n,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'agreement':(tp+tn)/n if n else None,
                               'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
    # Đồng thuận giữa hai bộ kiểm tra khác hãng trên các đầu ra cả hai cùng chấm.
    gpt={(str(r['item_id']),r['method']):r for r in records if r.get('judge')=='gpt'}
    gem={(str(r['item_id']),r['method']):r for r in records if r.get('judge')=='gemini'}
    both=[k for k in gem if k in gpt]
    audit['gemini_vs_gpt']={label:{'n':len(both),'agreement':(sum(has(gem[k],label)==has(gpt[k],label) for k in both)/len(both)) if both else None,
                                   'gemini_pos':sum(has(gem[k],label) for k in both),'gpt_pos':sum(has(gpt[k],label) for k in both)}
                            for label in ('UNSUPPORTED_ATTRIBUTE','UNSUPPORTED_POLICY')}
    summary={'word_definition':'token tách bằng khoảng trắng','split':a.split,'sample_size':len(briefs),'E1':e1,'E2':pairwise(root,a.split,briefs,gens,products,'description'),'E4':fc,'E4_manual_audit':audit,'E5':{'nhắc chủ đề':e5,'script_judge':pairwise(root,a.split,briefs,gens,products,'script')},'economics':economy,'seller_note':'SELLER chỉ tham khảo; brief không chứa mọi dữ kiện thật; không tính tỷ lệ bịa.','audit_sample_n':len(sample)}
    (dest/'eval_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('method\tvalid/n\ttitle_50_60/n\tdescription_200/n')
    for method,v in e1.items():
        def f(x): return '-' if x is None else str(x['successes'])+'/'+str(x['n'])
        print(method+'\t'+f(v['valid_output'])+'\t'+f(v['criteria'].get('title_len_50_60'))+'\t'+f(v['criteria'].get('desc_words_ge_200')))

if __name__=='__main__': main()
