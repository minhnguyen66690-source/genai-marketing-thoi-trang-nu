"""Extract and verify verbatim claims against product facts."""
import argparse
import json
from pathlib import Path
import llm_backends
from judge_pairwise_v2 import ROOT, append, digest, inputs, manifest, output, read_jsonl, s

RUBRIC = '''Liệt kê MỌI khẳng định trong đầu ra (tên, meta, mô tả, kịch bản, tiêu đề email). Đối chiếu DUY NHẤT với BRIEF. Mỗi khẳng định gồm span trích NGUYÊN VĂN, liên tục từ đầu ra, và đúng một label:
- SUPPORTED: dữ kiện có trong brief (kể cả diễn đạt lại danh mục, cụm từ khóa, thuộc tính, giá).
- SUGGESTION: gợi ý phối đồ, dịp mặc, lời khuyên chọn size chung, hoặc nhận xét chủ quan không kiểm chứng được (đẹp, nữ tính, dễ phối, thoải mái khi diện).
- SERVICE: hành động phục vụ mà người bán tự làm được khi khách hỏi (gửi thêm ảnh, video thật; tư vấn size theo số đo; đo lại thông số; kiểm tra hàng trước khi gói; giải đáp thắc mắc).
- UNSUPPORTED_ATTRIBUTE: dữ kiện vật lý KIỂM CHỨNG ĐƯỢC về sản phẩm mà brief không có: chất liệu, thành phần, màu, kích thước, số đo, số lượng trong bộ, cấu tạo (có mút, có gọng, có lót, khóa kéo...), tính năng (chống nắng, thấm hút, co giãn 4 chiều...), xuất xứ, thương hiệu.
- UNSUPPORTED_POLICY: chính sách bán hàng brief không có: đổi trả, hoàn tiền, bảo hành, phí hoặc thời gian giao hàng, miễn phí vận chuyển, giảm giá, quà tặng, tồn kho, cam kết chất lượng tuyệt đối (100% giống hình, cam kết hoàn tiền).
Trả JSON {"claims":[{"span":"...","label":"..."}]}.'''
LABELS = {'SUPPORTED','SUGGESTION','SERVICE','UNSUPPORTED_ATTRIBUTE','UNSUPPORTED_POLICY'}

def check(brief, out, judge, model):
    fields = {k: out.get(k) for k in ('title','meta_description','description','sales_script','email_subjects')}
    facts = {k: brief[k] for k in ('category','keywords','attributes','price') if k in brief}
    prompt = RUBRIC+'\nBRIEF: '+json.dumps(facts,ensure_ascii=False)+'\nĐẦU RA: '+json.dumps(fields,ensure_ascii=False)
    raw = llm_backends.judge(prompt, model=model, tag='v2_factcheck') if judge == 'gemini' else llm_backends.generate(prompt, model=model, tag='v2_factcheck')
    claims = llm_backends.extract_json(raw)['claims']
    if not isinstance(claims,list): raise ValueError('claims phải là danh sách')
    values = [v for x in fields.values() for v in (x if isinstance(x,list) else [x]) if isinstance(v,str)]
    accepted=[]; discarded=0
    for c in claims:
        if not isinstance(c,dict) or c.get('label') not in LABELS or not isinstance(c.get('span'),str) or not c['span'].strip():
            raise ValueError('Claim sai định dạng')
        if any(c['span'] in value for value in values): accepted.append(c)
        else: discarded+=1
    return accepted,discarded

def verify(claims, out):
    """Giữ khẳng định có câu trích nguyên văn nằm trong chính đầu ra đó (kiểm toán A33)."""
    fields = {k: out.get(k) for k in ('title','meta_description','description','sales_script','email_subjects')}
    values = [v for x in fields.values() for v in (x if isinstance(x,list) else [x]) if isinstance(v,str)]
    accepted=[]; discarded=0
    for c in claims:
        # Khẳng định sai định dạng (nhãn lạ, thiếu câu trích) bị bỏ và đếm, không làm hỏng cả lô.
        if not isinstance(c,dict) or c.get('label') not in LABELS or not isinstance(c.get('span'),str) or not c['span'].strip():
            discarded+=1; continue
        if any(c['span'] in value for value in values): accepted.append(c)
        else: discarded+=1
    return accepted,discarded


def check_batch(jobs, judge, model):
    """Gộp nhiều đầu ra vào một lần gọi; các lượt được xử lý tuần tự.
    Thang nhãn giữ nguyên; mỗi đầu ra có brief riêng và được kiểm câu trích riêng."""
    req=[]
    for n,(key,method,item,brief,out) in enumerate(jobs):
        req.append({'id':str(n),'BRIEF':{k: brief[k] for k in ('category','keywords','attributes','price') if k in brief},
                    'ĐẦU RA':{k: out.get(k) for k in ('title','meta_description','description','sales_script','email_subjects')}})
    prompt=(RUBRIC+'\nCó nhiều mục, mỗi mục có BRIEF và ĐẦU RA riêng; chỉ đối chiếu ĐẦU RA với BRIEF của chính mục đó. '
            'Trả duy nhất JSON {"items":[{"id":"0","claims":[{"span":"...","label":"..."}]}]} đủ mọi id.\n'
            +json.dumps(req,ensure_ascii=False))
    raw = llm_backends.judge(prompt, model=model, tag='v2_factcheck_batch') if judge == 'gemini' else llm_backends.generate(prompt, model=model, tag='v2_factcheck_batch')
    items={str(x['id']):x.get('claims',[]) for x in llm_backends.extract_json(raw)['items']}
    if set(items)!={str(n) for n in range(len(jobs))}: raise ValueError('thiếu id trong lô')
    rows=[]
    for n,(key,method,item,brief,out) in enumerate(jobs):
        claims,discarded=verify(items[str(n)],out)
        rows.append({'key':key,'item_id':item,'method':method,'judge':judge,'model':model,'rubric_hash':digest(RUBRIC),
                     'claims':claims,'discarded_spans':discarded,'batched':len(jobs)})
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--judge',choices=['gemini','gpt'],required=True); p.add_argument('--limit',type=int); p.add_argument('--workers',type=int,default=4); p.add_argument('--sample',type=int); p.add_argument('--batch',type=int,default=1)
    a=p.parse_args()
    if a.limit is not None and a.limit < 1: p.error('limit phải dương')
    model=llm_backends.JUDGE_MODEL if a.judge == 'gemini' else 'gpt-6.1-terra'
    manifest(a.root,__file__,model,a.split)
    briefs,products,gens=inputs(a.root,a.split)
    if a.sample:
        # Mẫu con cố định (seed 42), CÙNG các sản phẩm cho mọi cấu hình để so sánh cặp được (sai khác #6).
        import random
        keep=set(random.Random(42).sample(sorted(briefs),min(a.sample,len(briefs))))
        briefs={k:v for k,v in briefs.items() if k in keep}
    tasks=[]
    # Không chấm người bán: brief không chứa mọi dữ kiện thật nên tỷ lệ bịa của người bán không có nghĩa.
    for method in ('M1','M2','M2noObj','M2ex','M4a','M4b'):
        for item,brief in briefs.items():
            if method == 'SELLER':
                if item not in products: continue
                out={'title':s(products[item].get('name')),'description':s(products[item].get('description_text'))}
            else:
                if item not in gens[method]: continue
                out=output(gens[method][item])
            tasks.append((method,item,brief,out))
    if a.limit: tasks=tasks[:a.limit]
    path=a.root/'results_v2/factcheck'/(a.judge+'_'+a.split+'.jsonl')
    done={x.get('key') for x in read_jsonl(path)}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock=threading.Lock()
    errors=[]; count=0
    pending=[]
    for method,item,brief,out in tasks:
        key=digest(json.dumps([model,digest(RUBRIC),method,item,brief,out],sort_keys=True,ensure_ascii=False))
        if key not in done: pending.append((key,method,item,brief,out))
    def work(job):
        key,method,item,brief,out=job
        claims,discarded=check(brief,out,a.judge,model)
        return {'key':key,'item_id':item,'method':method,'judge':a.judge,'model':model,'rubric_hash':digest(RUBRIC),'claims':claims,'discarded_spans':discarded}
    if a.batch>1:
        batches=[pending[i:i+a.batch] for i in range(0,len(pending),a.batch)]
        def work_batch(b):
            for attempt in range(2):
                try: return check_batch(b,a.judge,model)
                except Exception as exc: last=exc
            raise last
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futures={ex.submit(work_batch,b):b for b in batches}
            for f in as_completed(futures):
                try:
                    for row in f.result():
                        with lock: append(path,row)
                        count+=1
                except Exception as exc: errors.append({'batch':[j[2] for j in futures[f]],'error':str(exc)})
    else:
      with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(work,j):j for j in pending}
        for f in as_completed(futures):
            key,method,item,_,_=futures[f]
            try:
                row=f.result()
                with lock: append(path,row)
                count+=1
            except Exception as exc: errors.append({'item_id':item,'method':method,'error':str(exc)})
    print(json.dumps({'completed':count,'errors':errors},ensure_ascii=False))
    if errors: raise SystemExit(1)

if __name__ == '__main__': main()
