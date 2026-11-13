"""Sơ đồ nghiên cứu; chạy `python src/diagrams.py --only D1,D3`."""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures'
BLUE, TEAL, ORANGE, YELLOW, GREY = '#1F4E79', '#2A9D8F', '#E76F51', '#E9C46A', '#8D99AE'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})


def canvas(title, height=7.2):
    fig, ax = plt.subplots(figsize=(13.33, height))
    ax.set_position([.035, .025, .93, .91])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    fig.suptitle(title, fontsize=18, weight='bold', y=.985)
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fs=13, textcolor='white', weight='normal', align='center'):
    text = '\n'.join('\n'.join(textwrap.wrap(line, width=max(8, int(w * 1.15)), break_long_words=False, break_on_hyphens=False)) if line else '' for line in text.split('\n'))
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.2',
                           facecolor=color, edgecolor=color, linewidth=.8)
    ax.add_patch(patch)
    t = ax.text(x+w/2 if align == 'center' else x+1, y+h/2, text, ha=align, va='center', fontsize=max(fs, 13),
                color=textcolor, weight=weight, linespacing=1.0)
    if not hasattr(ax, '_box_text'): ax._box_text = []
    ax._box_text.append((t, (x, y, x+w, y+h)))
    return patch


def arrow(ax, a, b, dashed=False, rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10,
                 linewidth=1.2, color=GREY if dashed else BLUE,
                 linestyle='--' if dashed else '-', connectionstyle=f'arc3,rad={rad}'))


def save(fig, ax, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    for t in ax.texts:
        b = t.get_window_extent(renderer)
        if not (bounds.x0 <= b.x0 and b.x1 <= bounds.x1 and bounds.y0 <= b.y0 and b.y1 <= bounds.y1):
            raise ValueError(f'Text ngoài axes: {name}: {t.get_text()}')
    for t, (x0, y0, x1, y1) in getattr(ax, '_box_text', []):
        b = t.get_window_extent(renderer)
        lo = ax.transData.transform((x0, y0)); hi = ax.transData.transform((x1, y1))
        if not (lo[0] <= b.x0 and b.x1 <= hi[0] and lo[1] <= b.y0 and b.y1 <= hi[1]):
            raise ValueError(f'Text tràn khung: {name}: {t.get_text()}')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(OUT / (name + '.svg'), facecolor='white')
    plt.close(fig)
    with Image.open(OUT / (name + '.png')) as im: im.verify()


def d1():
    fig, ax = canvas('Kiến trúc hệ thống GenAI sinh nội dung marketing', 7.5)
    rows = [
        ('Nguồn dữ liệu', ['Tiki API – Thời trang nữ\n(tiếng Việt)', 'Amazon Reviews 2023 – Clothing\n(tiếng Anh)']),
        ('Xử lý dữ liệu', ['Làm sạch HTML,\nchuẩn hóa Unicode', 'Tách từ tiếng Việt\n(underthesea)', 'Trích từ khóa đặc trưng\n(TF-IDF, YAKE)', 'Trích phản đối từ\nđánh giá 1–3 sao']),
        ('Kho tri thức', ['Tập huấn luyện mô tả\nbán chạy', 'Chỉ mục TF-IDF\nđể truy hồi ví dụ']),
        ('Lớp sinh nội dung', ['M0\nMẫu cố định', 'M1\nZero-shot\n(GPT)', 'M2\nPrompt engineering\n(GPT)', 'M3\nFew-shot có\ntruy hồi (GPT)', 'M4\nQwen2.5-0.5B\n+ LoRA']),
        ('Lớp đánh giá', ['Tương đồng tham chiếu:\nROUGE, chrF++, BERTScore', 'Checklist tối ưu hóa\ntìm kiếm & email (có nguồn)', 'LLM-as-a-judge\n(Gemini)', 'Mô hình dự báo\nsản phẩm bán chạy']),
        ('Ứng dụng TMĐT', ['Web demo Streamlit', 'Người duyệt nội dung\n(human-in-the-loop)', 'Đăng sản phẩm /\ngửi email'])]
    for i, (title, items) in enumerate(rows):
        x = 1 + i*16.55
        box(ax, x, 84, 15, 9, title, BLUE, weight='bold')
        h = 12
        for j, item in enumerate(items):
            box(ax, x, 70-j*12.5, 15, h, item.replace('\n', ' ') if i != 3 else item, TEAL if i != 5 else ORANGE)
        if i < 5: arrow(ax, (x+15.2, 88.5), (x+16.2, 88.5))
    arrow(ax, (74, 32), (74, 10), True)
    arrow(ax, (74, 10), (58, 10), True)
    arrow(ax, (58, 10), (58, 19), True)
    ax.text(66, 5, 'phản hồi để chỉnh prompt', fontsize=13, ha='center', color=BLUE)
    save(fig, ax, 'diag_01_kien_truc_he_thong')


def d2():
    fig, ax = canvas('Quy trình nghiên cứu theo CRISP-DM', 7)
    data = [
        ('Hiểu bài toán kinh doanh', 'Xác định nhu cầu viết nội dung\nSEO, kịch bản, email'),
        ('Hiểu dữ liệu', 'Cào Tiki 3.000 SP, Amazon\n5.000 SP; EDA'),
        ('Chuẩn bị dữ liệu', 'Làm sạch, đặc trưng, từ khóa,\nchia train/val/test'),
        ('Mô hình hóa', 'Phân lớp bán chạy;\nsinh nội dung M0–M4'),
        ('Đánh giá', 'Checklist có nguồn, ROUGE/\nBERTScore, LLM-judge'),
        ('Triển khai', 'Web Streamlit, GitHub, Colab')]
    positions = [(2,59),(35,59),(68,59),(68,18),(35,18),(2,18)]
    for i, ((title, desc), (x,y)) in enumerate(zip(data, positions)):
        box(ax, x, y, 30, 22, '', [BLUE,TEAL,TEAL,BLUE,ORANGE,TEAL][i])
        ax.text(x+15, y+16, f'{i+1}. {title}', ha='center', va='center', fontsize=13, weight='bold', color='white')
        ax.text(x+15, y+7, desc, ha='center', va='center', fontsize=13, color='white')
    for a,b in [((32,70),(35,70)),((65,70),(68,70)),((83,59),(83,40)),((68,29),(65,29)),((35,29),(32,29)),((17,18),(17,5)),((17,5),(1,5)),((1,5),(1,70)),((1,70),(2,70))]: arrow(ax,a,b)
    save(fig, ax, 'diag_02_quy_trinh_crisp_dm')


def d3():
    fig, ax = canvas('Quy trình thu thập và kiểm soát dữ liệu Tiki', 6)
    items = ['Danh mục Thời trang nữ (mã 931)', 'Duyệt danh mục lá',
             'Danh sách sản phẩm\n(sắp xếp bán chạy, mặc định)', 'API chi tiết sản phẩm',
             'API đánh giá\n(600 SP bán chạy nhất, 2 trang)',
             'Lọc: ngày tạo ≥ 20/11/2019;\nmô tả ≥ 50 ký tự; bỏ trùng',
             'Ẩn danh: bỏ tên, ID, ảnh khách hàng', 'Khóa dữ liệu SHA-256 (chỉ đọc)']
    positions = [(2,65),(21,65),(40,65),(59,65),(59,22),(40,22),(21,22),(2,22)]
    for i, (item, (x,y)) in enumerate(zip(items, positions)):
        box(ax, x, y, 17, 20, item, ORANGE if i == 7 else BLUE)
    for a,b in [((19,75),(21,75)),((38,75),(40,75)),((57,75),(59,75)),((67.5,65),(67.5,42)),((59,32),(57,32)),((40,32),(38,32)),((21,32),(19,32))]: arrow(ax,a,b)
    box(ax, 79, 30, 20, 48, 'Giới hạn 1 request/giây\nTôn trọng robots.txt\nDừng khi gặp lỗi\n429/CAPTCHA', YELLOW, 13, BLUE)
    save(fig, ax, 'diag_03_thu_thap_du_lieu')


def d4():
    fig, ax = canvas('Khung đánh giá nội dung đa tầng', 7.5)
    cols = ['Mô tả sản phẩm', 'Tiêu đề email', 'Kịch bản chốt sale']
    for i, c in enumerate(cols): box(ax, 17+i*27.5, 88, 26, 8, c, BLUE, weight='bold')
    labels = [('Chỉ số tự động\ncó nguồn', 38, 48), ('LLM-as-a-judge', 20, 16), ('Chỉ số\nkinh doanh', 2, 16)]
    for label,y,h in labels: box(ax, 1, y, 14, h, label, BLUE, weight='bold')
    texts = [
        ['Tiêu đề 50–60 ký tự\n(Ahrefs)\n\nMeta 120–156 ký tự\n(Yoast)\n\n≥ 200 từ (Yoast, Tiki)\n\nTừ khóa ở tiêu đề,\nđoạn đầu (Yoast)\n\nMật độ 0,5–3% (Yoast)\n\nROUGE-L, chrF++,\nBERTScore',
         '≤ 9 từ, ≤ 60 ký tự\n(Mailchimp)\n\n4–6 từ hoặc 21–40\nký tự (Omnisend)\n\n≤ 3 dấu câu, ≤ 1 emoji\n(Mailchimp)\n\nKhông giả Re:/Fwd:\n(Gmail)',
         'Có lời kêu gọi\nhành động (AIDA)\n\nBao phủ phản đối từ\nđánh giá thật\n\nTỷ lệ thuộc tính\nkhông có căn cứ'],
        ['Tự nhiên, liên quan,\nthuyết phục, nhất quán', 'Rõ ràng, gây tò mò,\nliên quan', 'CTA, nhu cầu–lợi ích\n(SPIN), xử lý phản đối,\nđúng sự thật'],
        ['Đặc trưng của nhóm bán chạy\n(gạch đầu dòng, độ dài tên)', 'Độ trễ sinh\n(giây/sản phẩm)', 'Thời gian duyệt so với\nviết mới (đề xuất đo\nkhi triển khai)']]
    for r, (_,y,h) in enumerate(labels):
        for c in range(3):
            text = '• ' + texts[r][c].replace('\n\n', '\n• ')
            box(ax, 17+c*27.5, y, 26, h, text, ['#EAF4F2','#FFF1EC','#F4F5F7'][r], 13, BLUE, align='left')
    save(fig, ax, 'diag_04_khung_danh_gia')


def d5():
    fig, ax = canvas('Cấu trúc prompt của ba cấu hình M1–M3', 6)
    base = ['Vai trò','Nhiệm vụ','Định dạng JSON','Brief']
    extra = ['Quy tắc dữ kiện:\nkhông bịa','Ràng buộc chuẩn\ncó nguồn','Khung AIDA + SPIN']
    for i, (title, items) in enumerate([('M1',base),('M2',base+extra),('M3',base+extra+['3 ví dụ truy hồi từ SP\nbán chạy (TF-IDF cosine)'])]):
        x=3+i*32.5
        box(ax,x,86,29,8,title,BLUE,weight='bold')
        for j,item in enumerate(items):
            inherited = i > 0 and (j < 4 or (i == 2 and j < 7))
            color = '#BFD7EA' if inherited else ORANGE
            box(ax,x,76-j*9,29,8.25,item,color,13,'black' if inherited else 'white')
    box(ax, 17, 1, 3, 4, '', ORANGE)
    ax.text(21, 3, 'Khối mới so với cấu hình trước', va='center', fontsize=13, color=BLUE)
    box(ax, 59, 1, 3, 4, '', '#BFD7EA')
    ax.text(63, 3, 'Khối kế thừa', va='center', fontsize=13, color=BLUE)
    save(fig, ax, 'diag_05_cau_truc_prompt')


def d6():
    fig, ax = canvas('Tinh chỉnh LoRA trên Qwen2.5-0.5B-Instruct', 5.5)
    box(ax,40,86,20,8,'Đầu vào x',BLUE,9)
    box(ax,4,57,37,12,'Trọng số gốc W\n(đóng băng)',BLUE,8)
    box(ax,59,67,30,9,'A (r×d)',TEAL,9)
    box(ax,59,48,30,9,'B (d×r)',TEAL,9)
    ax.text(74, 61.5, 'r = 16, α = 32', ha='center', va='center', fontsize=13, color=BLUE)
    box(ax,43,30,14,9,'+',ORANGE,14,weight='bold')
    box(ax,30,10,40,10,'Đầu ra h = Wx + BAx',BLUE,9)
    for a,b in [((43,86),(23,69)),((57,86),(74,76)),((74,67),(74,57)),((23,57),(43,35)),((74,48),(57,35)),((50,30),(50,20))]: arrow(ax,a,b)
    ax.text(50,6,'q_proj, k_proj, v_proj, o_proj',ha='center',va='center',fontsize=13,color=BLUE)
    ratio=2162688/496195456*100
    ax.text(50,2,f'Tham số huấn luyện: ~{2162688/1e6:.2f} triệu / {496195456/1e6:.0f} triệu ({ratio:.2f}%)'.replace('.',','),ha='center',va='center',fontsize=13,color=BLUE)
    save(fig, ax, 'diag_06_lora')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only', help='Danh sách sơ đồ, ví dụ D1,D3')
    args=parser.parse_args()
    functions={'D1':d1,'D2':d2,'D3':d3,'D4':d4,'D5':d5,'D6':d6}
    selected = [s.strip().upper() for s in args.only.split(',')] if args.only else list(functions)
    unknown = set(selected)-functions.keys()
    if unknown: parser.error('Sơ đồ không hợp lệ: '+', '.join(sorted(unknown)))
    for key in selected: functions[key]()


if __name__ == '__main__': main()
