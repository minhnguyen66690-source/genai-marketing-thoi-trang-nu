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
    trim_white(OUT / (name + '.png'))
    with Image.open(OUT / (name + '.png')) as im: im.verify()


def trim_white(path, margin=26):
    """Cắt viền trắng thừa quanh nội dung.

    matplotlib luôn vẽ trên khung đầy đủ nên ảnh hay thừa một dải trắng lớn; dải đó khiến
    sơ đồ bị thu nhỏ khi chèn vào bài báo và slide.
    """
    from PIL import ImageChops
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        background = Image.new('RGB', rgb.size, (255, 255, 255))
        box = ImageChops.difference(rgb, background).getbbox()
        if not box:
            return
        left = max(box[0] - margin, 0)
        top = max(box[1] - margin, 0)
        right = min(box[2] + margin, rgb.width)
        bottom = min(box[3] + margin, rgb.height)
        rgb.crop((left, top, right, bottom)).save(path)


def stage(ax, x, y, w, h, number, title, lines, color=TEAL):
    """Một bước của luồng: số thứ tự, tên bước, vài dòng nội dung ngắn."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.4',
                                facecolor='white', edgecolor=color, linewidth=2.2))
    ax.add_patch(FancyBboxPatch((x, y + h - 7.2), w, 7.2, boxstyle='round,pad=0.3,rounding_size=1.4',
                                facecolor=color, edgecolor=color, linewidth=0))
    ax.add_patch(plt.Circle((x + 3.4, y + h - 3.6), 2.1, facecolor='white', edgecolor=color, linewidth=1.6, zorder=3))
    ax.text(x + 3.4, y + h - 3.6, str(number), ha='center', va='center', fontsize=12,
            weight='bold', color=color, zorder=4)
    ax.text(x + 6.0, y + h - 3.6, title, ha='left', va='center', fontsize=12, weight='bold', color='white')
    for i, line in enumerate(lines):
        ax.text(x + 1.8, y + h - 11.0 - i * 4.6, line, ha='left', va='center', fontsize=10.5, color='#1D2B36')


def d1():
    fig, ax = canvas('Hệ thống chạy như thế nào: từ dữ liệu đến nội dung người bán đăng', 5.2)
    w, gap, y, h = 17.4, 2.3, 46, 38
    steps = [
        ('Dữ liệu thật', ['Tiki: 1.941 sản phẩm', 'thời trang nữ', '2.924 đánh giá khách', 'Amazon: 674 sản phẩm'], BLUE),
        ('Tóm tắt hàng', ['Từ khóa khách hay tìm', 'Chất liệu, màu, size', 'Giá bán', 'Lo ngại từ đánh giá thật'], TEAL),
        ('Viết nội dung', ['Mô hình ngôn ngữ viết', 'tên, mô tả, kịch bản', 'nhắn tin và email', 'So sánh 5 cách viết'], TEAL),
        ('Chấm tự động', ['Đối chiếu tiêu chí có', 'nguồn công bố', 'Cảnh báo chi tiết bịa', 'Giám khảo chấm 1–5'], TEAL),
        ('Duyệt và đăng', ['Xem bản nháp và điểm', 'Sửa chỗ chưa đạt', 'Đăng lên sàn, gửi email', 'Làm trên website demo'], ORANGE),
    ]
    for i, (title, lines, color) in enumerate(steps):
        x = 1.9 + i * (w + gap)
        stage(ax, x, y, w, h, i + 1, title, lines, color)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.4, y + h / 2), (x + w + gap - 0.6, y + h / 2))
    # Vòng lặp cải tiến: kết quả chấm quay lại bước viết để sửa câu lệnh.
    x3 = 1.9 + 2 * (w + gap)
    x4 = 1.9 + 3 * (w + gap)
    arrow(ax, (x4 + w / 2, y - 0.6), (x4 + w / 2, y - 9), True)
    arrow(ax, (x4 + w / 2, y - 9), (x3 + w / 2, y - 9), True)
    arrow(ax, (x3 + w / 2, y - 9), (x3 + w / 2, y - 0.6), True)
    ax.text((x3 + x4) / 2 + w / 2, y - 13.5, 'Chấm xong thấy lỗi thì sửa cách viết rồi chạy lại',
            fontsize=12, ha='center', color=BLUE)
    ax.text(1.9, 90, 'Mỗi bước đều để lại số liệu kiểm chứng được; người bán vẫn là người quyết định cuối cùng',
            fontsize=12.5, ha='left', color=GREY, style='italic')
    save(fig, ax, 'diag_01_kien_truc_he_thong')


def d2():
    fig, ax = canvas('Quy trình nghiên cứu theo CRISP-DM', 7)
    data = [
        ('Hiểu bài toán kinh doanh', 'Xác định nhu cầu viết nội dung\nSEO, kịch bản, email'),
        ('Hiểu dữ liệu', 'Thu thập Tiki 2.181 SP,\nAmazon 5.000 bản ghi; EDA'),
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
    fig, ax = canvas('Từ dữ liệu thô đến bộ dữ liệu dùng được: mỗi bước lọc bỏ bao nhiêu sản phẩm', 6.2)
    ax.text(2, 95, 'Tiki (thời trang nữ)', fontsize=13, weight='bold', color=BLUE, va='center')
    ax.text(72, 95, 'Amazon (dữ liệu đối chứng)', fontsize=13, weight='bold', color=BLUE, va='center')
    # Phễu Tiki: độ rộng thanh tỷ lệ với số sản phẩm.
    levels = [(2181, 'Thu về từ Tiki', BLUE), (2021, 'Sản phẩm còn mới', TEAL),
              (1979, 'Có mô tả đủ dài', TEAL), (1941, 'Không trùng lặp', ORANGE)]
    cuts = ['Loại 160: tạo trước 20/11/2019 (quá cũ)',
            'Loại 42: mô tả dưới 50 ký tự (quá ngắn)',
            'Loại 38: trùng cả tên và thương hiệu']
    cx, bh, step, top = 33.0, 12.5, 22.0, 85.0
    for i, (n, label, color) in enumerate(levels):
        w = 62 * n / 2181
        y = top - i * step
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - bh), w, bh, boxstyle='round,pad=0.3,rounding_size=1.4',
                                    facecolor=color, edgecolor=color))
        ax.text(cx, y - bh / 2, f'{n:,} sản phẩm'.replace(',', '.') + chr(10) + f'{label}', ha='center', va='center',
                fontsize=12, weight='bold', color='white', linespacing=1.3)
        if i < 3:
            ym = y - bh - (step - bh) / 2
            arrow(ax, (cx, y - bh - 0.5), (cx, y - step + 0.5))
            ax.text(cx + 3, ym, cuts[i], ha='left', va='center', fontsize=10.5, color=ORANGE, weight='bold')
    ax.text(cx, 1.5, 'Còn lại 1.941 sản phẩm: khóa dữ liệu, ẩn danh khách hàng', ha='center', va='center',
            fontsize=10.5, color=GREY, style='italic')
    # Nhánh Amazon.
    ax.plot([68, 68], [3, 90], color=GREY, linewidth=1, linestyle=':')
    az = [(5000, 'bản ghi ban đầu', BLUE), (2872, 'thuộc nhóm quần áo nữ', TEAL),
          (674, 'có mô tả từ 50 ký tự', ORANGE)]
    for i, (n, label, color) in enumerate(az):
        y = 85 - i * 28
        stage_w = 27
        ax.add_patch(FancyBboxPatch((72, y - 17), stage_w - 0.5, 17, boxstyle='round,pad=0.3,rounding_size=1.4',
                                    facecolor='white', edgecolor=color, linewidth=2.2))
        ax.text(72 + (stage_w - 0.5) / 2, y - 6.5, f'{n:,}'.replace(',', '.'), ha='center', va='center',
                fontsize=18, weight='bold', color=color)
        ax.text(72 + (stage_w - 0.5) / 2, y - 13, label, ha='center', va='center', fontsize=10.5, color='#1D2B36')
        if i < 2:
            arrow(ax, (72 + (stage_w - 0.5) / 2, y - 17.6), (72 + (stage_w - 0.5) / 2, y - 28 + 0.6))
    save(fig, ax, 'diag_03_thu_thap_du_lieu')


def d4():
    fig, ax = canvas('Ba tầng chấm điểm nội dung: từ thước đo cứng đến hiệu quả công việc', 5.3)
    w, gap, y, h = 30.0, 4.0, 20, 62
    tiers = [
        ('Chấm bằng quy tắc', BLUE, ['Chấm: cả 3 loại nội dung', '(mô tả, tiêu đề email, kịch bản)', 'Độ dài, từ khóa, dấu câu theo', 'chuẩn Ahrefs, Yoast, Mailchimp', 'So với bản gốc: ROUGE-L,', 'chrF++, BERTScore'],
         'Ra: tỷ lệ đạt từng tiêu chí'),
        ('Giám khảo AI chấm', TEAL, ['Chấm: chất lượng câu chữ', 'Mô tả: tự nhiên, thuyết phục', 'Email: rõ ràng, gây tò mò', 'Kịch bản: lời kêu gọi, xử lý', 'phản đối, đúng sự thật'],
         'Ra: điểm 1–5 cho mỗi tiêu chí'),
        ('Đo giá trị kinh doanh', ORANGE, ['Chấm: tác động thực tế', 'Đặc trưng của nhóm bán chạy', 'Độ trễ sinh (giây/sản phẩm)', 'Thời gian duyệt so với viết', 'mới (đề xuất đo khi triển khai)'],
         'Ra: số liệu vận hành'),
    ]
    for i, (title, color, lines, out) in enumerate(tiers):
        x = 1.0 + i * (w + gap)
        stage(ax, x, y, w, h, i + 1, title, lines, color)
        ax.add_patch(FancyBboxPatch((x + 1.2, y + 2.2), w - 2.4, 8.5, boxstyle='round,pad=0.2,rounding_size=1.0',
                                    facecolor=color, edgecolor=color, alpha=.15))
        ax.text(x + w / 2, y + 6.5, out, ha='center', va='center', fontsize=11, weight='bold', color=color)
        if i < 2:
            arrow(ax, (x + w + 0.4, y + h / 2), (x + w + gap - 0.6, y + h / 2))
    ax.text(50, 9, 'Cùng một bài viết đi qua cả ba tầng, từ dễ kiểm chứng nhất đến sát thực tế nhất',
            ha='center', va='center', fontsize=12.5, color=GREY, style='italic')
    save(fig, ax, 'diag_04_khung_danh_gia')


def d5():
    fig, ax = canvas('Mỗi cấu hình thêm một lớp hướng dẫn vào câu lệnh (prompt)', 6.2)
    base = ['Vai trò', 'Nhiệm vụ', 'Định dạng JSON', 'Bản tóm tắt']
    extra = ['Quy tắc dữ kiện: không bịa', 'Ràng buộc chuẩn có nguồn', 'Khung AIDA + SPIN']
    shot = ['Ví dụ: 3 SP bán chạy (TF-IDF)']
    cols = [('M1', 'Câu lệnh cơ bản', BLUE, base, 0),
            ('M2', 'Thêm quy tắc', TEAL, base + extra, 4),
            ('M3', 'Thêm ví dụ mẫu', ORANGE, base + extra + shot, 7)]
    w, gap, rh, step = 28.0, 5.0, 7.2, 8.6
    for i, (name, sub, color, rows, n_old) in enumerate(cols):
        x = 2.0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 86), w, 10, boxstyle='round,pad=0.3,rounding_size=1.4',
                                    facecolor=color, edgecolor=color))
        ax.text(x + 2, 91, name, ha='left', va='center', fontsize=15, weight='bold', color='white')
        ax.text(x + 9, 91, sub, ha='left', va='center', fontsize=12, weight='bold', color='white')
        for j, label in enumerate(rows):
            y = 77 - j * step
            old = j < n_old
            ax.add_patch(FancyBboxPatch((x, y), w, rh, boxstyle='round,pad=0.2,rounding_size=1.0',
                                        facecolor='#DCE8F2' if old else color,
                                        edgecolor='#DCE8F2' if old else color))
            ax.text(x + 1.5, y + rh / 2, label if old else '+ ' + label, ha='left', va='center',
                    fontsize=11, color='#1D2B36' if old else 'white', weight='normal' if old else 'bold')
        if i < 2:
            arrow(ax, (x + w + 0.8, 91), (x + w + gap - 0.8, 91))
    ax.add_patch(FancyBboxPatch((2, 1.5), 3, 4.5, boxstyle='round,pad=0.2,rounding_size=0.8', facecolor='#DCE8F2', edgecolor='#DCE8F2'))
    ax.text(6.5, 3.8, 'Giữ nguyên từ cấu hình trước', va='center', fontsize=11.5, color=BLUE)
    ax.add_patch(FancyBboxPatch((36, 1.5), 3, 4.5, boxstyle='round,pad=0.2,rounding_size=0.8', facecolor=BLUE, edgecolor=BLUE))
    ax.text(40.5, 3.8, 'Phần thêm mới (chữ đậm, có dấu +, màu của cột)', va='center', fontsize=11.5, color=BLUE)
    save(fig, ax, 'diag_05_cau_truc_prompt')


def d6():
    fig, ax = canvas('Tinh chỉnh LoRA trên Qwen2.5-0.5B-Instruct', 6.4)
    box(ax,40,86,20,8,'Đầu vào x',BLUE,9)
    box(ax,4,60,37,12,'Trọng số gốc W\n(đóng băng)',BLUE,8)
    box(ax,59,69,30,9,'A (r×d)',TEAL,9)
    box(ax,59,51,30,9,'B (d×r)',TEAL,9)
    ax.text(77, 64.5, 'r = 16, α = 32', ha='left', va='center', fontsize=13, color=BLUE)
    box(ax,43,36,14,9,'+',ORANGE,14,weight='bold')
    box(ax,30,21,40,10,'Đầu ra h = Wx + BAx',BLUE,9)
    for a_,b_ in [((43,86),(23,72)),((57,86),(74,78)),((74,69),(74,60)),((23,60),(43,40)),((74,51),(57,40)),((50,36),(50,31))]: arrow(ax,a_,b_)
    ax.text(50,16.5,'Áp dụng cho q_proj, k_proj, v_proj, o_proj',ha='center',va='center',fontsize=13,color=BLUE)
    ratio=2162688/496195456*100
    ax.text(50,12,f'Tham số huấn luyện: ~{2162688/1e6:.2f} triệu / {496195456/1e6:.0f} triệu ({ratio:.2f}%)'.replace('.',','),ha='center',va='center',fontsize=13,color=BLUE)
    ax.add_patch(FancyBboxPatch((3, 0.5), 94, 7.5, boxstyle='round,pad=0.3,rounding_size=1.2', facecolor='#FBF3D5', edgecolor=YELLOW, linewidth=1.5))
    ax.text(50, 4.2, 'Nói đơn giản: giữ nguyên mô hình gốc, chỉ học thêm một "miếng vá" nhỏ (A và B) rồi cộng vào kết quả.',
            ha='center', va='center', fontsize=12, color='#1D2B36', weight='bold')
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
