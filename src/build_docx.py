"""Dựng bài báo Word từ Markdown và cập nhật trường bằng Microsoft Word."""
import argparse
import copy
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from latex2mathml.converter import convert
from lxml import etree
from PIL import Image
from docx.enum.style import WD_STYLE_TYPE
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
XSL = Path(r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL')
WARNINGS = []


def warn(message):
    """Ghi và hiển thị cảnh báo để người dùng rà soát."""
    WARNINGS.append(message)
    print('CẢNH BÁO:', message)


def font(style, size, bold=False, italic=False):
    """Đặt font cho cả chữ Latin, Đông Á và complex script."""
    style.font.name = 'Times New Roman'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    for name in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        fonts.set(qn('w:' + name), 'Times New Roman')


def field(paragraph, instruction):
    """Chèn trường Word có thể cập nhật sau khi mở tài liệu."""
    r = paragraph.add_run()._r
    for typ, value in [('begin', None), ('instr', instruction), ('separate', None), ('text', '0'), ('end', None)]:
        if typ in ('begin', 'separate', 'end'):
            e = OxmlElement('w:fldChar')
            e.set(qn('w:fldCharType'), typ)
        elif typ == 'instr':
            e = OxmlElement('w:instrText')
            e.set(qn('xml:space'), 'preserve')
            e.text = ' ' + value + ' '
        else:
            e = OxmlElement('w:t')
            e.text = value
        r.append(e)


def cover_replace(paragraph, old, new):
    """Thay chuỗi qua các nút text trong cùng đoạn, giữ các run gốc."""
    nodes = paragraph.xpath('.//w:t')
    text = ''.join(n.text or '' for n in nodes)
    start = text.find(old)
    if start < 0:
        return False
    end = start + len(old)
    offset = 0
    inserted = False
    for node in nodes:
        s = node.text or ''
        a, b = offset, offset + len(s)
        if a < end and b > start:
            left = s[:max(0, start-a)]
            right = s[max(0, end-a):] if end < b else ''
            node.text = left + (new if not inserted else '') + right
            node.set(qn('xml:space'), 'preserve')
            inserted = True
        offset = b
    return True


def setup_cover(doc, title):
    """Sửa bìa kể cả đoạn nằm trong hộp văn bản."""
    paragraphs = doc.element.body.xpath('.//w:p')
    changes = [('BÀI TẬP 3', 'BÀI THU HOẠCH CUỐI KỲ'),
               ('KHAI PHÁ DỮ LIỆU NÂNG CAO', 'CHUYÊN ĐỀ NGOẠI KHÓA'),
               ('26D1ADV60900401', 'HKC2026-CDNK-26C1EXT60901501'),
               ('TP. Hồ Chí Minh, ngày 16 tháng 02 năm 2026', 'TP. Hồ Chí Minh, ngày 20 tháng 11 năm 2026')]
    for old, new in changes:
        if not any(cover_replace(p, old, new) for p in paragraphs):
            warn('Không tìm thấy trên bìa: ' + old)
    for p in paragraphs:
        text_value = ''.join(p.xpath('.//w:t/text()'))
        if any(label in text_value.lower() for label in ('giảng viên', 'lớp', 'học viên')):
            cover_replace(p, 'Học Viên:', 'Học viên')
            para = Paragraph(p, doc)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if 'lớp' in text_value.lower():
                for run in para.runs:
                    run.font.size = Pt((run.font.size.pt if run.font.size else 13) - 1)
    for p in paragraphs:
        if 'MÔN HỌC:' in ''.join(p.xpath('.//w:t/text()')):
            new = copy.deepcopy(p)
            runs = new.xpath('.//w:r')
            for run in runs:
                for child in list(run):
                    if child.tag != qn('w:rPr'):
                        run.remove(child)
            if not runs:
                runs = [OxmlElement('w:r')]
                new.append(runs[0])
            text = OxmlElement('w:t')
            text.text = 'ĐỀ TÀI: ' + title
            runs[0].append(text)
            rpr = runs[0].find(qn('w:rPr'))
            if rpr is None:
                rpr = OxmlElement('w:rPr')
                runs[0].insert(0, rpr)
            for tag in ('b', 'sz', 'szCs'):
                el = rpr.find(qn('w:' + tag))
                if el is None:
                    el = OxmlElement('w:' + tag)
                    rpr.append(el)
                if tag != 'b':
                    el.set(qn('w:val'), '28')
            p.addnext(new)
            break
    else:
        warn('Không tìm thấy đoạn MÔN HỌC để chèn đề tài')


def math(paragraph, latex, display=False):
    """Chuyển LaTeX thành OMML; nếu lỗi thì giữ nguyên nguồn."""
    try:
        transform = etree.XSLT(etree.parse(str(XSL)))
        result = transform(etree.fromstring(convert(latex).encode('utf-8')))
        root = result.getroot()
        if display and root.tag == qn('m:oMath'):
            para = OxmlElement('m:oMathPara')
            para.append(root)
            root = para
        elif not display and root.tag == qn('m:oMathPara'):
            root = root.find(qn('m:oMath'))
        if root is None:
            raise ValueError('Không có phần tử OMML')
        paragraph._p.append(root)
    except Exception as exc:
        warn('Không chuyển được công thức ' + latex + ': ' + str(exc))
        paragraph.add_run(('$$' if display else '$') + latex + ('$$' if display else '$'))


def inline(p, text):
    """Thêm chữ đậm, nghiêng và công thức nội dòng."""
    pattern = r'(\$[^$\n]+\$|\*\*.+?\*\*|\*[^*\n]+\*)'
    for part in re.split(pattern, text):
        if not part:
            continue
        if part.startswith('$') and part.endswith('$'):
            math(p, part[1:-1])
        elif part.startswith('**') and part.endswith('**'):
            p.add_run(part[2:-2]).bold = True
        elif part.startswith('*') and part.endswith('*'):
            p.add_run(part[1:-1]).italic = True
        else:
            p.add_run(part)


def centered(p):
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)


def caption(doc, label, text):
    p = doc.add_paragraph()
    centered(p)
    p.add_run(label + ' ').bold = True
    field(p, 'SEQ ' + label + ' \\* ARABIC')
    # Trường nằm trong run riêng nên đặt định dạng đậm trực tiếp.
    p.runs[-1].bold = True
    p.add_run('.').bold = True
    p.add_run(' ' + text)


def source(doc, text):
    p = doc.add_paragraph()
    centered(p)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)


def insert_table(doc, name):
    path = PAPER / 'tables' / (name + '.json')
    if not path.is_file():
        warn('Thiếu bảng: ' + str(path))
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    caption(doc, 'Bảng', data['caption'])
    table = doc.add_table(rows=1, cols=len(data['columns']))
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:' + side)
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:color'), '000000')
        borders.append(border)
    table._tbl.tblPr.append(borders)
    widths = data.get('col_widths_cm')
    if widths:
        table.autofit = False
    for i, value in enumerate(data['columns']):
        table.cell(0, i).text = str(value)
    for row in data['rows']:
        cells = table.add_row().cells
        for i, value in enumerate(row[:len(cells)]):
            cells[i].text = str(value)
    for row_index, row in enumerate(table.rows):
        for i, cell in enumerate(row.cells):
            if widths and i < len(widths):
                cell.width = Cm(widths[i])
            if row_index == 0:
                shading = OxmlElement('w:shd')
                shading.set(qn('w:fill'), 'D9E2F3')
                cell._tc.get_or_add_tcPr().append(shading)
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.bold = row_index == 0
    source(doc, data.get('source', ''))


def render_markdown(doc, path):
    """Đọc các khối Markdown theo thứ tự và dựng nội dung Word."""
    if not path.is_file():
        warn('Thiếu phần: ' + str(path))
        return
    text = path.read_text(encoding='utf-8')
    placeholders = sorted(set(re.findall(r'\{\{[^{}]+\}\}', text)))
    if placeholders:
        warn(f'{path.name} còn biến chưa thay: ' + ', '.join(placeholders))
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('```'):
            block = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Pt(0)
            r = p.add_run('\n'.join(block))
            r.font.name = 'Consolas'
            r.font.size = Pt(10)
            r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), 'Consolas')
            ppr = p._p.get_or_add_pPr()
            shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), 'F2F2F2'); ppr.append(shd)
            borders = OxmlElement('w:pBdr')
            for side in ('top', 'left', 'bottom', 'right'):
                border = OxmlElement('w:' + side)
                for key, value in [('val', 'single'), ('sz', '4'), ('space', '4'), ('color', 'BFBFBF')]:
                    border.set(qn('w:' + key), value)
                borders.append(border)
            ppr.append(borders)
        elif line.startswith('$$'):
            expr = line[2:]
            if expr.endswith('$$'):
                expr = expr[:-2]
            else:
                chunks = [expr]
                while i < len(lines) and '$$' not in lines[i]:
                    chunks.append(lines[i]); i += 1
                if i < len(lines):
                    chunks.append(lines[i].split('$$', 1)[0]); i += 1
                expr = '\n'.join(chunks)
            p = doc.add_paragraph(); centered(p); math(p, expr, True)
        elif match := re.fullmatch(r'\[\[FIG:([^|]+)\|([^|]+)\|(.+)\]\]', line):
            path = ROOT / 'figures' / match[1]
            if path.is_file():
                with Image.open(path) as image:
                    width_cm = min(14.5, 8.5 * image.width / image.height)  # rộng tối đa 14,5 cm, cao tối đa 8,5 cm
                p = doc.add_paragraph(); centered(p); p.add_run().add_picture(str(path), width=Cm(width_cm))
                caption(doc, 'Hình', match[2]); source(doc, match[3])
            else:
                warn('Thiếu hình: ' + str(path))
        elif match := re.fullmatch(r'\[\[TABLE:([^\]]+)\]\]', line):
            insert_table(doc, match[1])
        elif match := re.match(r'^(#{1,3})\s+(.+)$', line):
            p = doc.add_paragraph(style='Heading ' + str(len(match[1])))
            inline(p, match[2])
        elif match := re.match(r'^(-|\d+\.)\s+(.+)$', line):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Pt(0)
            p.add_run(('•' if match[1] == '-' else match[1]) + ' ')
            inline(p, match[2])
        else:
            chunks = [line]
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,3}\s|[-]\s|\d+\.\s|```|\$\$|\[\[(?:FIG|TABLE):)', lines[i].strip()):
                chunks.append(lines[i].strip()); i += 1
            inline(doc.add_paragraph(), ' '.join(chunks))


def build(out, skip_com=False):
    """Dựng DOCX, cập nhật trường và xuất PDF bằng Word."""
    meta = json.loads((PAPER / 'meta.json').read_text(encoding='utf-8'))
    doc = Document(str(ROOT / 'assets' / 'cover_template.docx'))
    setup_cover(doc, meta['title_vi'])
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.left_margin, section.right_margin = Cm(3), Cm(2)
        section.top_margin = section.bottom_margin = Cm(2)
    normal = doc.styles['Normal']
    font(normal, 12)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.first_line_indent = Cm(1)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, bold in [('TOC 1', True), ('TOC 2', False)]:
        style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        font(style, 12, bold, False)
    for level, size in [(1, 13), (2, 12), (3, 12)]:
        style = doc.styles['Heading ' + str(level)]
        font(style, size, True, level == 3)
        style.paragraph_format.first_line_indent = Pt(0)
        if level == 1:
            style.paragraph_format.space_before = Pt(12)
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    section.footer.is_linked_to_previous = False
    footer = section.footer.paragraphs[0]; centered(footer); field(footer, 'PAGE')
    pgnum = OxmlElement('w:pgNumType'); pgnum.set(qn('w:start'), '1'); section._sectPr.append(pgnum)
    def title(text):
        p = doc.add_paragraph(); centered(p); p.add_run(text).bold = True
    title('MỤC LỤC')
    field(doc.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u')
    doc.add_page_break()
    title('DANH MỤC HÌNH'); field(doc.add_paragraph(), 'TOC \\h \\z \\c "Hình"')
    title('DANH MỤC BẢNG'); field(doc.add_paragraph(), 'TOC \\h \\z \\c "Bảng"')
    doc.add_page_break()
    p = doc.add_paragraph(); centered(p); r = p.add_run(meta['title_vi']); r.bold = True; r.font.size = Pt(16)
    title('TÓM TẮT')
    doc.add_paragraph(meta['abstract_vi'])
    p = doc.add_paragraph(); p.add_run('Từ khóa: ' + ', '.join(meta['keywords_vi'])).italic = True
    for part in meta['parts']:
        render_markdown(doc, PAPER / part)
    doc.add_paragraph('TÀI LIỆU THAM KHẢO', style='Heading 1')
    refs = PAPER / meta['references']
    if refs.is_file():
        for line in refs.read_text(encoding='utf-8').splitlines():
            if line.strip():
                p = doc.add_paragraph()
                inline(p, line.strip())  # xử lý *in nghiêng* theo APA
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.left_indent = Cm(1.27)
                p.paragraph_format.first_line_indent = Cm(-1.27)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(3)
                for run in p.runs:
                    run.font.size = Pt(11)
    else:
        warn('Thiếu tài liệu tham khảo: ' + str(refs))
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    if skip_com:
        return
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        word = win32com.client.DispatchEx('Word.Application')
        word.Visible = False
        word.DisplayAlerts = 0
        opened = None
        try:
            opened = word.Documents.Open(str(out.resolve()))
            opened.Fields.Update()
            for i in range(1, opened.TablesOfContents.Count + 1):
                opened.TablesOfContents(i).Update()
            for i in range(1, opened.TablesOfFigures.Count + 1):
                opened.TablesOfFigures(i).Update()
            opened.Fields.Update()
            opened.Repaginate()
            opened.Save()
            opened.ExportAsFixedFormat(str(out.with_suffix('.pdf').resolve()), 17)
            print('Số trang Word:', opened.ComputeStatistics(2))
        finally:
            if opened is not None:
                opened.Close(False)
            word.Quit()
            pythoncom.CoUninitialize()
    except Exception as exc:
        warn('Không cập nhật/xuất PDF bằng Word: ' + str(exc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dựng bài báo khoa học từ Markdown.')
    parser.add_argument('--out', type=Path, default=Path('paper/build/Bai_bao_GenAI_ThoiTrangNu.docx'))
    parser.add_argument('--skip-com', action='store_true', help='Bỏ qua cập nhật bằng Word COM')
    args = parser.parse_args()
    build(args.out.resolve(), skip_com=args.skip_com)
