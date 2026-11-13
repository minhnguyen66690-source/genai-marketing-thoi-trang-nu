"""Tạo dữ liệu tổng hợp cho bản demo, không xuất mô tả sản phẩm."""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "webapp" / "data"
ASSETS = ROOT / "webapp" / "assets"
RESULTS = ROOT / "_workspace" / "03_results"


def save(name, data):
    """Ghi JSON UTF-8 có dấu."""
    with open(DATA / name, "w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    """Tổng hợp các nguồn hiện có và bỏ qua nguồn tùy chọn."""
    DATA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    source = json.loads((RESULTS / "keywords_by_category.json").read_text(encoding="utf-8"))
    categories = source["tiki"]["categories"]
    save("categories.json", {name: [item["keyword"] for item in info["top_keywords"][:15]] for name, info in categories.items()})
    objections = RESULTS / "objections_by_category.json"
    if objections.exists():
        save("objections_by_category.json", json.loads(objections.read_text(encoding="utf-8")))
    csv = ROOT / "data" / "processed" / "tiki_features.csv"
    if csv.exists():
        frame = pd.read_csv(csv, usecols=["category_l2", "best_seller", "desc_len_words", "n_bullets", "title_len_chars"])
        frame["best_seller"] = pd.to_numeric(frame["best_seller"], errors="coerce")
        for column in ["desc_len_words", "n_bullets", "title_len_chars"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        groups = frame[frame["best_seller"].isin([0, 1])].groupby(["category_l2", "best_seller"])
        benchmarks = {}
        for (category, flag), group in groups:
            medians = group[["desc_len_words", "n_bullets", "title_len_chars"]].median()
            values = {key: None if pd.isna(value) else float(value) for key, value in medians.items()}
            values["n"] = len(group)
            benchmarks.setdefault(str(category), {})[str(int(flag))] = values
        save("category_benchmarks.json", benchmarks)
    summaries = {}
    for dataset in ("tiki", "amazon"):
        path = RESULTS / f"eval_summary_{dataset}.json"
        if path.exists():
            summaries[dataset] = json.loads(path.read_text(encoding="utf-8"))
    if summaries:
        save("results_summary.json", summaries)
    # Hình và bảng hiển thị ở tab "Kết quả nghiên cứu" (cùng bộ với bài báo).
    for old in ASSETS.glob("*.png"):
        old.unlink()
    figures = {
        "diag_01_kien_truc_he_thong.png": "Kiến trúc hệ thống",
        "diag_07_mo_phong_thuc_nghiem.png": "Sơ đồ mô phỏng thực nghiệm và kết quả từng bước",
        "keywords_01_danh_muc.png": "Từ khóa đặc trưng theo danh mục (dữ liệu Tiki)",
        "eda_08_ty_le_wilson.png": "Tỷ lệ bán chạy theo đặc trưng nội dung",
        "eda_09_chu_de_phan_doi.png": "Chủ đề phản đối trong đánh giá 1–3 sao",
        "res_01_tuong_dong.png": "RQ2: tương đồng với mô tả thật của người bán",
        "res_02_checklist.png": "RQ3: tỷ lệ đạt checklist tối ưu hóa tìm kiếm và email",
        "res_03_giam_khao.png": "RQ4: điểm giám khảo theo tiêu chí",
        "res_04_ro_ri_m4.png": "Vòng lặp prompt và lỗi của mô hình nhỏ",
    }
    for order, (name, caption) in enumerate(figures.items(), 1):
        path = ROOT / "figures" / name
        if path.exists():
            shutil.copyfile(path, ASSETS / f"{order:02d}_{name}")
    captions = {f"{order:02d}_{name}": caption for order, (name, caption) in enumerate(figures.items(), 1)}
    (ASSETS / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=1), encoding="utf-8")
    tables_dir = DATA / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name in ("predictive", "odds_ratios", "similarity", "checklist", "judge", "leakage", "m4_business"):
        source = ROOT / "paper" / "tables" / f"{name}.json"
        if source.exists():
            shutil.copyfile(source, tables_dir / f"{name}.json")
    print("Đã chuẩn bị dữ liệu tổng hợp và hình hiện có.")


if __name__ == "__main__":
    main()
