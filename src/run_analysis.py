"""Chạy toàn bộ các bước phân tích theo thứ tự."""
import preprocess, eda, keywords, predictive

def main():
    """Chạy các bước phân tích theo đúng thứ tự phụ thuộc."""
    for module in (preprocess,keywords,eda,predictive):
        print(f"--- {module.__name__} ---",flush=True)
        module.main()
    print("Hoàn tất pipeline.")
if __name__=="__main__": main()
