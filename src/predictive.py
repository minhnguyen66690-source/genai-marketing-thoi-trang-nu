"""Dự báo liên hệ với bán chạy; không dùng biến rò rỉ nhãn."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from scipy.stats import loguniform, randint
from xgboost import XGBClassifier
from common import PROCESSED, RESULTS, MODELS, FIGURES, SEED, save_json, setup_plot, save_figure, source_text
from preprocess import text_features

CONTROL=["log_price","discount_rate","product_age_days","n_images","n_options"]
CONTENT=["title_len_chars","title_len_words","desc_len_chars","desc_len_words","n_paragraphs","n_bullets","n_sentences","avg_sentence_len_words","n_exclamations","uppercase_ratio","emoji_count","has_size_info","has_material_info","has_care_info","has_cta","has_meta_description","keyword_in_title_first60","has_video","n_specs"]
LABELS={"log_price":"Giá log","discount_rate":"Tỷ lệ giảm giá","product_age_days":"Tuổi sản phẩm","n_images":"Số ảnh","n_options":"Số tùy chọn","title_len_chars":"Độ dài tiêu đề","title_len_words":"Số từ tiêu đề","desc_len_chars":"Độ dài mô tả","desc_len_words":"Số từ mô tả","n_paragraphs":"Số đoạn","n_bullets":"Số gạch đầu dòng","n_sentences":"Số câu","avg_sentence_len_words":"Số từ trung bình mỗi câu","n_exclamations":"Số dấu cảm thán","uppercase_ratio":"Tỷ lệ chữ hoa","emoji_count":"Số biểu tượng","has_size_info":"Thông tin kích thước","has_material_info":"Thông tin chất liệu","has_care_info":"Hướng dẫn bảo quản","has_cta":"Lời kêu gọi","has_meta_description":"Có mô tả meta","keyword_in_title_first60":"Từ khóa trong đầu tiêu đề","has_video":"Có video","n_specs":"Số thông số","category_l2":"Danh mục cấp 2"}

def prepare(df):
    """Chuẩn bị các biến dự báo và danh mục."""
    d=df.copy(); d["log_price"]=np.log1p(pd.to_numeric(d.get("price",pd.Series(index=d.index,dtype=float)),errors="coerce").clip(lower=0))
    for c in CONTROL+CONTENT:
        if c not in d: d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors="coerce")
    if "category_l2" not in d: d["category_l2"]="Không rõ"
    d["category_l2"]=d.category_l2.fillna("Không rõ").astype(str)
    return d

def pipeline(model,cols):
    """Tạo quy trình tiền xử lý và huấn luyện mô hình."""
    numeric=Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler())])
    transform=ColumnTransformer([("num",numeric,cols),("cat",OneHotEncoder(handle_unknown="ignore"),["category_l2"])])
    return Pipeline([("prep",transform),("model",model)])

def score_text_features(df, model_path=MODELS/"best_seller_clf.joblib"):
    """Tính lại đặc trưng văn bản rồi dự báo; cần cả biến kiểm soát ban đầu."""
    bundle=joblib.load(model_path); d=df.copy()
    for i,r in d.iterrows():
        features=text_features(r.get("name",""),r.get("description_text",""),r.get("meta_description",""))
        # Cờ từ khóa cần được cung cấp từ bước keywords.
        features.pop("keyword_in_title_first60",None)
        for k,v in features.items(): d.loc[i,k]=v
    return bundle["model"].predict_proba(prepare(d)[bundle["columns"]])[:,1]

def main():
    """So sánh mô hình, phân tích đóng góp và lưu kết quả."""
    d=pd.read_csv(PROCESSED/"tiki_features.csv"); d=prepare(d); y=pd.to_numeric(d.best_seller,errors="coerce"); counts=y.value_counts(); result={"n_total":len(d),"class_counts":counts.to_dict()}
    # Cần đủ mẫu mỗi lớp cho kiểm tra và CV 5-fold.
    if len(counts)<2 or counts.min()<7 or len(d)<20:
        result.update(status="skipped",warning="Không đủ hai lớp và số mẫu cho chia train/test và CV 5-fold; chưa huấn luyện."); save_json(RESULTS/"predictive_metrics.json",result); print("CẢNH BÁO: " + result["warning"]); return
    x=d[CONTROL+CONTENT+["category_l2"]]; xt,xv,yt,yv=train_test_split(x,y,test_size=.2,random_state=SEED,stratify=y)
    if yt.value_counts().min()<5 or yv.nunique()<2:
        result.update(status="skipped",warning="Tập train/test không đủ lớp cho CV 5-fold hoặc ROC-AUC."); save_json(RESULTS/"predictive_metrics.json",result); print("CẢNH BÁO: "+result["warning"]); return
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    models={"logistic":(LogisticRegression(class_weight="balanced",max_iter=2000,random_state=SEED),{"model__C":loguniform(.001,100)}),"random_forest":(RandomForestClassifier(class_weight="balanced",random_state=SEED,n_jobs=1),{"model__n_estimators":randint(100,401),"model__max_depth":[None,3,5,10,20],"model__min_samples_leaf":randint(1,8),"model__max_features":["sqrt",.7,1.0]}),"xgboost":(XGBClassifier(random_state=SEED,n_jobs=1,eval_metric="logloss"),{"model__n_estimators":randint(100,401),"model__max_depth":randint(2,7),"model__learning_rate":loguniform(.01,.3),"model__subsample":[.7,.85,1.]})}
    models["xgboost"][0].set_params(scale_pos_weight=float((yt == 0).sum() / (yt == 1).sum()))
    result.update(status="ok",n_train=len(xt),n_test=len(xv),class_rate_total=float(y.mean()),class_rate_train=float(yt.mean()),class_rate_test=float(yv.mean()),models={})
    fitted={}; setup_plot(); fig,ax=plt.subplots(figsize=(7,6))
    for name,(est,params) in models.items():
        search=RandomizedSearchCV(pipeline(est,CONTROL+CONTENT),params,n_iter=20,scoring="roc_auc",cv=cv,random_state=SEED,n_jobs=1,refit=True,error_score="raise"); search.fit(xt,yt); pred=search.predict(xv); proba=search.predict_proba(xv)[:,1]; fitted[name]=search
        result["models"][name]={"roc_auc":roc_auc_score(yv,proba),"pr_auc":average_precision_score(yv,proba),"f1":f1_score(yv,pred,zero_division=0),"accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),"recall":recall_score(yv,pred,zero_division=0),"cv_mean":search.best_score_,"cv_std":search.cv_results_["std_test_score"][search.best_index_],"best_params":search.best_params_}
        fpr,tpr,_=roc_curve(yv,proba); ax.plot(fpr,tpr,label=f"{name} (AUC {result['models'][name]['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],"--",color="gray"); ax.set(title="Đường cong ROC trên tập kiểm tra",xlabel="Tỷ lệ dương tính giả",ylabel="Tỷ lệ dương tính thật"); ax.legend(); save_figure(fig,FIGURES/"predictive_01_roc.png",source_text(d))
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, search in fitted.items():
        precision, recall, _ = precision_recall_curve(yv, search.predict_proba(xv)[:, 1])
        ax.plot(recall, precision, label=f"{name} (AP {result['models'][name]['pr_auc']:.3f})")
    ax.set(title="Đường cong precision-recall trên tập kiểm tra", xlabel="Độ bao phủ", ylabel="Độ chính xác")
    ax.legend()
    save_figure(fig, FIGURES / "predictive_04_precision_recall.png", source_text(d))
    best=max(result["models"],key=lambda k:result["models"][k]["cv_mean"]); model=fitted[best].best_estimator_; result["best_model_by_cv_auc"]=best
    # Ablation giữ cùng thuật toán.
    estimator,params=models[best]; ctrl=RandomizedSearchCV(pipeline(estimator,CONTROL),params,n_iter=20,scoring="roc_auc",cv=cv,random_state=SEED,n_jobs=1,error_score="raise"); ctrl.fit(xt,yt)
    auc_ctrl=roc_auc_score(yv,ctrl.predict_proba(xv)[:,1]); result["ablation"]={"control_roc_auc":auc_ctrl,"control_cv_mean":ctrl.best_score_,"control_cv_std":ctrl.cv_results_["std_test_score"][ctrl.best_index_],"control_best_params":ctrl.best_params_,"control_plus_content_roc_auc":result["models"][best]["roc_auc"],"delta_roc_auc":result["models"][best]["roc_auc"]-auc_ctrl,"selection_note":"Chọn thuật toán theo AUC trung bình CV trên tập train."}
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_estimator(model,xv,yv,display_labels=["Khác","Bán chạy"],ax=ax,colorbar=False); ax.set_title("Ma trận nhầm lẫn trên tập kiểm tra"); save_figure(fig,FIGURES/"predictive_02_ma_tran.png",source_text(d))
    imp=permutation_importance(model,xv,yv,n_repeats=10,random_state=SEED,scoring="roc_auc",n_jobs=1); order=np.argsort(imp.importances_mean)[-15:]; fig,ax=plt.subplots(figsize=(9,7)); ax.barh([LABELS.get(x.columns[i],x.columns[i]) for i in order],imp.importances_mean[order]); ax.set(title="Mức quan trọng hoán vị trên tập kiểm tra",xlabel="Mức giảm ROC-AUC"); save_figure(fig,FIGURES/"predictive_03_quan_trong.png",source_text(d))
    result["permutation_importance"]=[{"feature":x.columns[i],"mean_auc_decrease":float(imp.importances_mean[i]),"std":float(imp.importances_std[i])} for i in order[::-1]]
    joblib.dump({"model":model,"columns":list(x.columns),"model_name":best},MODELS/"best_seller_clf.joblib"); save_json(RESULTS/"predictive_metrics.json",result); print(f"Mô hình tốt nhất theo AUC CV: {best}")
if __name__=="__main__": main()
