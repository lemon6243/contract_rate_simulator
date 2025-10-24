import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression 
from scipy.optimize import curve_fit 
import warnings
import io
import platform
from matplotlib import font_manager, rc 
import os 
# import shutil # 오류의 원인이 되는 폰트 캐시 제거 라이브러리 및 로직을 제거했습니다.

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 0. 한글 폰트 설정 (V4.0 - Streamlit Cloud 오류 수정)
# ----------------------------------------------------

def set_korean_font():
    # 폰트 경로 지정 (GitHub/fonts/NanumGothic.ttf 가정)
    font_path_cloud = os.path.join(os.getcwd(), "fonts", "NanumGothic.ttf")
    
    if platform.system() == 'Windows':
        # (Windows 로컬 환경)
        try:
            font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            rc('font', family=font_name)
        except:
            rc('font', family='sans-serif')
    elif platform.system() == 'Darwin':
        # (Mac 로컬 환경)
        rc('font', family='AppleGothic')
    else:
        # Streamlit Cloud (Linux) 환경 강제 적용
        # 폰트 파일 등록 및 설정만 남겨 오류를 해결합니다.
        if os.path.exists(font_path_cloud):
            font_manager.fontManager.addfont(font_path_cloud)
            font_name_nanum = font_manager.FontProperties(fname=font_path_cloud).get_name()
            rc('font', family=font_name_nanum)
        else:
            # 폰트 파일이 없으면 기본 설정으로 fallback
            rc('font', family='sans-serif')
            
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    
# 폰트 설정 함수 호출
set_korean_font()

# ----------------------------------------------------

st.set_page_config(page_title="사용계약률 평가/미래예측 시뮬레이터 (V4.0 - 혼합 평가지표 적용)", layout="wide")
st.title("🏆 도시가스 사용계약률 평가/미래예측 시뮬레이터 (V4.0)")
st.caption("Gap과 절대 계약률을 **혼합**하여 **911점 충족**을 위한 최적 배점 비율을 시뮬레이션합니다.")

# ----------------------------------------------------
# 1. 고정 KPI 및 기본 환경 설정
# ----------------------------------------------------
st.sidebar.header("🎯 평가 목표 및 고정 기준")
st.sidebar.markdown(f"**2026년 평가 시점:** **2026년 12월 예측**")
st.sidebar.markdown(f"**계약 유지 절대 기준:** **911점**")
st.sidebar.markdown("---")

# 2026년 신규 평가 지표 배점 (총 900점)
SCORE_WEIGHTS = {
    "안전점검(실점검률)": 500,
    "중점고객점검(보일러)": 100,
    "상담응대율": 100,
    "상담기여율": 100,
    "고객만족도": 100,
}
PERCENTAGE_COLUMNS = ["안전점검(실점검률)", "중점고객점검(보일러)", "상담응대율", "상담기여율"]
SCORE_COLUMN = "고객만족도" 

OTHER_SCORE_COLUMNS = list(SCORE_WEIGHTS.keys())
OTHER_SCORE_MAX_SUM = sum(SCORE_WEIGHTS.values()) # 900점

GOALS = {
    2026: 90.0
}

st.sidebar.markdown(f"##### 2026년 계약률 목표: **{GOALS[2026]}%**")
st.sidebar.markdown(f"##### 기타 지표 총점 Max: **{OTHER_SCORE_MAX_SUM}점**")
st.sidebar.markdown("---")


# --- 🌟 사용계약률 100점 지표 배점 로직 설정 ---
st.sidebar.subheader("💯 사용계약률 100점 평가지표 구성 (V4.0)")

# 1. 혼합 평가지표 비율 설정
st.sidebar.markdown("##### 1. Gap 및 절대 계약률 비율 설정 (합산 100%)")
gap_ratio = st.sidebar.slider("Gap 평가지표 비율 (%)", min_value=0, max_value=100, value=50, step=10)
abs_rate_ratio = 100 - gap_ratio
st.sidebar.info(f"👉 **Gap 비율: {gap_ratio}%** (100점 중 {gap_ratio}점 반영)")
st.sidebar.info(f"👉 **절대 계약률 비율: {abs_rate_ratio}%** (100점 중 {abs_rate_ratio}점 반영)")
st.sidebar.markdown("---")

# 2. Gap 기준 설정
st.sidebar.markdown("##### 2. Gap 기준 점수 (Max 100점)")
target_goal = st.sidebar.number_input("목표 기준 계약률 (%)", value=GOALS[2026], min_value=90.0, max_value=100.0, step=0.1)
DEFAULT_BINS_STR = "-1, 0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 20.0, 100" 
DEFAULT_LABELS_STR = "100, 97, 94, 88, 85, 82, 79, 76, 73, 70, 0, 0" 
bins_input = st.sidebar.text_input("Gap 경계값 (%, 쉼표 구분)", value=DEFAULT_BINS_STR)
labels_input = st.sidebar.text_input("Gap 부여 점수 (Max 100점)", value=DEFAULT_LABELS_STR)
st.sidebar.markdown("---")

# 3. 절대 계약률 기준 설정
st.sidebar.markdown("##### 3. 절대 계약률 기준 점수 (Max 100점)")
abs_rate_bins_str = st.sidebar.text_area(
    "절대 계약률 경계값 (%, 쉼표 구분, 예: 80, 85, 90, 95)", 
    value="0, 80.0, 85.0, 90.0, 93.0, 100"
)
abs_rate_labels_str = st.sidebar.text_area(
    "절대 계약률 부여 점수 (Max 100점, 경계값-1)", 
    value="70, 80, 85, 95, 100"
)
st.sidebar.markdown("---")


# ----------------------------------------------------
# 2. 데이터 처리 및 예측 함수 (V3.6 - 로지스틱 성장 곡선)
# ----------------------------------------------------

def extract_year_month(s):
    # 텍스트에서 연도와 월 추출
    numbers = re.findall(r'(\d{4}|\d{1,2})', s)
    if not numbers:
        return (0, 0)
    year_part = int(numbers[0])
    if year_part < 100 and year_part > 20: 
        year_part += 2000 
    
    month_part = int(numbers[1]) if len(numbers) > 1 else 0
    return (year_part, month_part)

# 로지스틱 성장 함수 정의
def logistic_func(t, k, t0):
    L = 1.0 
    k = np.clip(k, None, 10.0) 
    return L / (1 + np.exp(-k * (t - t0)))

def predict_rates_logistic(df_rate, rate_cols, future_year):
    years = np.array([extract_year_month(c)[0] + extract_year_month(c)[1]/12 for c in rate_cols])
    future_year_float = future_year + 11/12
    prediction_col_name = f"예측_{future_year}년_12월"
    pred_list = []
    
    df_rate_float = df_rate[rate_cols].copy() / 100 

    for idx, row in df_rate_float.iterrows():
        y = row.values.astype(float)
        x = years.reshape(-1, 1)
        valid_idx = ~np.isnan(y) & (y > 0.001) 
        x_valid, y_valid = x[valid_idx].flatten(), y[valid_idx]
        
        pred_rate_pct = np.nan
        
        if len(y_valid) >= 5: 
            try:
                p0_initial = [0.5, x_valid.mean()]
                popt, pcov = curve_fit(logistic_func, x_valid, y_valid, 
                                     p0=p0_initial, maxfev=5000, 
                                     bounds=((-np.inf, x_valid.min()), (np.inf, x_valid.max() + 5)))
                
                k_opt, t0_opt = popt
                
                pred_raw = logistic_func(future_year_float, k_opt, t0_opt)
                pred_rate_pct = np.clip(pred_raw * 100, 0, 100)
                
            except Exception as e:
                # 선형 회귀로 Fallback (안정성 확보)
                if len(y_valid) >= 2:
                    model = LinearRegression().fit(x_valid.reshape(-1, 1), y_valid)
                    pred_raw_linear = model.predict(np.array([future_year_float]).reshape(-1,1))[0]
                    pred_rate_pct = np.clip(pred_raw_linear * 100, 0, 100)
                elif len(y_valid) == 1:
                    pred_rate_pct = np.clip(y_valid[-1] * 100, 0, 100)
                
        elif len(y_valid) >= 2:
            model = LinearRegression().fit(x_valid.reshape(-1, 1), y_valid)
            pred_raw_linear = model.predict(np.array([future_year_float]).reshape(-1,1))[0]
            pred_rate_pct = np.clip(pred_raw_linear * 100, 0, 100)
        elif len(y_valid) == 1:
            pred_rate_pct = np.clip(y_valid[-1] * 100, 0, 100)
        
        pred_list.append(pred_rate_pct)
        
    pred_df = pd.DataFrame({
        "센터명": df_rate["센터명"],
        prediction_col_name: np.array(pred_list)
    })
    return pred_df.round(2)


# 평가 점수 계산 함수 (V4.0 - 혼합 평가지표 로직 추가)
def calculate_score_2026_v4(data, predicted_col, target_goal, 
                          gap_bins_list, gap_labels_list, 
                          abs_bins_list, abs_labels_list, 
                          gap_ratio, abs_rate_ratio,
                          df_other_score_data, score_weights):
    
    result = data[["센터명", predicted_col]].copy().rename(columns={predicted_col: "예측계약률"})
    
    # --- 900점 기타 지표 점수 계산 및 병합 ---
    DEFAULT_RATE = 95 
    DEFAULT_SCORE = 95
    
    if df_other_score_data is not None and not df_other_score_data.empty:
        result = result.merge(df_other_score_data, on="센터명", how="left")
        
        calculated_scores = []
        for index, row in result.iterrows():
            total_score = 0
            for col in PERCENTAGE_COLUMNS:
                weight = score_weights[col]
                rate = row.get(col)
                if pd.isna(rate):
                    score = weight * (DEFAULT_RATE / 100)
                    result.loc[index, col] = DEFAULT_RATE
                else:
                    rate = np.clip(rate, 0, 100)
                    score = weight * (rate / 100)
                total_score += score
            
            col = SCORE_COLUMN
            score = row.get(col)
            if pd.isna(score):
                score_value = DEFAULT_SCORE
                result.loc[index, col] = DEFAULT_SCORE
            else:
                score_value = np.clip(score, 0, score_weights[col])
            
            total_score += score_value
            calculated_scores.append(round(total_score))

        result["기타 지표 총점 (Max 900점)"] = calculated_scores
        result["기타 지표 총점 (Max 900점)"] = result["기타 지표 총점 (Max 900점)"].astype(int)
        
    else:
        result["기타 지표 총점 (Max 900점)"] = round(OTHER_SCORE_MAX_SUM * (DEFAULT_RATE / 100))
        for col in PERCENTAGE_COLUMNS:
             result[col] = DEFAULT_RATE
        result[SCORE_COLUMN] = DEFAULT_SCORE
    
    
    # --- 1. Gap 기준 점수 산정 (Max 100점) ---
    result["Gap(%)"] = np.clip(target_goal - result["예측계약률"], 0, None).round(2)
    
    # Gap에 따른 100점 만점 점수 부여
    gap_score_raw = pd.cut(
        result["Gap(%)"],
        bins=gap_bins_list,
        labels=gap_labels_list,
        ordered=False 
    ).astype(float).fillna(0)
    
    result["Gap 기준 점수"] = np.clip(gap_score_raw, 0, 100).astype(int)
    
    
    # --- 2. 절대 계약률 기준 점수 산정 (Max 100점) ---
    abs_rate_score_raw = pd.cut(
        result["예측계약률"], 
        bins=abs_bins_list,
        labels=abs_labels_list,
        ordered=True 
    ).astype(float).fillna(0)
    
    result["절대 계약률 기준 점수"] = np.clip(abs_rate_score_raw, 0, 100).astype(int)

    
    # --- 3. 최종 사용계약률 점수 (혼합 로직) ---
    # 최종 점수 = (Gap 기준 점수 * Gap 비율) + (절대 계약률 기준 점수 * 절대 계약률 비율)
    result["사용계약률 점수 (Max 100점)"] = (
        (result["Gap 기준 점수"] * gap_ratio / 100) + 
        (result["절대 계약률 기준 점수"] * abs_rate_ratio / 100)
    ).round(0).astype(int)
        
    # --- 4. 최종 평가 결과 분석 ---
    result["총점 (Max 1000점)"] = result["기타 지표 총점 (Max 900점)"] + result["사용계약률 점수 (Max 100점)"]
    result["911점 도달 여부"] = np.where(result["총점 (Max 1000점)"] >= 911, "✅ 도달", "❌ 위험")
    
    return result

# ----------------------------------------------------
# 3. 데이터 로드 및 실행
# ----------------------------------------------------
st.header("1. 센터별 계약률 데이터 및 기타 지표 점수 입력")

# --- 기타 지표 900점 파일 업로드 양식 제공 ---
st.subheader("1-1. 기타 지표 (900점) 항목별 비율/점수 입력 양식")
template_data = {
    "센터명": ["장안", "서부센터", "동부센터"],
    "안전점검(실점검률)": [91, 95, 98],
    "중점고객점검(보일러)": [95, 90, 95],
    "상담응대율": [98, 100, 99],
    "상담기여율": [92, 98, 97],
    "고객만족도": [96.0, 95.5, 98.2],
}
df_template = pd.DataFrame(template_data)

towrite = io.BytesIO()
df_template.to_excel(towrite, index=False, engine='openpyxl')
towrite.seek(0)

st.download_button(
    "📥 기타 지표 항목별 비율/점수 입력 양식 다운로드 (Excel)", 
    data=towrite.getvalue(), 
    file_name="기타_지표_항목별_비율_점수_입력양식.xlsx", 
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="다운로드 후 4개 항목은 비율(%)을, **고객만족도**는 **실수 점수**를 입력하여 업로드하세요."
)

st.markdown("---")

# --- 기타 지표 900점 파일 업로드 ---
other_score_file = st.file_uploader(
    "1-2. 센터별 기타 지표 총점 (Max 900점) 항목별 비율/점수 파일 업로드", 
    type=["xlsx", "csv"],
    key="other_score_file"
)

df_other_score_rates = None
if other_score_file:
    try:
        if other_score_file.name.endswith('.csv'):
            df_raw_other = pd.read_csv(other_score_file)
        else: 
            df_raw_other = pd.read_excel(other_score_file, sheet_name=0)
        
        df_raw_other.columns = df_raw_other.columns.astype(str).str.strip()
        center_col_other_candidates = [c for c in df_raw_other.columns if "센터" in c or "지점" in c or "구분" in c]
        if not center_col_other_candidates:
             raise ValueError("센터명 컬럼을 찾을 수 없습니다.")
        center_col_other = center_col_other_candidates[0]
        
        df_raw_other["센터명_temp"] = df_raw_other[center_col_other].astype(str).str.strip()
        df_raw_other = df_raw_other[
            ~df_raw_other["센터명_temp"].str.contains('합계|총계', case=False, na=False)
        ].drop(columns=["센터명_temp"])
        
        if df_raw_other.empty:
            st.warning("⚠️ 기타 지표 파일에서 '합계', '총계' 로우를 제외한 후 유효한 센터 데이터가 남아있지 않습니다. 파일을 다시 확인해주세요.")
            df_other_score_rates = None
            st.stop()
        
        cols_to_use = [center_col_other] + OTHER_SCORE_COLUMNS
        cols_in_file = [c for c in cols_to_use if c in df_raw_other.columns] 
        
        df_other_score_rates = df_raw_other[cols_in_file].rename(columns={center_col_other: "센터명"})
        df_other_score_rates["센터명"] = df_other_score_rates["센터명"].astype(str).str.strip()
        for col in OTHER_SCORE_COLUMNS:
            if col in df_other_score_rates.columns:
                df_other_score_rates[col] = pd.to_numeric(df_other_score_rates[col], errors='coerce')
        df_other_score_rates = df_other_score_rates.drop_duplicates(subset=["센터명"]).reset_index(drop=True)
        st.success("✅ 센터별 기타 지표 항목별 비율/점수 로드 및 '합계/총계' 로우 정리 완료.")
    except Exception as e:
        st.error(f"🚨 기타 지표 파일 로드 및 확인 중 오류 발생: {e}")
        df_other_score_rates = None

if df_other_score_rates is None or df_other_score_rates.empty:
    st.info(f"⚠️ 기타 지표 항목별 비율/점수 파일을 업로드해야 정확히 계산됩니다. 현재는 모든 센터에 **전 항목 95%**가 임시 적용되어 **총점 855점**으로 계산됩니다.")

st.markdown("---")

# --- 사용계약률 파일 업로드 ---
st.subheader("1-3. 센터별 누적 사용계약률 데이터")
uploaded_file = st.file_uploader(
    "센터별 누적 사용계약률 엑셀 파일 (.xlsx 또는 .csv) 업로드", 
    type=["xlsx", "csv"],
    key="rate_file"
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else: 
            df_raw = pd.read_excel(uploaded_file, sheet_name=0)
    except Exception as e:
        st.error(f"🚨 사용계약률 파일 로드 중 오류 발생: {e}")
        st.stop()

    df_raw.columns = df_raw.columns.astype(str).str.strip()
    center_candidates = [c for c in df_raw.columns if ("센터" in c or "지점" in c or "구분" in c)]
    col_center = center_candidates[0] if center_candidates else df_raw.columns[0]
    col_center = st.selectbox("센터명/구분 컬럼 선택", df_raw.columns, index=df_raw.columns.get_loc(col_center))
    
    rate_cols_candidates = [c for c in df_raw.columns if c != col_center and re.search(r'\d{4}|\d{2}년|\%', c)]
    if not rate_cols_candidates:
        st.error("🚨 계약률 데이터로 인식할 수 있는 컬럼(예: '24년 02월', '체결률')이 없습니다. 파일 형식을 확인해주세요.")
        st.stop()

    rate_cols_sorted = sorted(rate_cols_candidates, key=extract_year_month)
    latest_col = rate_cols_sorted[-1]
    
    data_analysis = df_raw[[col_center] + rate_cols_sorted].rename(columns={col_center: "센터명"})
    data_analysis["센터명"] = data_analysis["센터명"].astype(str).str.strip()
    
    for c in rate_cols_sorted:
        data_analysis[c] = pd.to_numeric(
            data_analysis[c].astype(str).str.replace("%", "").str.replace(",", ""), 
            errors='coerce'
        )
        data_analysis[c] = np.where(
            (data_analysis[c] > 0.001) & (data_analysis[c] <= 1.00), 
            data_analysis[c] * 100, 
            data_analysis[c]
        )
        data_analysis[c] = np.clip(data_analysis[c].fillna(0), 0, 100)
        
    st.success(f"✅ 사용계약률 데이터 로드 완료. 최신 계약률 컬럼: **{latest_col}**")
    
    
    # ----------------------------------------------------
    # 4. 평가 기준 유효성 검사 및 데이터 준비
    # ----------------------------------------------------
    valid_criteria = True
    try:
        gap_bins_list = [float(x.strip()) for x in bins_input.split(',') if x.strip()]
        gap_labels_list = [float(x.strip()) for x in labels_input.split(',') if x.strip()]
        
        if -1 not in gap_bins_list: gap_bins_list.insert(0, -1)
        gap_bins_list = sorted(list(set(gap_bins_list))) 
        gap_labels_list = [int(l) for l in gap_labels_list] 
        
        if len(gap_bins_list) - 1 != len(gap_labels_list):
            st.sidebar.error("Gap 경계값과 부여 점수의 개수가 일치하지 않습니다. (경계값-1 = 점수 개수)")
            valid_criteria = False
            
        abs_rate_bins_str = abs_rate_bins_str.replace('[', '').replace(']', '') 
        abs_rate_labels_str = abs_rate_labels_str.replace('[', '').replace(']', '')
            
        abs_rate_bins_list = [float(x.strip()) for x in abs_rate_bins_str.split(',') if x.strip()]
        abs_rate_labels_list = [float(x.strip()) for x in abs_rate_labels_str.split(',') if x.strip()]
        
        if len(abs_rate_bins_list) - 1 != len(abs_rate_labels_list):
            st.sidebar.error("절대 계약률 경계값과 부여 점수의 개수가 일치하지 않습니다. (경계값-1 = 점수 개수)")
            valid_criteria = False

        abs_rate_labels_list = [int(l) for l in abs_rate_labels_list]

    except ValueError:
        st.sidebar.error("구간 경계값 또는 부여 점수를 올바른 숫자로 입력해 주세요.")
        valid_criteria = False

    if valid_criteria:
        # ----------------------------------------------------
        # 5. 미래 예측 및 평가 점수 시뮬레이션
        # ----------------------------------------------------
        
        # 5-1. 2026년 12월 예측 (로지스틱 함수 사용)
        target_year = 2026
        pred_df = predict_rates_logistic(data_analysis, rate_cols_sorted, target_year)
        predicted_col_name = f"예측_{target_year}년_12월"
        data_merged = data_analysis.merge(pred_df, on="센터명", how="left")
        
        # 5-2. 2026년 예측치 기반 평가 점수 시뮬레이션 (V4.0 혼합 로직 사용)
        final_score_df = calculate_score_2026_v4(
            data_merged, predicted_col_name, target_goal, 
            gap_bins_list, gap_labels_list, 
            abs_bins_list, abs_labels_list, 
            gap_ratio, abs_rate_ratio,
            df_other_score_rates, SCORE_WEIGHTS
        )
        
        # ----------------------------------------------------
        # 6. 최종 분석 결과 및 시각화
        # ----------------------------------------------------
        st.markdown("---")
        st.header(f"🔍 2026년 12월 예측 평가 시뮬레이션 결과 (V4.0 - 혼합 평가지표 적용)")

        # 6-1. 결과표 (핵심 정보)
        core_cols = (
            ["센터명"] + OTHER_SCORE_COLUMNS + 
            ["기타 지표 총점 (Max 900점)", "예측계약률", "Gap(%)", "Gap 기준 점수", 
             "절대 계약률 기준 점수", "사용계약률 점수 (Max 100점)", 
             "총점 (Max 1000점)", "911점 도달 여부"]
        )
        
        st.subheader("1. 센터별 2026년 12월 예측 계약률 및 평가 점수 종합표")
        
        # Streamlit Dataframe Column Config
        column_config = {
            "안전점검(실점검률)": st.column_config.NumberColumn("안전점검(%)", format="%.2f"),
            "중점고객점검(보일러)": st.column_config.NumberColumn("중점고객(%)", format="%.2f"),
            "상담응대율": st.column_config.NumberColumn("응대율(%)", format="%.2f"),
            "상담기여율": st.column_config.NumberColumn("기여율(%)", format="%.2f"),
            "고객만족도": st.column_config.NumberColumn("만족도(점)", format="%.2f점"),
            "예측계약률": st.column_config.NumberColumn("예측계약률(%)", format="%.2f%%"),
            "Gap(%)": st.column_config.NumberColumn("Gap(%)", format="%.2f%%"),
            "Gap 기준 점수": st.column_config.NumberColumn("Gap 점수", format="%d점"),
            "절대 계약률 기준 점수": st.column_config.NumberColumn("절대율 점수", format="%d점"),
            "사용계약률 점수 (Max 100점)": st.column_config.NumberColumn("계약률 총점", format="%d점"),
            "기타 지표 총점 (Max 900점)": st.column_config.NumberColumn("900점 총점", format="%d점"),
            "총점 (Max 1000점)": st.column_config.NumberColumn("최종 총점", format="%d점"),
        }
        
        valid_centers_df = final_score_df.dropna(subset=OTHER_SCORE_COLUMNS, how='all').copy()
        valid_centers_df = valid_centers_df[
            ~valid_centers_df["센터명"].str.contains('합계|총계', case=False, na=False)
        ]

        st.dataframe(valid_centers_df[core_cols].sort_values("총점 (Max 1000점)", ascending=False).reset_index(drop=True), 
                     use_container_width=True,
                     column_config=column_config)

        # 6-2. 911점 도달 리스크 진단 및 서부센터 기준 분석
        st.subheader("2. 911점 도달 리스크 진단 및 '서부센터' 분석")
        
        # 911점 리스크 진단
        risk_911 = valid_centers_df[valid_centers_df["911점 도달 여부"] == "❌ 위험"].copy()
        risk_count = len(risk_911)
        if risk_count > 0:
            st.error(f"⚠️ **계약 유지 위험 센터 ({risk_count}곳):** 2026년 12월 예측치로 평가했을 때 **911점 미만**입니다.")
            st.dataframe(risk_911[["센터명", "예측계약률", "기타 지표 총점 (Max 900점)", "총점 (Max 1000점)"]].sort_values("총점 (Max 1000점)"))
        else:
            st.success("✅ 2026년 12월 예측치로 평가했을 때 모든 센터가 911점(계약 유지 기준)에 도달 가능합니다.")

        # 서부센터 분석 (911점 충족 목표)
        west_center_df_search = valid_centers_df[valid_centers_df["센터명"].str.contains("서부센터", case=False, na=False)]
        if not west_center_df_search.empty:
            west_center_df = west_center_df_search.iloc[0:1]
            west_total_score = west_center_df["총점 (Max 1000점)"].iloc[0]
            st.markdown(f"#### 🎯 **'서부센터' 911점 충족 분석**")
            st.info(f"**서부센터**의 **예측 총점**은 **{west_total_score}점**입니다. "
                    f"Gap **{gap_ratio}%**와 절대 계약률 **{abs_rate_ratio}%**의 혼합 비율을 조절하여 {west_total_score}점이 911점 이상이 되도록 맞춰보세요.")
        
        # 6-3. 시각화 1: 계약률 예측 추이 vs 2026년 목표
        st.markdown("---")
        st.subheader(f"3. 센터별 계약률 추이 및 {target_year}년 12월 예측 결과 (로지스틱 성장 곡선)")
        
        pred_col_float = target_year + 11/12
        plot_cols_names = rate_cols_sorted + [predicted_col_name]
        
        plot_data_trend = data_merged.set_index("센터명")[plot_cols_names].T
        
        plot_data_trend = plot_data_trend.drop(columns=[c for c in plot_data_trend.columns if '합계' in c.lower() or '총계' in c.lower()], errors='ignore')

        plot_data_trend.index = [extract_year_month(col)[0] + extract_year_month(col)[1]/12 if '예측' not in col else pred_col_float for col in plot_data_trend.index]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in plot_data_trend.columns:
            past_data = plot_data_trend[col].iloc[:-1]
            ax.plot(past_data.index, past_data.values, marker='o', linestyle='-', alpha=0.6, label=col)
            
            # 예측 데이터는 점선
            pred_data = plot_data_trend[col].iloc[-2:]
            ax.plot(pred_data.index, pred_data.values, marker='x', linestyle='--', alpha=0.7, color=ax.lines[-1].get_color())

        # 2026년 목표선
        ax.axhline(y=GOALS[2026], color='red', linestyle=':', linewidth=2, label=f'2026년 KPI 목표 ({GOALS[2026]:.1f}%)')
        
        ax.set_title(f"센터별 누적 사용계약률 추이 및 {target_year}년 12월 예측")
        ax.set_ylabel("사용계약률 (%)")
        ax.set_xlabel("기간")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        x_ticks_labels = rate_cols_sorted + [f"{target_year}년 12월 예측"]
        x_ticks_values = [extract_year_month(col)[0] + extract_year_month(col)[1]/12 for col in rate_cols_sorted] + [pred_col_float]
        ax.set_xticks(x_ticks_values)
        ax.set_xticklabels(x_ticks_labels, rotation=45, ha='right')

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', ncol=3, fontsize='small')
        
        plt.tight_layout()
        st.pyplot(fig)


        # 6-4. 시각화 2: 총점 및 911점 달성 히트맵
        st.markdown("---")
        st.subheader("4. 센터별 총점 및 911점 달성 현황 (히트맵)")
        
        plot_data_heatmap = valid_centers_df[["센터명", "총점 (Max 1000점)"]].set_index("센터명")
        colors = valid_centers_df["911점 도달 여부"].map({"✅ 도달": 'green', "❌ 위험": 'red'}).tolist()

        fig2, ax3 = plt.subplots(figsize=(12, 6))
        
        bars = ax3.bar(plot_data_heatmap.index, plot_data_heatmap["총점 (Max 1000점)"], color=colors)
        
        ax3.axhline(y=911, color='blue', linestyle='--', linewidth=2, label='계약 유지 기준 (911점)')
        
        ax3.set_title(f'센터별 최종 총점 및 911점 달성 현황 (2026년 예측 기반, 혼합 평가지표 반영)')
        ax3.set_ylabel("최종 총점 (Max 1000점)")
        ax3.set_xlabel("고객센터")
        ax3.set_ylim(min(plot_data_heatmap["총점 (Max 1000점)"].min() * 0.9, 800), 1000)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)


        # 6-5. 다운로드
        st.markdown("---")
        st.subheader("5. 결과 데이터 다운로드")
        
        download_df_final = data_merged.merge(valid_centers_df.drop(columns=["예측계약률"]), on="센터명", how="inner")
        
        rate_cols_in_df = [c for c in download_df_final.columns if c in rate_cols_sorted]
        score_cols_to_keep = OTHER_SCORE_COLUMNS + ["기타 지표 총점 (Max 900점)", "Gap(%)", "Gap 기준 점수", "절대 계약률 기준 점수", "사용계약률 점수 (Max 100점)", "총점 (Max 1000점)", "911점 도달 여부"]
        
        final_download_cols = (
            ["센터명"] + rate_cols_in_df + [predicted_col_name] + 
            [c for c in download_df_final.columns if c in OTHER_SCORE_COLUMNS] + 
            [c for c in score_cols_to_keep if c in download_df_final.columns and c not in OTHER_SCORE_COLUMNS] 
        )
        final_download_df = download_df_final[final_download_cols]

        csv = final_download_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, 
                           file_name=f"사용계약률_평가시뮬레이션_V4.0_혼합평가지표_결과.csv", mime="text/csv")
        
        st.markdown(f"""
        ---
        ### 💡 정책적 제언 및 분석 결과 요약
        - **Gap 비율 ({gap_ratio}%)**과 **절대 계약률 비율 ({abs_rate_ratio}%)**을 혼합하여 **사용계약률 점수 (Max 100점)**를 산정했습니다.
        - **'서부센터'**의 **예측 총점**이 **911점**을 충족하는지 확인하면서 **두 비율을 조정**하는 것이 목표 달성을 위한 최적의 방안이 될 것입니다.
        - **Gap 평가지표**는 **노력 대비 성과**를, **절대 계약률 지표**는 **센터의 기반 경쟁력**을 반영하는 데 유리합니다.
        ---
        """)

    else:
        st.info("엑셀 파일을 업로드하고 사이드바의 평가 기준 및 실적을 설정하면 시뮬레이션이 시작됩니다.")
