import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

# ---------------------------------------------------------
# [설정] 페이지 스타일
# ---------------------------------------------------------
st.set_page_config(page_title="AI Co-Scientist: Deep Optimization", page_icon="🧬", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .gemini-box { background-color: #1E1E1E; padding: 20px; border-left: 5px solid #8e44ad; border-radius: 10px; margin-top:10px;}
    .metric-box { background-color: #262730; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [0] 리소스 로딩
# ---------------------------------------------------------
class ExpertAI(nn.Module):
    def __init__(self):
        super(ExpertAI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2) 
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_system():
    # 배포 환경과 로컬 환경 경로 호환성 확보
    base = os.getcwd()
    model_path = os.path.join(base, '04_Trained_Model', 'real_model.pth')
    db_path = os.path.join(base, '03_Model_Input', 'real_paper_db.csv')
    
    # 파일이 없으면 None 반환
    if not os.path.exists(model_path) or not os.path.exists(db_path): 
        return None, None, None, None
    
    model = ExpertAI()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.train() # MC Dropout 활성화
    
    sx = joblib.load(os.path.join(base, '04_Trained_Model', 'scaler_X_real.pkl'))
    sy = joblib.load(os.path.join(base, '04_Trained_Model', 'scaler_y_real.pkl'))
    df = pd.read_csv(db_path)
    return model, sx, sy, df

model, sx, sy, df_db = load_system()

# ---------------------------------------------------------
# [1] 핵심 기능 함수들
# ---------------------------------------------------------
def predict_with_uncertainty(X_tensor, n_iter=20):
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            preds.append(model(X_tensor).numpy())
    preds = np.array(preds)
    
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    
    final_mean = sy.inverse_transform(mean_pred)
    scale_factor = sy.data_max_ - sy.data_min_
    final_std = std_pred * scale_factor
    
    return final_mean, final_std

def run_genetic_algorithm(min_temp, max_temp, thickness):
    pop_size = 300
    pop = []
    for _ in range(pop_size):
        r = np.random.rand(4); r /= r.sum()
        t = np.random.randint(min_temp, max_temp)
        pop.append(list(r) + [t])
    
    df_pop = pd.DataFrame(pop, columns=['In','Ga','Zn','Sn','Temp'])
    
    thick_factor_mob = np.log10(thickness + 10) / np.log10(60) 
    thick_factor_stab = 1.0 
    
    for _ in range(10): 
        X = sx.transform(df_pop.values)
        model.eval() 
        with torch.no_grad():
            pred = sy.inverse_transform(model(torch.tensor(X, dtype=torch.float32)).detach().numpy())
        model.train() 
        
        df_pop['Mobility'] = pred[:,0] * thick_factor_mob
        df_pop['Stability'] = pred[:,1] * thick_factor_stab
        df_pop['Score'] = df_pop['Mobility'] - (df_pop['Stability'] * 5)
        
        top = df_pop.sort_values('Score', ascending=False).head(int(pop_size*0.2))
        new_pop = top.values[:,:5].tolist()
        
        while len(new_pop) < pop_size:
            p = top.sample(2).values[:,:5]
            child = (p[0] + p[1]) / 2
            if np.random.rand() < 0.1: 
                child[:4] += np.random.normal(0,0.05,4); child[:4] = np.clip(child[:4],0,1); child[:4] /= child[:4].sum()
                child[4] = np.clip(child[4] + np.random.randint(-20,20), min_temp, max_temp)
            new_pop.append(child)
        df_pop = pd.DataFrame(new_pop, columns=['In','Ga','Zn','Sn','Temp'])
    
    # 점수 재계산
    X_final = sx.transform(df_pop.values)
    model.eval()
    with torch.no_grad():
        pred_final = sy.inverse_transform(model(torch.tensor(X_final, dtype=torch.float32)).detach().numpy())
    model.train()
    
    df_pop['Mobility'] = pred_final[:,0] * thick_factor_mob
    df_pop['Stability'] = pred_final[:,1] * thick_factor_stab
    df_pop['Score'] = df_pop['Mobility'] - (df_pop['Stability'] * 5)
    
    final_res = df_pop.sort_values('Score', ascending=False)
    min_s, max_s = final_res['Score'].min(), final_res['Score'].max()
    final_res['PlotSize'] = 5 + ((final_res['Score'] - min_s) / (max_s - min_s + 1e-9)) * 15
    return final_res

def plot_radar(row):
    mob_s = min(100, (row['Mobility']/80)*100)
    stab_s = min(100, max(0, (1.0-row['Stability'])*100))
    proc_s = 100 if row['Temp'] < 350 else 60
    
    fig = go.Figure(go.Scatterpolar(
        r=[mob_s, stab_s, proc_s, mob_s*0.9, 80, mob_s],
        theta=['Mobility', 'Stability', 'Process', 'Low Power', 'Cost', 'Mobility'],
        fill='toself', name='Candidate'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(l=30, r=30, t=20, b=20))
    return fig

def find_evidence(in_r, ga_r, zn_r, sn_r, temp):
    if df_db.empty: return None
    d = np.sqrt((df_db['In']-in_r)**2 + (df_db['Ga']-ga_r)**2 + (df_db['Sn']-sn_r)**2 + ((df_db['Temp']-temp)/500)**2)
    return df_db.loc[d.nsmallest(1).index].iloc[0]

def ask_gemini(api_key, evidence, u):
    try:
        genai.configure(api_key=api_key)
        # 2.5 Flash 모델 사용
        g_model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        당신은 반도체 분야에서 최고로 권위가 있는 연구원입니다.
        [근거 논문]: {evidence['Paper_ID']} (Mechanism: {evidence['Mechanism']})
        [제안 조건]: In:{u['In']:.2f}, Ga:{u['Ga']:.2f}, Sn:{u['Sn']:.2f}, Temp:{u['Temp']}C, Thickness:{u['Thick']}nm
        
        1. 이 제안이 고성능(이동도)과 고신뢰성(안정성)을 모두 만족하는 이유를 논리적으로 설명하세요.
        2. 특히 사용자가 설정한 두께({u['Thick']}nm)와 온도({u['Temp']}C)가 Flexible AMOLED 공정에 적합한지 구체적으로 평가하세요.
        """
        return g_model.generate_content(prompt).text
    except Exception as e: return f"Error: {e}"

# ---------------------------------------------------------
# [2] UI 구성 (Secrets 연동 부분 수정됨)
# ---------------------------------------------------------
st.title("🧬 AI Co-Scientist: Deep Optimization")
st.markdown("#### Evidence-Based Candidate Discovery (Powered by Genetic Algorithm)")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # [핵심 수정] Secrets에서 먼저 키를 찾고, 없으면 입력창 표시
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key Loaded from Server")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
        st.caption("배포 환경에서는 Secrets가 자동 로드됩니다.")
    
    st.markdown("---")
    st.markdown("**1. Process Constraints**")
    min_temp, max_temp = st.slider("Temp Range (°C)", 100, 500, (200, 350))
    thickness = st.slider("Active Layer Thickness (nm)", 10, 100, 50)
    
    st.markdown("**2. Target Performance**")
    target_mob = st.number_input("Target Mobility (>)", 30.0)

if model is None:
    st.error("데이터 로드 실패. GitHub에 '03_Model_Input'과 '04_Trained_Model' 폴더가 있는지 확인하세요.")
    st.stop()

tab1, tab2 = st.tabs(["🚀 Evolutionary Search", "🔬 Researcher's Lab"])

# === Tab 1: 유전 알고리즘 ===
with tab1:
    if st.button("🚀 Run Genetic Algorithm", type="primary"):
        with st.spinner("Evolving candidates with Physics-aware logic..."):
            res = run_genetic_algorithm(min_temp, max_temp, thickness)
            top3 = res[res['Mobility'] > target_mob].head(3)
            
            if top3.empty: st.warning("조건 만족 후보 없음.")
            else:
                st.success(f"✅ Optimization Complete! (Physics Adjusted for {thickness}nm)")
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    for i in range(len(top3)):
                        row = top3.iloc[i]
                        with st.expander(f"🥇 Rank {i+1}: In-rich IGZTO (Mob: {row['Mobility']:.1f})", expanded=(i==0)):
                            st.plotly_chart(plot_radar(row), use_container_width=True)
                            st.caption(f"In {row['In']:.2f} : Sn {row['Sn']:.2f} @ {row['Temp']:.0f}°C, {thickness}nm")
                with c2:
                    st.subheader("🌌 Search Space")
                    fig = px.scatter_ternary(res.head(300), a="In", b="Ga", c="Sn", color="Mobility", size="PlotSize", color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)

# === Tab 2: 연구자 모드 ===
with tab2:
    c1, c2 = st.columns([1,1])
    with c1:
        in_r = st.slider("In",0.0,1.0,0.4); sn_r = st.slider("Sn",0.0,1.0,0.1)
        temp = st.slider("Temp",100,500,300)
    with c2:
        rem = max(0, 1.0-in_r-sn_r); ga_r = rem*0.3; zn_r = rem*0.7
        st.info(f"Auto-Calc: Ga {ga_r:.2f} / Zn {zn_r:.2f}")
        
        # 불확실성 예측
        X = sx.transform([[in_r, ga_r, zn_r, sn_r, temp]])
        mu, sigma = predict_with_uncertainty(torch.tensor(X, dtype=torch.float32))
        
        # 두께 보정
        thick_factor = np.log10(thickness + 10) / np.log10(60)
        final_mob = mu[0,0] * thick_factor
        final_stab = mu[0,1]
        
        st.metric("Predicted Mobility", f"{final_mob:.1f} ± {sigma[0,0]:.1f}")
        st.metric("Predicted Stability", f"{final_stab:.2f} ± {sigma[0,1]:.2f}")
        
        ev = find_evidence(in_r, ga_r, zn_r, sn_r, temp)
        if api_key and st.button("🧠 Deep Analysis"):
            with st.spinner("Analyzing..."):
                u = {'In':in_r, 'Ga':ga_r, 'Zn':zn_r, 'Sn':sn_r, 'Temp':temp, 'Thick':thickness}
                st.markdown(f"<div class='gemini-box'>{ask_gemini(api_key, ev, u)}</div>", unsafe_allow_html=True)
