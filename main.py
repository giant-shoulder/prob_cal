import streamlit as st
import math
from scipy.stats import binom, norm, poisson
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="확률 계산기", layout="wide")
st.title("🎲 확률 계산기")

# --- Tabs
basic_tab, dist_tab, comb_tab, sim_tab, plot_tab = st.tabs([
    "기초 확률", "확률 분포", "조합/순열", "시뮬레이션", "시각화"
])

# --- 1. 기초 확률 계산
def basic_probability():
    st.subheader("단일 사건 성공 확률")
    p = st.number_input("성공 확률 p (0 ~ 1)", min_value=0.0, max_value=1.0, value=0.5)
    k = st.number_input("성공 횟수 k", min_value=1, value=1)
    expected_trials = k / p if p > 0 else float('inf')
    st.write(f"✅ 평균적으로 {expected_trials:.1f}번 시도하면 {k}번 성공합니다.")

# --- 2. 확률 분포 계산기
def probability_distributions():
    st.subheader("확률 분포 선택")
    dist_type = st.selectbox("분포 종류", ["이항분포", "정규분포", "포아송분포"])

    if dist_type == "이항분포":
        n = st.number_input("시행 횟수 n", value=10)
        p = st.number_input("성공 확률 p", min_value=0.0, max_value=1.0, value=0.5)
        k = st.number_input("성공 횟수 k", value=3)
        prob = binom.pmf(k, n, p)
        st.write(f"🎯 P(X = {k}) = {prob:.5f}")

    elif dist_type == "정규분포":
        mu = st.number_input("평균 ", value=0.0)
        sigma = st.number_input("표준편차", value=1.0)
        x1 = st.number_input("하한값 x1", value=-1.0)
        x2 = st.number_input("상한값 x2", value=1.0)
        prob = norm.cdf(x2, mu, sigma) - norm.cdf(x1, mu, sigma)
        st.write(f"📐 P({x1} ≤ X ≤ {x2}) = {prob:.5f}")

    elif dist_type == "포아송분포":
        lam = st.number_input("평균 발생률 λ", value=3.0)
        k = st.number_input("사건 수 k", value=2)
        prob = poisson.pmf(k, lam)
        st.write(f"🧮 P(X = {k}) = {prob:.5f}")

# --- 3. 조합/순열 계산기
def combinatorics():
    st.subheader("조합과 순열 계산기")
    n = st.number_input("전체 수 n", min_value=0, value=5)
    r = st.number_input("선택 수 r", min_value=0, value=3)

    if r <= n:
        st.write(f"🔢 조합 (nCr): {math.comb(n, r)}")
        st.write(f"🔄 순열 (nPr): {math.perm(n, r)}")
    else:
        st.warning("r은 n보다 작거나 같아야 합니다.")

# --- 4. 시뮬레이션
def simulation():
    st.subheader("성공 횟수 시뮬레이션")
    p = st.slider("성공 확률 p", 0.0, 1.0, 0.5)
    trials = st.number_input("시도 횟수", value=1000)
    simulations = st.number_input("시뮬레이션 횟수", value=500)

    results = np.random.binomial(trials, p, int(simulations))
    avg_success = np.mean(results)
    st.write(f"🎯 평균 성공 횟수: {avg_success:.2f}")
    st.bar_chart(np.histogram(results, bins=20)[0])

# --- 5. 시각화
def visualizations():
    st.subheader("확률 밀도 함수 시각화")
    dist_type = st.selectbox("분포 선택", ["정규분포", "이항분포"])

    fig, ax = plt.subplots()
    x = None
    y = None

    if dist_type == "정규분포":
        mu = st.number_input("평균", value=0.0, key='norm_mu')
        sigma = st.number_input("표준편차", value=1.0, key='norm_sigma')
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y)
        ax.set_title("정규분포 PDF")

    elif dist_type == "이항분포":
        n = st.number_input("시행 횟수", value=20, key='binom_n')
        p = st.slider("성공 확률", 0.0, 1.0, 0.5, key='binom_p')
        x = np.arange(n + 1)
        y = binom.pmf(x, n, p)
        ax.bar(x, y)
        ax.set_title("이항분포 PMF")

    st.pyplot(fig)


with basic_tab:
    basic_probability()

with dist_tab:
    probability_distributions()

with comb_tab:
    combinatorics()

with sim_tab:
    simulation()

with plot_tab:
    visualizations()