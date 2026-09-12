import streamlit as st
import pandas as pd

from calculator import (
    CellConfig, prb_table, run_simulation, format_console_report,
    plot_resource_grid_heatmap, plot_users_dynamics, plot_mcs_histogram,
)

st.set_page_config(page_title="5G NR Hardware Load Calculator", layout="wide")
st.title("Калькулятор аппаратной нагрузки 5G NR PHY")

# боковая панель
with st.sidebar:
    st.header("Параметры конфигурации")

    duplex = st.selectbox("Режим дуплекса", ["FDD", "TDD"], index=0)
    scs_khz = st.selectbox("SCS (нумерология), кГц", [15, 30, 60], index=1)

    available_bw = sorted({bw for (bw, scs) in prb_table if scs == scs_khz})
    bw_mhz = st.selectbox("Полоса (BW), МГц", available_bw,
                           index=available_bw.index(100) if 100 in available_bw else 0)

    tdd_pattern = "DDDSU"
    if duplex == "TDD":
        tdd_pattern = st.text_input("TDD slot-паттерн (D/U/S)", value="DDDSU")

    num_layers = st.selectbox("MIMO слои", [1, 2], index=0,
                               help="1 = SISO, 2 = 2x2 MIMO")

    st.divider()
    num_sim_ttis = st.slider("Длительность симуляции, TTI", 500, 20000, 6000, step=500)

    st.caption(f"≈ {num_sim_ttis * (1.0/(scs_khz/15)) / 1000:.2f} сек реального времени")

    run_button = st.button("Запустить симуляцию", type="primary", use_container_width=True)

# запуск симуляции
if run_button:
    config = CellConfig(bw_mhz=bw_mhz, scs_khz=scs_khz, duplex=duplex,
                         tdd_pattern=tdd_pattern, num_layers=num_layers)

    with st.spinner(f"Идёт симуляция ({num_sim_ttis} TTI)..."):
        df, metrics = run_simulation(config, num_sim_ttis)

    st.session_state["df"] = df
    st.session_state["metrics"] = metrics
    st.session_state["config"] = config
    st.session_state["report"] = format_console_report(df, metrics, config)

# вывод результатов
if "df" in st.session_state:
    df = st.session_state["df"]
    metrics = st.session_state["metrics"]
    config = st.session_state["config"]

    # верхние карточки-метрики
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cell DL Throughput", f"{metrics['cell_throughput_mbps']:.2f} Мбит/с")
    c2.metric("AXI средняя", f"{metrics['axi_avg_per_tti']:.0f} байт/TTI")
    c3.metric("AXI пиковая", f"{metrics['axi_max_per_tti']:.0f} байт/TTI")
    c4.metric("Записей в БД", f"{metrics['total_records']:,}")

    tab_console, tab_charts, tab_table = st.tabs(["Консоль / метрики", "Графики", "Полная таблица"])

    with tab_console:
        st.text(st.session_state["report"])

    with tab_charts:
        st.subheader("Resource Grid — распределение PRB между UE")
        best_tti = max(metrics["dl_scenario"], key=lambda t: sum(
            int(ue.format_prb.split('x')[0]) for ue in t.users))
        fig1 = plot_resource_grid_heatmap(best_tti, n_size_grid=config.n_size_grid)
        st.pyplot(fig1)

        st.subheader("Динамика активных абонентов")
        fig2 = plot_users_dynamics(df, metrics["num_ttis"], config.tti_duration_ms)
        st.pyplot(fig2)

        st.subheader("Распределение MCS (28 индексов)")
        fig3 = plot_mcs_histogram(df)
        st.pyplot(fig3)

    with tab_table:
        st.subheader("Полная таблица результатов")
        st.dataframe(df, use_container_width=True, height=500)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать simulation_results.csv", data=csv_bytes,
                            file_name="simulation_results.csv", mime="text/csv")
else:
    st.info("Задайте параметры слева и нажмите «Запустить симуляцию»")
