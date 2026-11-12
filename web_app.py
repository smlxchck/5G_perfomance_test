import streamlit as st
import pandas as pd

from calculator import (
    CellConfig, prb_table, run_simulation, format_console_report,
    plot_resource_grid_heatmap, plot_users_dynamics, plot_mcs_histogram,
    plot_re_level_grid,      # RE-level визуализация одного TTI
    plot_soc_load_diagram,   # SoC-схема: нагрузка блоков через Shared RAM / AXI4
    plot_axi_waterfall,      # накопление нагрузки на шину по конвейеру
    DEFAULT_BLOCK_SPEEDS_GBPS,
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

    # сколько PRB показывать в RE-level визуализации
    st.divider()
    st.subheader("RE-level визуализация")
    re_max_prb = st.slider(
        "Сколько PRB показать в RE-level графике",
        min_value=10, max_value=273, value=50, step=10,
        help="При n_size_grid=273 показывать все — картинка слишком вытянутая. "
             "По умолчанию показываем первые 50 PRB."
    )

    # ёмкость шины и скорость каждого блока конвейера
    st.divider()
    with st.expander("Быстродействие SoC, ГБ/с"):
        st.caption("Ёмкость AXI4-шины и скорость каждого блока. "
                   "По постановке у QAM mapper и Gridmapper она ниже, чем у шины.")
        axi_bus_capacity = st.number_input(
            "Ёмкость шины AXI", min_value=0.5, max_value=100.0,
            value=7.5, step=0.5, help="Пропускная способность AXI4 / Shared RAM")
        block_speeds = {}
        for _stage, _default in DEFAULT_BLOCK_SPEEDS_GBPS.items():
            block_speeds[_stage] = st.number_input(
                _stage, min_value=0.1, max_value=50.0,
                value=float(_default), step=0.5,
                help=f"Быстродействие блока '{_stage}' конвейера")

    run_button = st.button("Запустить симуляцию", type="primary", use_container_width=True)

# запуск симул
if run_button:
    config = CellConfig(bw_mhz=bw_mhz, scs_khz=scs_khz, duplex=duplex,
                         tdd_pattern=tdd_pattern, num_layers=num_layers)

    with st.spinner(f"Идёт симуляция ({num_sim_ttis} TTI)..."):
        df, metrics = run_simulation(config, num_sim_ttis,
                                     axi_bus_capacity_gbps=axi_bus_capacity,
                                     block_speeds_gbps=block_speeds)

    st.session_state["df"] = df
    st.session_state["metrics"] = metrics
    st.session_state["config"] = config
    st.session_state["report"] = format_console_report(df, metrics, config)
    st.session_state["re_max_prb"] = re_max_prb
    # Сбрасываем выбранный TTI - после новой симуляции покажем самый загруженный
    st.session_state.pop("re_selected_tti", None)

# вывод результатов
if "df" in st.session_state:
    df = st.session_state["df"]
    metrics = st.session_state["metrics"]
    config = st.session_state["config"]
    re_max_prb = st.session_state.get("re_max_prb", 50)  

    # верхние карточки метрики
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cell DL Throughput", f"{metrics['cell_throughput_mbps']:.2f} Мбит/с")
    c2.metric("AXI средняя", f"{metrics['axi_avg_per_tti']:.0f} байт/TTI")
    c3.metric("AXI пиковая", f"{metrics['axi_max_per_tti']:.0f} байт/TTI")
    c4.metric("Утилизация AXI (пик)", f"{metrics['bus_util_peak_pct']:.2f} %",
              help=f"От ёмкости шины {metrics['axi_bus_capacity_gbps']} ГБ/с. "
                   f"Средняя: {metrics['bus_util_avg_pct']:.2f} %")
    c5.metric("Записей в БД", f"{metrics['total_records']:,}")

    # HARQ-метрики
    h1_, h2_, h3_, h4_ = st.columns(4)
    h1_.metric("HARQ ретрансмиссий", f"{metrics['harq_retx_total']:,}",
               help="Сколько раз пакеты пришлось передавать повторно (RV1-RV3)")
    h2_.metric("BLER первой передачи", f"{metrics['harq_bler_effective'] * 100:.1f} %",
               help="Доля TB, не декодировавшихся с первой попытки (проектная точка MCS — 10%)")
    h3_.metric("Ретр. на TB (среднее)", f"{metrics['harq_avg_retx_per_tb']:.2f}",
               help=f"Среднее число повторов на транспортный блок; после RV{3} TB отбрасывается")
    h4_.metric("Soft buffer пик (UL)", f"{metrics['harq_buffer_max_bytes']:,} Б",
               help="Пиковый объём мягких бит в буфере HARQ приёмника gNB (UL) — нагрузка на SRAM чипа")

    tab_console, tab_soc, tab_charts, tab_table = st.tabs(
        ["Консоль / метрики", "SoC-схема AXI", "Графики", "Полная таблица"])

    with tab_console:
        st.text(st.session_state["report"])

    with tab_soc:
        st.subheader("SoC 5G Downlink: нагрузка блоков через Shared RAM / AXI4")
        st.caption(
            "Схема из постановки задачи: каждый блок конвейера читает и пишет в Shared RAM "
            "через AXI4-шину. Для каждого блока показаны чтение (Ч) и запись (З) в среднем за TTI "
            "и требуемая скорость против его быстродействия (настраивается в сайдбаре). "
            "Цвет блока — загрузка: зелёный < 50%, жёлтый 50–80%, красный выше."
        )
        st.pyplot(plot_soc_load_diagram(metrics, config))

        st.subheader("Накопление нагрузки по конвейеру")
        st.caption(
            "Сколько байт каждый блок кладёт/забирает из Shared RAM и как с каждым блоком "
            "растёт суммарная нагрузка на шину (среднее за TTI). Справа — управляющий тракт PDCCH."
        )
        st.pyplot(plot_axi_waterfall(metrics))

    with tab_charts:
        st.subheader("Resource Grid — распределение PRB между UE")
        best_tti = max(metrics["dl_scenario"], key=lambda t: sum(
            int(ue.format_prb.split('x')[0]) for ue in t.users))
        fig1 = plot_resource_grid_heatmap(best_tti, n_size_grid=config.n_size_grid)
        st.pyplot(fig1)

        # RE-level визуализация с выбором TTI
        st.subheader("RE-level Grid — детализация до поднесущих и символов")
        st.caption(
            "Каждая клетка = один Resource Element (1 поднесущая × 1 OFDM-символ). "
            "Чёрные столбцы — управляющий канал PDCCH (sym 0–1). "
            "Золотистый столбец — опорный сигнал DM-RS (sym 3). "
            "Цветные блоки — данные абонентов (PDSCH). "
            "Белый пунктир — ретрансмиссия HARQ (в легенде помечена RV)."
        )

        dl_scenario = metrics["dl_scenario"]
        num_ttis = len(dl_scenario)

        # самый загруженный TTI
        best_tti_idx = max(range(num_ttis), key=lambda i: sum(
            int(ue.format_prb.split('x')[0]) for ue in dl_scenario[i].users))

        # Инициализируем выбранное значение в session_state 
        if "re_selected_tti" not in st.session_state:
            st.session_state["re_selected_tti"] = best_tti_idx

        col_a, col_b, col_c = st.columns([5, 2, 1])
        with col_a:
            # Ползунок привязан к session_state ручной ввод его обновляет
            selected_tti = st.slider(
                "Выберите TTI для отображения",
                min_value=0, max_value=num_ttis - 1,
                value=st.session_state["re_selected_tti"], step=1,
                help=f"Всего сымитировано {num_ttis} TTI. "
                     f"По умолчанию показан самый загруженный (#{best_tti_idx})."
            )
            st.session_state["re_selected_tti"] = selected_tti

        with col_b:
            # Ручной ввод номера TTI
            with st.form("tti_input_form", clear_on_submit=False):
                manual_tti = st.number_input(
                    "Или введите номер TTI вручную",
                    min_value=0, max_value=num_ttis - 1,
                    value=st.session_state["re_selected_tti"],
                    step=1, key="manual_tti_input"
                )
                submitted = st.form_submit_button("Перейти", use_container_width=True)
                if submitted:
                    st.session_state["re_selected_tti"] = int(manual_tti)
                    st.rerun()

        with col_c:
            st.metric("Активных UE", len(dl_scenario[st.session_state["re_selected_tti"]].users))

        # Финальное значение для отрисовки
        selected_tti = st.session_state["re_selected_tti"]

        try:
            fig_re = plot_re_level_grid(dl_scenario[selected_tti], config, max_prb=re_max_prb)
            st.pyplot(fig_re)
        except Exception as e:
            st.warning(f"Не удалось построить RE-level график: {e}")

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
