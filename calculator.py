import py3gpp
import numpy as np
import math
import random
import pandas as pd
from dataclasses import dataclass, field
from typing import List

# Полная таблица MCS 256QAM (Модуляция, R x 1024, Qm) TS 38.214 
mcs_table = {
    0:  ('qpsk',   120,   2), 1:  ('qpsk',   193,   2), 2:  ('qpsk',   308,   2),
    3:  ('qpsk',   449,   2), 4:  ('qpsk',   602,   2), 5:  ('qam16',  378,   4),
    6:  ('qam16',  434,   4), 7:  ('qam16',  490,   4), 8:  ('qam16',  553,   4),
    9:  ('qam16',  616,   4), 10: ('qam16',  658,   4), 11: ('qam64',  466,   6),
    12: ('qam64',  517,   6), 13: ('qam64',  567,   6), 14: ('qam64',  616,   6),
    15: ('qam64',  666,   6), 16: ('qam64',  719,   6), 17: ('qam64',  772,   6),
    18: ('qam64',  822,   6), 19: ('qam64',  873,   6), 20: ('qam256', 682.5, 8),
    21: ('qam256', 711,   8), 22: ('qam256', 754,   8), 23: ('qam256', 797,   8),
    24: ('qam256', 841,   8), 25: ('qam256', 885,   8), 26: ('qam256', 916.5, 8),
    27: ('qam256', 948,   8),
}

def _build_mcs_snr_thresholds(mcs_table, margin_db=3.0):
    # Требуемый SNR по формуле Шеннона
    thresholds = []
    for idx, (_, r_x1024, qm) in mcs_table.items():
        eff = qm * (r_x1024 / 1024.0)  # спектральная эффективность
        snr_req = 10 * math.log10(2 ** eff - 1) + margin_db
        thresholds.append((snr_req, idx))
    thresholds.sort()
    return thresholds

MCS_SNR_THRESHOLDS = _build_mcs_snr_thresholds(mcs_table)

def snr_to_mcs(snr, thresholds=MCS_SNR_THRESHOLDS):
    mcs = thresholds[0][1]
    for req_snr, idx in thresholds:
        if snr >= req_snr:
            mcs = idx
        else:
            break
    return mcs

# Таблица PRB по (BW МГц, SCS кГц) TS 38.101-1 Табл. 5.3.2-1 
prb_table = {
    (5,15):25,  (10,15):52,  (15,15):79,  (20,15):106, (25,15):133, (40,15):216, (50,15):270,
    (10,30):24, (15,30):38,  (20,30):51,  (25,30):65,  (30,30):78,  (40,30):106, (50,30):133,
    (60,30):162,(80,30):217, (100,30):273,
    (10,60):11, (20,60):24,  (30,60):38,  (40,60):51,  (50,60):65,  (60,60):79,
    (80,60):107,(100,60):135,
}

def get_n_size_grid(bw_mhz, scs_khz):
    key = (bw_mhz, scs_khz)
    if key not in prb_table:
        raise ValueError(f"Нет данных TS 38.101-1 для BW={bw_mhz}МГц / SCS={scs_khz}кГц")
    return prb_table[key]

def get_tti_duration_ms(scs_khz):
    # TTI = 1 слот; длительность слота = 1мс / (SCS/15кГц)
    return 1.0 / (scs_khz / 15)

def get_numerology_mu(scs_khz):
    return int(math.log2(scs_khz / 15))

# Упрощённый TDD slot-формат (аналог DDDSU, типично для band n78/n41)
# D = только DL, U = только UL, S = Special slot (в этой модели считаем как DL с уменьшенным окном)
TDD_PATTERN_DEFAULT = "DDDSU"

def get_slot_type(tti_index, pattern=TDD_PATTERN_DEFAULT):
    return pattern[tti_index % len(pattern)]

## структуры данных определяющие сценарий
@dataclass
class AxiTransaction:
    # одна операция чтения/записи блока в Shared RAM через AXI
    stage: str         # имя этапа конвейера
    read_bytes: int    # чтение из RAM перед обработкой
    write_bytes: int   # запись результата обратно в RAM

@dataclass
class PhyProcessingSizes:
    # размеры блоков на каждом этапе обработки PHY Downlink
    tb_bits: int            # A transport block
    tb_crc_bits: int        # CRC attachment
    base_graph: str         # LDPC base graph selection
    cb_crc_bits: int        # Code block segmentation and CRC
    mapped_res: int         # Mapping to RBs (REs)
    axi_bus_bytes: int      # Нагрузка на шину AXI
    axi_transactions: List[AxiTransaction] = field(default_factory=list)

@dataclass
class UeAllocation:
    # описание ресурсов, выделенных для абонента (UE)
    ue_id: int
    snr: float                     # CSI-channel state information (SNR)
    mcs: int                       # выбранный MCS
    format_prb: str                # формат - nPRB x nOFDMsymb x nLayers
    phy_sizes: PhyProcessingSizes
    pdcch_axi_bytes: int = 0       # нагрузка control-channel (PDCCH) на AXI, только DL

@dataclass
class TtiSlot:
    # Один элемент массива TTI (subframe)
    tti_index: int
    users: List[UeAllocation] = field(default_factory=list)

@dataclass
class CellConfig:
    # конфигурация CELL базовой станции
    duplex: str = "FDD"
    bw_mhz: int = 100
    scs_khz: int = 30
    tdd_pattern: str = TDD_PATTERN_DEFAULT
    num_layers: int = 1   # 1 = SISO, 2 = 2x2 MIMO итд
    tti_duration_ms: float = field(init=False)
    n_size_grid: int = field(init=False)
    mu: int = field(init=False)

    def __post_init__(self):
        self.tti_duration_ms = get_tti_duration_ms(self.scs_khz)
        self.n_size_grid = get_n_size_grid(self.bw_mhz, self.scs_khz)
        self.mu = get_numerology_mu(self.scs_khz)
        if self.duplex not in ("FDD", "TDD"):
            raise ValueError("duplex должен быть 'FDD' или 'TDD'")

def calculate_resources_for_packet(packet_size_bytes, mcs_index, start_prb=0, scs_khz=30, n_size_grid=273, num_layers=1):
    # start_prb нужен для сдвига абонентов друг за другом в сетке

    modulation, R_target, Qm = mcs_table[mcs_index]
    R = R_target / 1024.0 # скорость кодирования

    ## настройка общей сетки базовой станции (carrier)
    carrier = py3gpp.nrCarrierConfig()
    carrier.NSizeGrid = n_size_grid
    carrier.SubcarrierSpacing = scs_khz
    
    # настраиваем выделение ресурсов под нашего абонента (PDSCH - канал данных)
    pdsch = py3gpp.nrPDSCHConfig()
    
    ## логика планировщика
    bits_per_re = Qm * R
    re_per_prb = 144 # 12 поднесущих * 12 символов 
    bits_per_prb = re_per_prb * bits_per_re
    
    allocated_prb = max(1, math.ceil((packet_size_bytes * 8 + 24) / bits_per_prb))
    
    # исправление бага библиотеки 
    pdsch.NStartBWP = 0                       
    pdsch.NSizeBWP = carrier.NSizeGrid
    
    # PRB выделяются не с нуля, а со смещением start_prb, чтобы предотвратить накладывание абонентов
    end_prb = min(start_prb + allocated_prb, carrier.NSizeGrid)
    pdsch.PRBSet = list(range(start_prb, end_prb))
    
    # выделяем символы в слоте
    pdsch.SymbolAllocation = [2, 12] 
    pdsch.Modulation = modulation
    
    ## вычисляем Resource Elements
    result_indices = py3gpp.nrPDSCHIndices(carrier, pdsch)
    
    if isinstance(result_indices, tuple):
        ind = result_indices[0] # берем первый элемент
    else:
        ind = result_indices    # если вернулся только один массив
        
    num_re = len(ind) # количество загруженных Resouce Elements
    num_re *= num_layers  # каждый доп слой MIMO это независимый поток RE (как N x SISO)
    
    ## определение кодека (LDPC BG1 BG2) (3GPP TS 38.212)
    A = packet_size_bytes * 8  # Считаем размер в битах

    if A <= 292:
        codec_type = "LDPC Base Graph 2"
    elif A <= 3824 and R <= 0.67:
        codec_type = "LDPC Base Graph 2"
    elif R <= 0.25:
        codec_type = "LDPC Base Graph 2"
    else:
        codec_type = "LDPC Base Graph 1"
    
    ## разбиваем нагрузку на AXI по стадиям конвейера (SoC: каждый блок обращается к Shared RAM)
    tb_bytes = math.ceil(A / 8)
    cb_crc_bytes = math.ceil((A + 48) / 8)   # A + CRC(24) + CB-CRC(24)
    grid_bytes = num_re * 4                   # 16бит I + 16бит Q на RE

    transactions = [
        AxiTransaction("Encoder LDPC",  read_bytes=tb_bytes,     write_bytes=cb_crc_bytes),
        AxiTransaction("QAM mapper",    read_bytes=cb_crc_bytes, write_bytes=grid_bytes),
        AxiTransaction("Gridmapper",    read_bytes=grid_bytes,   write_bytes=grid_bytes),
    ]
    axi_bus_bytes = sum(t.read_bytes + t.write_bytes for t in transactions)

    # заполняем структуру конвейера PHY
    phy_sizes = PhyProcessingSizes(
        tb_bits=A,
        tb_crc_bits=A + 24, # добавляем 24 бита CRC
        base_graph=codec_type,
        cb_crc_bits=A + 24 + 24, # 1 code block + crc
        mapped_res=num_re,
        axi_bus_bytes=axi_bus_bytes,
        axi_transactions=transactions
    )
    return {
        "Packet Size": f"{packet_size_bytes} байт",
        "Allocated PRBs": allocated_prb,
        "Resource Elements Used": num_re,
        "Codec": codec_type,
        "AXI Bus Load": f"{axi_bus_bytes} байт сырых I/Q данных",
        "Phy Sizes": phy_sizes # Возвращаем объект для генератора
    }

# Упрощённая модель PDCCH: aggregation level по SNR, Polar-кодер (TS 38.212 п. 7.3, TS 38.213 п. 10)
RE_PER_CCE = 72          # 1 CCE = 6 REG, 1 REG = 12 RE
PDCCH_DCI_BITS = 40      # типовой payload DCI format 1_0

def calculate_pdcch_resources(snr, dci_bits=PDCCH_DCI_BITS):
    # чем хуже SNR тем больше CCE нужно для надёжного декодирования
    if snr >= 20:   agg_level = 1
    elif snr >= 15: agg_level = 2
    elif snr >= 10: agg_level = 4
    elif snr >= 5:  agg_level = 8
    else:           agg_level = 16

    num_re = agg_level * RE_PER_CCE
    A = dci_bits + 24  # + CRC scrambled RNTI

    dci_bytes = math.ceil(dci_bits / 8)
    coded_bytes = math.ceil(A / 8)
    grid_bytes = num_re * 4

    transactions = [
        AxiTransaction("Encoder Polar",    read_bytes=dci_bytes,   write_bytes=coded_bytes),
        AxiTransaction("PDCCH Gridmapper", read_bytes=coded_bytes, write_bytes=grid_bytes),
    ]
    axi_bus_bytes = sum(t.read_bytes + t.write_bytes for t in transactions)

    return {
        "Aggregation Level": agg_level,
        "PDCCH REs": num_re,
        "AXI Bus Load": axi_bus_bytes,
        "Transactions": transactions
    }

def apply_power_control(snr, distance_penalty_db=0.0, max_mcs_ceiling=27):
    # абонент далеко -> имитируем через штраф SNR; ceiling ограничивает MCS сверху
    effective_snr = snr - distance_penalty_db
    mcs = snr_to_mcs(effective_snr)
    return min(mcs, max_mcs_ceiling)

## генератор сценариев с регулируемой динамикой
def generate_simulation_scenario(num_ttis, config=None):
    if config is None:
        config = CellConfig()
    array_of_ttis_dl = []
    array_of_ttis_ul = []
    
    # сценарий
    start_users_count = 6      # кол-во пользователей в начале
    change_probability = 0.01  # шанс изменения (0.01 = 1%)
    max_cell_users = 15        # максимум пользователей в соте
    min_cell_users = 1         # минимум пользователей
    max_prb_per_ue = 40        # лимит ресурсов на одного (чтобы всем хватило места) так как 273 ресурсных блока


    # стартовый пул пользователей
    active_users = {i: random.uniform(10.0, 25.0) for i in range(1, start_users_count + 1)}

    for tti in range(num_ttis):
        # слоты для downlink и uplink
        current_tti_dl = TtiSlot(tti_index=tti)
        current_tti_ul = TtiSlot(tti_index=tti)

        # В TDD на каждый TTI активно только одно направление (пример: DDDSU)
        # В FDD DL и UL всегда одновременно активны (разные частоты)
        if config.duplex == "TDD":
            slot_type = get_slot_type(tti, config.tdd_pattern)
            dl_active = slot_type in ("D", "S")  # S считаем DL-ориентированным
            ul_active = slot_type == "U"
        else:
            dl_active = True
            ul_active = True
        
        # В режиме FDD (разные частоты) у DL и UL свои независимые 273 PRB
        current_prb_dl = 0 
        current_prb_ul = 0 
        
        # Групповое изменение состава пользователей
        if random.random() < change_probability:
            change_type = random.choice(['add', 'remove'])
            # выбираем сколько человек придет или уйдет (от 1 до 3 не уверен, каждый TTI = 0.5 мс, по этому шанс поставил маленький)
            num_to_change = random.randint(1, 3) 
            
            if change_type == 'add':
                for _ in range(num_to_change):
                    if len(active_users) < max_cell_users:
                        new_id = max(active_users.keys()) + 1 if active_users else 1
                        active_users[new_id] = random.uniform(5.0, 25.0)
            
            elif change_type == 'remove':
                for _ in range(num_to_change):
                    if len(active_users) > min_cell_users:
                        target_id = random.choice(list(active_users.keys()))
                        active_users.pop(target_id)

        # физика активных пользователей
        for ue_id in list(active_users.keys()):
            # плавный дрейф сигнала
            active_users[ue_id] += random.uniform(-0.5, 0.5) 
            snr = max(0.0, min(30.0, active_users[ue_id]))
            
            # маппинг SNR -> MCS
            # штраф по мощности зависит от псевдо-удалённости (детерминирован от ue_id)
            distance_penalty = (ue_id % 5) * 1.5   # 0..6 дБ
            mcs = apply_power_control(snr, distance_penalty_db=distance_penalty)
            
            # downlink
            if dl_active:
                payload_dl = random.choice([50, 100, 300, 600, 1000])
                res_dl = calculate_resources_for_packet(payload_dl, mcs, start_prb=current_prb_dl, scs_khz=config.scs_khz, n_size_grid=config.n_size_grid, num_layers=config.num_layers)
                alloc_prbs_dl = min(res_dl["Allocated PRBs"], max_prb_per_ue)
                # PDCCH считается отдельно от PDSCH — свой Polar-кодер, не занимает PRB данных
                res_pdcch = calculate_pdcch_resources(snr)

                if current_prb_dl + alloc_prbs_dl <= config.n_size_grid:
                    current_prb_dl += alloc_prbs_dl
                    current_tti_dl.users.append(
                        UeAllocation(ue_id, round(snr, 1), mcs, f"{alloc_prbs_dl}x12x1",
                                    res_dl["Phy Sizes"], pdcch_axi_bytes=res_pdcch["AXI Bus Load"])
                    )

            # uplink
            # Трафик UL обычно меньше 
            if ul_active:
                payload_ul = random.choice([10, 20, 50, 100, 300])
                res_ul = calculate_resources_for_packet(payload_ul, mcs, start_prb=current_prb_ul, scs_khz=config.scs_khz, n_size_grid=config.n_size_grid, num_layers=config.num_layers)
                alloc_prbs_ul = min(res_ul["Allocated PRBs"], max_prb_per_ue)

                if current_prb_ul + alloc_prbs_ul <= config.n_size_grid:
                    current_prb_ul += alloc_prbs_ul
                    current_tti_ul.users.append(
                        UeAllocation(ue_id, round(snr, 1), mcs, f"{alloc_prbs_ul}x12x1", res_ul["Phy Sizes"])
                    )
            
        # сохраняем заполненные TTI в соответствующие массивы
        array_of_ttis_dl.append(current_tti_dl)
        array_of_ttis_ul.append(current_tti_ul)
        
    return config, array_of_ttis_dl, array_of_ttis_ul

def recommend_buffer_architecture(df, axi_bus_capacity_gbps=7.5):
    # анализ пиковой нагрузки по стадиям конвейера -> рекомендация архитектуры буфера
    stage_totals = {}
    for _, row in df.iterrows():
        for t in row.get('_transactions', []) or []:
            stage_totals.setdefault(t.stage, 0)
            stage_totals[t.stage] += t.read_bytes + t.write_bytes

    if not stage_totals:
        return "Недостаточно данных по транзакциям для рекомендации"

    bottleneck_stage = max(stage_totals, key=stage_totals.get)
    bottleneck_bytes = stage_totals[bottleneck_stage]
    share = bottleneck_bytes / sum(stage_totals.values())

    if share > 0.5:
        return (f"Bottleneck: '{bottleneck_stage}' ({share:.0%} трафика AXI). "
                f"Рекомендуется локальный SRAM-буфер именно для этого блока, "
                f"остальные - через общий Shared RAM.")
    return "Нагрузка распределена равномерно — Shared RAM через единую AXI-шину достаточна."

# === ЗАПУСК СИМУЛЯЦИИ ===
# FDD пример: config = CellConfig(bw_mhz=100, scs_khz=30, duplex="FDD")
# TDD пример (типично для band n78, полоса меньше чем при FDD):
config = CellConfig(bw_mhz=100, scs_khz=30, duplex="TDD", tdd_pattern="DDDSU")  # можно менять scs 15/30/60 кГц, BW из prb_table
num_sim_ttis = 6000
_, dl_scenario, ul_scenario = generate_simulation_scenario(num_ttis=num_sim_ttis, config=config)

# === СБОРКА ДАННЫХ В PANDAS DATAFRAME ===
records = []

for tti_dl, tti_ul in zip(dl_scenario, ul_scenario):
    # Собираем данные Downlink
    for ue in tti_dl.users:
        records.append({
            "TTI": tti_dl.tti_index,
            "Direction": "DL",
            "UE_ID": ue.ue_id,
            "SNR_dB": ue.snr,
            "MCS": ue.mcs,
            "Resources": ue.format_prb,                 
            "PRBs": int(ue.format_prb.split('x')[0]),   # Оставил цифру для удобства аналитики
            "Payload_Bits": ue.phy_sizes.tb_bits,
            "Codec": ue.phy_sizes.base_graph,
            "AXI_Load_Bytes": ue.phy_sizes.axi_bus_bytes,
            "PDCCH_AXI_Bytes": ue.pdcch_axi_bytes,
        })
        
    # Собираем данные Uplink
    for ue in tti_ul.users:
        records.append({
            "TTI": tti_ul.tti_index,
            "Direction": "UL",
            "UE_ID": ue.ue_id,
            "SNR_dB": ue.snr,
            "MCS": ue.mcs,
            "Resources": ue.format_prb,               
            "PRBs": int(ue.format_prb.split('x')[0]),
            "Payload_Bits": ue.phy_sizes.tb_bits,
            "Codec": ue.phy_sizes.base_graph,
            "AXI_Load_Bytes": ue.phy_sizes.axi_bus_bytes,
            "PDCCH_AXI_Bytes": 0,
        })

df = pd.DataFrame(records)

# === ВЫВОД И АНАЛИТИКА ===
duplex_info = f"TDD [{config.tdd_pattern}]" if config.duplex == "TDD" else "FDD"
print(f"=== Конфигурация: {duplex_info} | BW: {config.bw_mhz}MHz | SCS: {config.scs_khz}kHz "
      f"| μ={config.mu} | NSizeGrid: {config.n_size_grid} PRB | TTI: {config.tti_duration_ms}ms ===")
print(f"Всего записей в симуляции: {len(df)}")

# 1. кусочек самой таблицы (первые 10 строк)
print("\n--- Фрагмент базы данных (первые 10 операций) ---")
print(df.head(10).to_string(index=False))

# Время всей симуляции в секундах (нужно для расчета скоростей)
total_time_sec = num_sim_ttis * config.tti_duration_ms / 1000.0

# 2. МЕТРИКА (По чекбоксу 1): "За один TTI сколько всего данных суммарно от всех челов мы передали"
print("\n--- Пропускная способность соты (Cell Capacity) ---")
# Группируем DL трафик по TTI и считаем сумму бит
dl_payload_per_tti = df[df['Direction'] == 'DL'].groupby('TTI')['Payload_Bits'].sum()

# Переводим в Килобайты (чтобы было как просил препод)
avg_kb_per_tti = (dl_payload_per_tti.mean() / 8) / 1024
max_kb_per_tti = (dl_payload_per_tti.max() / 8) / 1024
# Считаем реальную скорость соты в Мбит/с
cell_throughput_mbps = (dl_payload_per_tti.sum() / 1e6) / total_time_sec

print(f"Средний объем переданных данных (DL) за один TTI: {round(avg_kb_per_tti, 2)} КБ")
print(f"Пиковый объем переданных данных (DL) за один TTI:  {round(max_kb_per_tti, 2)} КБ")
print(f"Реальная средняя скорость соты (DL Cell Throughput): {round(cell_throughput_mbps, 2)} Мбит/с")

# 3. МЕТРИКИ (По чекбоксам 2 и 3): "Накопленные данные и средняя скорость пользователя"
print(f"\n--- Статистика по Абонентам за время симуляции ({total_time_sec} сек) ---")

# Считаем суммарный трафик в Битах
user_traffic_bits = df.groupby(['UE_ID', 'Direction'])['Payload_Bits'].sum().unstack().fillna(0)

# Переводим в Килобайты (Накопленные данные)
user_traffic_kb = user_traffic_bits / 8 / 1024
# Переводим в Скорость Мбит/с
user_throughput_mbps = (user_traffic_bits / 1e6) / total_time_sec

# Собираем красивую итоговую таблицу
user_stats = pd.DataFrame({
    'Скачал всего (DL, КБ)': user_traffic_kb['DL'].round(2),
    'Отправил всего (UL, КБ)': user_traffic_kb['UL'].round(2),
    'Скорость DL (Мбит/с)': user_throughput_mbps['DL'].round(2),
    'Скорость UL (Мбит/с)': user_throughput_mbps['UL'].round(2)
})
print(user_stats.to_string())

# 4. МЕТРИКА: "Какая максимальная нагрузка на кодеки и средняя"
print("\n--- Нагрузка на шину AXI (Hardware Performance) ---")
df['Total_AXI_Bytes'] = df['AXI_Load_Bytes'] + df['PDCCH_AXI_Bytes']
axi_per_tti = df.groupby('TTI')['Total_AXI_Bytes'].sum()
print(f"\n--- Разбивка PDSCH vs PDCCH ---")
print(f"PDSCH AXI (сумма): {df['AXI_Load_Bytes'].sum()} байт")
print(f"PDCCH AXI (сумма): {df['PDCCH_AXI_Bytes'].sum()} байт")
print(f"Средняя нагрузка: {round(axi_per_tti.mean(), 2)} байт / TTI")
print(f"Пиковая (Максимальная) нагрузка: {axi_per_tti.max()} байт / TTI")

print("\n--- Использование кодеков LDPC ---")
print(df['Codec'].value_counts().to_string())

# 4. Сохраняем в файл
df.to_csv('simulation_results.csv', index=False)
print("\nДанные сохранены в файл simulation_results.csv")


# === ПОСТРОЕНИЕ ГРАФИКА ДИНАМИКИ ПОЛЬЗОВАТЕЛЕЙ ===
import matplotlib.pyplot as plt

def plot_resource_grid_heatmap(tti_slot, n_size_grid=273):
    grid = np.full(n_size_grid, np.nan)  # nan вместо 0 = свободно
    cursor = 0
    for ue in tti_slot.users:
        prbs = int(ue.format_prb.split('x')[0])
        end = min(cursor + prbs, n_size_grid)
        grid[cursor:end] = ue.ue_id
        cursor = end
        if cursor >= n_size_grid:
            break

    cmap = plt.cm.tab20
    cmap.set_bad(color='lightgrey')  # nan = серый = свободно

    plt.figure(figsize=(14, 2))
    plt.imshow(grid.reshape(1, -1), aspect='auto', cmap=cmap)
    plt.title(f'Resource Grid DL: распределение PRB между UE (TTI={tti_slot.tti_index})')
    plt.xlabel('PRB index')
    plt.yticks([])
    plt.colorbar(label='UE_ID')
    plt.tight_layout()
    plt.show()

plot_resource_grid_heatmap(dl_scenario[0], n_size_grid=config.n_size_grid)

# Группируем данные: считаем количество уникальных UE_ID для каждого TTI
# (Берем только DL, так как количество абонентов в DL и UL у нас одинаковое)
users_per_tti = df[df['Direction'] == 'DL'].groupby('TTI')['UE_ID'].nunique()

# Настройка внешнего вида графика
plt.figure(figsize=(12, 6))
# drawstyle='steps-post' делает график "ступеньками", что отражает дискретную природу подключений
plt.plot(users_per_tti.index, users_per_tti.values, drawstyle='steps-post', color='#1f77b4', linewidth=2.5)

# Оформление
plt.title(f'Динамика активных абонентов в соте (Симуляция {num_sim_ttis * config.tti_duration_ms / 1000} сек)', fontsize=14, pad=15)
plt.xlabel('Время (TTI, 2000 TTI = 1 с)', fontsize=12)
plt.ylabel('Количество активных пользователей', fontsize=12)

# Сетки и лимиты
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 16) # От 0 до max_cell_users + 1
plt.xlim(0, num_sim_ttis)

# Показывать график на экране
plt.tight_layout()
plt.show()