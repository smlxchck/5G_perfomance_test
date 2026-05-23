import py3gpp
import numpy as np
import math
import random
import pandas as pd
from dataclasses import dataclass, field
from typing import List

## структуры данных определяющие сценарий
@dataclass
class PhyProcessingSizes:
    # размеры блоков на каждом этапе обработки PHY Downlink
    tb_bits: int            # A transport block
    tb_crc_bits: int        # CRC attachment
    base_graph: str         # LDPC base graph selection
    cb_crc_bits: int        # Code block segmentation and CRC
    mapped_res: int         # Mapping to RBs (REs)
    axi_bus_bytes: int      # Нагрузка на шину AXI

@dataclass
class UeAllocation:
    # описание ресурсов, выделенных для абонента (UE)
    ue_id: int
    snr: float              # CSI-channel state information (SNR)
    mcs: int                # выбранный MCS
    format_prb: str         # формат - nPRB x nOFDMsymb x nLayers
    phy_sizes: PhyProcessingSizes

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
    tti_duration_ms: float = 0.5  # Для 30kHz TTI = 0.5ms

def calculate_resources_for_packet(packet_size_bytes, mcs_index, start_prb=0):
    # start_prb нужен для сдвига абонентов друг за другом в сетке

    # Таблица MCS 256QAM (Модуляция, R x 1024, Qm)
    mcs_table = {
        0:  ('qpsk',   120, 2), 
        8:  ('qam16',  553, 4), 
        16: ('qam64',  719, 6), 
        27: ('qam256', 948, 8)  
    }
    modulation, R_target, Qm = mcs_table[mcs_index]
    R = R_target / 1024.0 # скорость кодирования

    ## настройка общей сетки базовой станции (carrier)
    carrier = py3gpp.nrCarrierConfig()
    carrier.NSizeGrid = 273        # 273 ресурсных блока (для полосы 100 МГц)
    carrier.SubcarrierSpacing = 30 # нумерология 30 кГц
    
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
        
    ## cчитаем трафик для шины AXI
    axi_bus_bytes = num_re * 4
    
    # заполняем структуру конвейера PHY
    phy_sizes = PhyProcessingSizes(
        tb_bits=A,
        tb_crc_bits=A + 24, # добавляем 24 бита CRC
        base_graph=codec_type,
        cb_crc_bits=A + 24 + 24, # 1 code block + crc
        mapped_res=num_re,
        axi_bus_bytes=axi_bus_bytes
    )

    return {
        "Packet Size": f"{packet_size_bytes} байт",
        "Allocated PRBs": allocated_prb,
        "Resource Elements Used": num_re,
        "Codec": codec_type,
        "AXI Bus Load": f"{axi_bus_bytes} байт сырых I/Q данных",
        "Phy Sizes": phy_sizes # Возвращаем объект для генератора
    }

## генератор сценариев с регулируемой динамикой
def generate_simulation_scenario(num_ttis):
    config = CellConfig()
    array_of_ttis_dl = []
    array_of_ttis_ul = []
    
    # сценарий
    start_users_count = 6      # кол-во пользователей в начале
    change_probability = 0.0001  # шанс изменения (0.01 = 1%)
    max_cell_users = 15        # максимум пользователей в соте
    min_cell_users = 1         # минимум пользователей
    max_prb_per_ue = 40        # лимит ресурсов на одного (чтобы всем хватило места) так как 273 ресурсных блока


    # стартовый пул пользователей
    active_users = {i: random.uniform(10.0, 25.0) for i in range(1, start_users_count + 1)}
    
    for tti in range(num_ttis):
        # слоты для downlink и uplink
        current_tti_dl = TtiSlot(tti_index=tti)
        current_tti_ul = TtiSlot(tti_index=tti) 
        
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
            mcs = 0 if snr < 10 else 8 if snr < 15 else 16 if snr < 20 else 27
            
            # downlink
            payload_dl = random.choice([50, 100, 300, 600, 1000])
            res_dl = calculate_resources_for_packet(payload_dl, mcs, start_prb=current_prb_dl)
            alloc_prbs_dl = min(res_dl["Allocated PRBs"], max_prb_per_ue)
            
            if current_prb_dl + alloc_prbs_dl <= 273: 
                current_prb_dl += alloc_prbs_dl
                current_tti_dl.users.append(
                    UeAllocation(ue_id, round(snr, 1), mcs, f"{alloc_prbs_dl}x12x1", res_dl["Phy Sizes"])
                )

            # uplink
            # Трафик UL обычно меньше 
            payload_ul = random.choice([10, 20, 50, 100, 300])
            res_ul = calculate_resources_for_packet(payload_ul, mcs, start_prb=current_prb_ul)
            alloc_prbs_ul = min(res_ul["Allocated PRBs"], max_prb_per_ue)
            
            if current_prb_ul + alloc_prbs_ul <= 273: 
                current_prb_ul += alloc_prbs_ul
                current_tti_ul.users.append(
                    UeAllocation(ue_id, round(snr, 1), mcs, f"{alloc_prbs_ul}x12x1", res_ul["Phy Sizes"])
                )
            
        # сохраняем заполненные TTI в соответствующие массивы
        array_of_ttis_dl.append(current_tti_dl)
        array_of_ttis_ul.append(current_tti_ul)
        
    return config, array_of_ttis_dl, array_of_ttis_ul

# === ЗАПУСК СИМУЛЯЦИИ ===
num_sim_ttis = 6000
config, dl_scenario, ul_scenario = generate_simulation_scenario(num_ttis=num_sim_ttis) 

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
            "AXI_Load_Bytes": ue.phy_sizes.axi_bus_bytes
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
            "AXI_Load_Bytes": ue.phy_sizes.axi_bus_bytes
        })

df = pd.DataFrame(records)

# === ВЫВОД И АНАЛИТИКА ===
print(f"=== Конфигурация: {config.duplex} | BW: {config.bw_mhz}MHz | SCS: {config.scs_khz}kHz | TTI: {config.tti_duration_ms}ms ===")
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
axi_per_tti = df.groupby('TTI')['AXI_Load_Bytes'].sum()
print(f"Средняя нагрузка: {round(axi_per_tti.mean(), 2)} байт / TTI")
print(f"Пиковая (Максимальная) нагрузка: {axi_per_tti.max()} байт / TTI")

print("\n--- Использование кодеков LDPC ---")
print(df['Codec'].value_counts().to_string())

# 4. Сохраняем в файл
df.to_csv('simulation_results.csv', index=False)
print("\nДанные сохранены в файл simulation_results.csv")


# === ПОСТРОЕНИЕ ГРАФИКА ДИНАМИКИ ПОЛЬЗОВАТЕЛЕЙ ===
import matplotlib.pyplot as plt

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