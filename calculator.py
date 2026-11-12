import py3gpp
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
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
    rv: int = 0                   # redundancy version (HARQ): 0=новый TB, 1-3=ретрансмиссии
    harq_soft_bytes: int = 0      # мягкие биты для комбинирования (реальная нагрузка на SRAM)

@dataclass
class UeAllocation:
    # описание ресурсов, выделенных для абонента (UE)
    ue_id: int
    snr: float                     # CSI-channel state information (SNR)
    mcs: int                       # выбранный MCS
    format_prb: str                # формат - nPRB x nOFDMsymb x nLayers
    phy_sizes: PhyProcessingSizes
    pdcch_axi_bytes: int = 0       # нагрузка control-channel (PDCCH) на AXI, только DL
    pdcch_transactions: list = field(default_factory=list)  # транзакции PDCCH
    tb_id: int = -1                # id транспортного блока
    harq_pid: int = -1             # номер HARQ-процесса
    harq_rv: int = 0               # 0 = новая передача, 1-3 = ретрансмиссии HARQ
    is_retransmission: bool = False  # True = ретрансмиссия, False = первая передача TB
    harq_soft_bytes: int = 0       # мягкие биты в soft buffer (только UL - приёмник gNB)

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

def calculate_resources_for_packet(packet_size_bytes, mcs_index, start_prb=0, scs_khz=30, n_size_grid=273, num_layers=1, rv=0):
    # start_prb нужен для сдвига абонентов друг за другом в сетке. На расположение RE rv не влияет (только на выбор кодированных бит), поэтому сохраняем его как метаданные в PhyProcessingSizes.

    modulation, R_target, Qm = mcs_table[mcs_index]
    R = R_target / 1024.0 # скорость кодирования
    A = packet_size_bytes * 8  # размер транспортного блока в битах

    ## определение кодека (LDPC BG1 BG2) (3GPP TS 38.212)
    if A <= 292:
        codec_type = "LDPC Base Graph 2"
    elif A <= 3824 and R <= 0.67:
        codec_type = "LDPC Base Graph 2"
    elif R <= 0.25:
        codec_type = "LDPC Base Graph 2"
    else:
        codec_type = "LDPC Base Graph 1"

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
    if start_prb >= end_prb:
        # свободных PRB нет - курсор планировщика дошёл до края сетки (start_prb == NSizeGrid).
        # py3gpp не принимает пустой PRBSet - его сеттер делает min(prbset) и падает на пустом списке ошибка - (ValueError: min() iterable argument is empty).
        # Поэтому возвращаем нулевой результат - планировщик такой пакет пропустит, а прямой вызов функции не обрушит расчёт.
        phy_sizes = PhyProcessingSizes(
            tb_bits=A, tb_crc_bits=A + 24, base_graph=codec_type,
            cb_crc_bits=A + 24 + 24, mapped_res=0,
            axi_bus_bytes=0, axi_transactions=[], rv=rv, harq_soft_bytes=0)
        return {
            "Packet Size": f"{packet_size_bytes} байт",
            "Allocated PRBs": 0,
            "Resource Elements Used": 0,
            "Codec": codec_type,
            "AXI Bus Load": "0 байт (нет свободных PRB)",
            "HARQ Soft Bytes": 0,
            "RV": rv,
            "Phy Sizes": phy_sizes
        }
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
    
    ## разбиваем нагрузку на AXI по стадиям конвейера
    # Полная цепочка по схеме SoC: Eth DU RX (binary source) кладёт TB в память -> Encoder LDPC -> QAM mapper -> Gridmapper -> Eth to RRU забирает грид.
    # Каждый прямоугольник = чтение из Shared RAM + запись обратно через AXI4-шину.
    tb_bytes = math.ceil(A / 8)
    cb_crc_bytes = math.ceil((A + 48) / 8)   # A + CRC(24) + CB-CRC(24)
    grid_bytes = num_re * 4                   # 16бит I + 16бит Q на RE

    transactions = [
        AxiTransaction("Eth DU RX",     read_bytes=0,            write_bytes=tb_bytes),
        AxiTransaction("Encoder LDPC",  read_bytes=tb_bytes,     write_bytes=cb_crc_bytes),
        AxiTransaction("QAM mapper",    read_bytes=cb_crc_bytes, write_bytes=grid_bytes),
        AxiTransaction("Gridmapper",    read_bytes=grid_bytes,   write_bytes=grid_bytes),
        AxiTransaction("Eth to RRU",    read_bytes=grid_bytes,   write_bytes=0),
    ]
    axi_bus_bytes = sum(t.read_bytes + t.write_bytes for t in transactions)

    # мягкие биты HARQ - каждый RE несёт Qm кодированных бит, в soft buffer приёмник хранит мягкие решения по SOFT_BITS_PER_SAMPLE бит на бит кода.
    harq_soft_bytes = int(math.ceil(num_re * Qm * SOFT_BITS_PER_SAMPLE / 8))

    # заполняем структуру конвейера PHY
    phy_sizes = PhyProcessingSizes(
        tb_bits=A,
        tb_crc_bits=A + 24, # добавляем 24 бита CRC
        base_graph=codec_type,
        cb_crc_bits=A + 24 + 24, # 1 code block + crc
        mapped_res=num_re,
        axi_bus_bytes=axi_bus_bytes,
        axi_transactions=transactions,
        rv=rv,
        harq_soft_bytes=harq_soft_bytes
    )
    return {
        "Packet Size": f"{packet_size_bytes} байт",
        "Allocated PRBs": allocated_prb,
        "Resource Elements Used": num_re,
        "Codec": codec_type,
        "AXI Bus Load": f"{axi_bus_bytes} байт сырых I/Q данных",
        "HARQ Soft Bytes": harq_soft_bytes,
        "RV": rv,
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

## Быстродействие блоков SoC в ГБ/с
# Скорость обработки каждого блока конвейера. По постановке у QAM mapper и Gridmapper быстродействие ниже, чем у шины, а у Shared RAM/шины - высокое, поэтому узкое место ищем в блоках, а не в шине. Значения настраиваются в UI (сайдбар слева в менюшке).
DEFAULT_BLOCK_SPEEDS_GBPS = {
    "Eth DU RX":        8.0,   # запись TB от DU в Shared RAM
    "Encoder LDPC":     3.0,
    "Encoder Polar":    4.0,
    "QAM mapper":       1.5,   # медленный блок
    "Gridmapper":       1.0,   # самый медленный блок конвейера
    "PDCCH Gridmapper": 2.0,
    "Eth to RRU":       8.0,   # чтение грида и отдача в RRU
}

# Порядок блоков в конвейере (для схемы SoC и графика накопления нагрузки)
AXI_PDSCH_ORDER = ["Eth DU RX", "Encoder LDPC", "QAM mapper", "Gridmapper", "Eth to RRU"]
AXI_PDCCH_ORDER = ["Encoder Polar", "PDCCH Gridmapper"]

## Параметры HARQ
# В 5G DL до 16 параллельных HARQ-процессов на UE 8 - типовая рабочая величина.
HARQ_PROCESS_COUNT = 8
# время ACK/NACK через сколько TTI после передачи приходит обратная связь.
# В FDD ~8 слотов (k1 по TS 38.213) - берём фикс. знач.
HARQ_RTT_TTIS = 8
# Максимум ретрансмиссий RV0 (первая) + RV1..RV3. После неудачи на RV3 TB отбрасывается.
HARQ_MAX_RETX = 3
# Бит на мягкий символ в soft buffer (типично 4-8 бит; влияет на размер SRAM чипа).
SOFT_BITS_PER_SAMPLE = 6

def bler_from_snr(effective_snr, mcs_index):
    #Оценка BLER по отставанию SNR от порога выбранного MCS. Проектная точка таблицы MCS (TS 38.214)
    
    snr_req = MCS_SNR_THRESHOLDS[0][0]
    for req, idx in MCS_SNR_THRESHOLDS:
        if idx == mcs_index:
            snr_req = req
            break
    delta = effective_snr - snr_req
    if delta >= 0:
        bler = 0.1 * math.exp(-delta / 2.0)
    else:
        bler = 1.0 - 0.9 * math.exp(delta / 2.0)
    return max(0.001, min(0.95, bler))

def apply_power_control(snr, distance_penalty_db=0.0, max_mcs_ceiling=27):
    # абонент далеко -> имитируем через штраф SNR; ceiling ограничивает MCS сверху
    effective_snr = snr - distance_penalty_db
    mcs = snr_to_mcs(effective_snr)
    return min(mcs, max_mcs_ceiling)

## генератор сценариев с регулируемой динамикой (+ HARQ)
def generate_simulation_scenario(num_ttis, config=None):
    if config is None:
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

    # состояние HARQ: у каждого абонента свой пул процессов
    # busy_until[pid] = TTI, в который придёт ACK/NACK
    # pending_dl / pending_ul - очередь ретрансмиссий (TB не декодировался с первой попытки)
    # soft_ul[pid] - мягкие биты в буфере HARQ. Только UL приёмник gNB хранит мягкие решения до успешного декодирования 
    harq_state = {}
    tb_counter = 0
    harq_acc = {
        'tb_total': 0,      # уникальных транспортных блоков
        'retx_total': 0,    # ретрансмиссий
        'dropped': 0,       # TB, не декодировавшиеся после HARQ_MAX_RETX попыток (или при уходе UE)
        'initial_tx': 0,    # первых передач
        'initial_nack': 0,  # NACK на первой передаче (т.е. фактический BLER)
        'buffer_sum': 0, 'buffer_count': 0, 'buffer_max': 0,
    }

    def _new_harq_state():
        return {
            'busy_until': {p: -1 for p in range(HARQ_PROCESS_COUNT)},
            'outcome': {},      # pid -> 'ack' | 'nack' | 'drop'
            'pending_dl': [],   # [{'tb_id','payload','mcs','rv','pid','next_tti'}]
            'pending_ul': [],
            'soft_ul': {},      # pid -> soft bytes
        }

    def _free_processes(st, tti):
        # RTT истёк - процесс освобождается. Soft buffer чистим только если TB дошёл ('ack') или отброшен ('drop'); при 'nack' буфер остаётся для комбинирования со следующей попыткой.
        for p, until in list(st['busy_until'].items()):
            if until != -1 and until <= tti:
                st['busy_until'][p] = -1
                out = st['outcome'].pop(p, None)
                if out in ('ack', 'drop'):
                    st['soft_ul'].pop(p, None)

    def _free_pid(st):
        # отдаём свободный процесс, НЕ зарезервированный под ждущую ретрансмиссию
        reserved = {e['pid'] for e in st['pending_dl']} | {e['pid'] for e in st['pending_ul']}
        for p, until in st['busy_until'].items():
            if until == -1 and p not in reserved:
                return p
        return -1

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
            # выбираем сколько человек придет или уйдет (от 1 до 3; каждый TTI = 0.5 мс, поэтому шанс маленький)
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
                        # ушедший абонент забирает недодекодированные TB с собой
                        st_gone = harq_state.pop(target_id, None)
                        if st_gone:
                            harq_acc['dropped'] += len(st_gone['pending_dl']) + len(st_gone['pending_ul'])

        # физика активных пользователей
        for ue_id in list(active_users.keys()):
            # плавный дрейф сигнала
            active_users[ue_id] += random.uniform(-0.5, 0.5)
            snr = max(0.0, min(30.0, active_users[ue_id]))

            # маппинг SNR -> MCS; effective_snr нужен и для модели BLER
            # штраф по мощности зависит от псевдо-удалённости (детерминирован от ue_id)
            distance_penalty = (ue_id % 5) * 1.5   # 0..6 дБ
            effective_snr = snr - distance_penalty
            mcs = apply_power_control(snr, distance_penalty_db=distance_penalty)

            st = harq_state.get(ue_id)
            if st is None:
                st = harq_state[ue_id] = _new_harq_state()

            # освобождаем HARQ-процессы, у которых истёк RTT
            _free_processes(st, tti)

            ## downlink
            if dl_active:
                dl_tx_done = False

                # 1) ретрансмиссии - приоритет над новыми пакетами
                for entry in list(st['pending_dl']):
                    if dl_tx_done:
                        break
                    if entry['next_tti'] > tti:
                        continue
                    if current_prb_dl >= config.n_size_grid:
                        break   # сетка заполнена предыдущими UE - ретрансмиссии не влезут в этот TTI
                    res = calculate_resources_for_packet(
                        entry['payload'], entry['mcs'], start_prb=current_prb_dl,
                        scs_khz=config.scs_khz, n_size_grid=config.n_size_grid,
                        num_layers=config.num_layers, rv=entry['rv'])
                    alloc = min(res["Allocated PRBs"], max_prb_per_ue)
                    if current_prb_dl + alloc > config.n_size_grid:
                        continue   # не влезло - попробуем в следующем TTI
                    current_prb_dl += alloc
                    dl_tx_done = True
                    harq_acc['retx_total'] += 1
                    # PDCCH нужен и для ретрансмиссии (DCI с флагом NDI=0)
                    res_pdcch = calculate_pdcch_resources(snr)
                    current_tti_dl.users.append(
                        UeAllocation(ue_id, round(snr, 1), entry['mcs'], f"{alloc}x12x1",
                                     res["Phy Sizes"], pdcch_axi_bytes=res_pdcch["AXI Bus Load"],
                                     pdcch_transactions=res_pdcch["Transactions"],
                                     tb_id=entry['tb_id'], harq_pid=entry['pid'],
                                     harq_rv=entry['rv'], is_retransmission=True))
                    # исход попытки: BLER по текущему effective_snr - комбинирование
                    # с прошлыми попытками неявно учтено тем, что SNR мог вырасти
                    st['busy_until'][entry['pid']] = tti + HARQ_RTT_TTIS
                    if random.random() < bler_from_snr(effective_snr, entry['mcs']):
                        if entry['rv'] < HARQ_MAX_RETX:
                            entry['rv'] += 1
                            entry['next_tti'] = tti + HARQ_RTT_TTIS
                            st['outcome'][entry['pid']] = 'nack'
                        else:
                            harq_acc['dropped'] += 1
                            st['outcome'][entry['pid']] = 'drop'
                    else:
                        st['outcome'][entry['pid']] = 'ack'
                    st['pending_dl'].remove(entry)

                # 2) новый пакет - только если есть свободный HARQ-процесс
                if not dl_tx_done:
                    pid = _free_pid(st)
                    # место в сетке проверяем ДО расчёта: если курсор дошёл до края (n_size_grid), свободных PRB нет - новый пакет в этом TTI не запланировать
                    if pid != -1 and current_prb_dl < config.n_size_grid:
                        payload_dl = random.choice([50, 100, 300, 600, 1000])
                        res_dl = calculate_resources_for_packet(payload_dl, mcs, start_prb=current_prb_dl, scs_khz=config.scs_khz, n_size_grid=config.n_size_grid, num_layers=config.num_layers)
                        alloc_prbs_dl = min(res_dl["Allocated PRBs"], max_prb_per_ue)
                        # PDCCH считается отдельно от PDSCH - свой Polar-кодер, не занимает PRB данных
                        res_pdcch = calculate_pdcch_resources(snr)

                        if current_prb_dl + alloc_prbs_dl <= config.n_size_grid:
                            current_prb_dl += alloc_prbs_dl
                            tb_counter += 1
                            harq_acc['tb_total'] += 1
                            harq_acc['initial_tx'] += 1
                            current_tti_dl.users.append(
                                UeAllocation(ue_id, round(snr, 1), mcs, f"{alloc_prbs_dl}x12x1",
                                             res_dl["Phy Sizes"], pdcch_axi_bytes=res_pdcch["AXI Bus Load"],
                                             pdcch_transactions=res_pdcch["Transactions"],
                                             tb_id=tb_counter, harq_pid=pid, harq_rv=0))
                            st['busy_until'][pid] = tti + HARQ_RTT_TTIS
                            if random.random() < bler_from_snr(effective_snr, mcs):
                                harq_acc['initial_nack'] += 1
                                st['pending_dl'].append(
                                    {'tb_id': tb_counter, 'payload': payload_dl, 'mcs': mcs,
                                     'rv': 1, 'pid': pid, 'next_tti': tti + HARQ_RTT_TTIS})
                                st['outcome'][pid] = 'nack'
                            else:
                                st['outcome'][pid] = 'ack'

            ## uplink
            # Трафик UL обычно меньше
            if ul_active:
                ul_tx_done = False

                # 1) ретрансмиссии
                for entry in list(st['pending_ul']):
                    if ul_tx_done:
                        break
                    if entry['next_tti'] > tti:
                        continue
                    if current_prb_ul >= config.n_size_grid:
                        break   # сетка заполнена предыдущими UE - ретрансмиссии не влезут в этот TTI
                    res = calculate_resources_for_packet(
                        entry['payload'], entry['mcs'], start_prb=current_prb_ul,
                        scs_khz=config.scs_khz, n_size_grid=config.n_size_grid,
                        num_layers=config.num_layers, rv=entry['rv'])
                    alloc = min(res["Allocated PRBs"], max_prb_per_ue)
                    if current_prb_ul + alloc > config.n_size_grid:
                        continue
                    current_prb_ul += alloc
                    ul_tx_done = True
                    harq_acc['retx_total'] += 1
                    current_tti_ul.users.append(
                        UeAllocation(ue_id, round(snr, 1), entry['mcs'], f"{alloc}x12x1",
                                     res["Phy Sizes"],
                                     tb_id=entry['tb_id'], harq_pid=entry['pid'],
                                     harq_rv=entry['rv'], is_retransmission=True,
                                     harq_soft_bytes=st['soft_ul'].get(entry['pid'], 0)))
                    st['busy_until'][entry['pid']] = tti + HARQ_RTT_TTIS
                    if random.random() < bler_from_snr(effective_snr, entry['mcs']):
                        if entry['rv'] < HARQ_MAX_RETX:
                            entry['rv'] += 1
                            entry['next_tti'] = tti + HARQ_RTT_TTIS
                            st['outcome'][entry['pid']] = 'nack'
                        else:
                            harq_acc['dropped'] += 1
                            st['outcome'][entry['pid']] = 'drop'
                    else:
                        st['outcome'][entry['pid']] = 'ack'
                    st['pending_ul'].remove(entry)

                # 2) новый пакет
                if not ul_tx_done:
                    pid = _free_pid(st)
                    if pid != -1 and current_prb_ul < config.n_size_grid:
                        payload_ul = random.choice([10, 20, 50, 100, 300])
                        res_ul = calculate_resources_for_packet(payload_ul, mcs, start_prb=current_prb_ul, scs_khz=config.scs_khz, n_size_grid=config.n_size_grid, num_layers=config.num_layers)
                        alloc_prbs_ul = min(res_ul["Allocated PRBs"], max_prb_per_ue)

                        if current_prb_ul + alloc_prbs_ul <= config.n_size_grid:
                            current_prb_ul += alloc_prbs_ul
                            tb_counter += 1
                            harq_acc['tb_total'] += 1
                            harq_acc['initial_tx'] += 1
                            soft = res_ul["Phy Sizes"].harq_soft_bytes
                            current_tti_ul.users.append(
                                UeAllocation(ue_id, round(snr, 1), mcs, f"{alloc_prbs_ul}x12x1",
                                             res_ul["Phy Sizes"],
                                             tb_id=tb_counter, harq_pid=pid, harq_rv=0,
                                             harq_soft_bytes=soft))
                            # gNB хранит мягкие биты UL до ACK - это реальная SRAM
                            st['soft_ul'][pid] = soft
                            st['busy_until'][pid] = tti + HARQ_RTT_TTIS
                            if random.random() < bler_from_snr(effective_snr, mcs):
                                harq_acc['initial_nack'] += 1
                                st['pending_ul'].append(
                                    {'tb_id': tb_counter, 'payload': payload_ul, 'mcs': mcs,
                                     'rv': 1, 'pid': pid, 'next_tti': tti + HARQ_RTT_TTIS})
                                st['outcome'][pid] = 'nack'
                            else:
                                st['outcome'][pid] = 'ack'

        # сохраняем заполненные TTI в соответствующие массивы
        array_of_ttis_dl.append(current_tti_dl)
        array_of_ttis_ul.append(current_tti_ul)

        # суммарный soft buffer UL по всем абонентам на этот TTI
        tt_soft = sum(sum(st['soft_ul'].values()) for st in harq_state.values())
        harq_acc['buffer_sum'] += tt_soft
        harq_acc['buffer_count'] += 1
        if tt_soft > harq_acc['buffer_max']:
            harq_acc['buffer_max'] = tt_soft

    return config, array_of_ttis_dl, array_of_ttis_ul, harq_acc

def recommend_buffer_architecture(stage_stats, num_ttis, tti_duration_ms,
                                  axi_bus_capacity_gbps=7.5, block_speeds_gbps=None):
    # Анализ шины и блоков - сравниваем требуемую скорость каждого блока с его быстродействием, а суммарный трафик - с ёмкостью шины AXI.
    # stage_stats: {стадия: {"read": байт, "write": байт, "peak": байт за самый нагруженный TTI}}
    if not stage_stats:
        return "Недостаточно данных по транзакциям для рекомендации"

    speeds = dict(DEFAULT_BLOCK_SPEEDS_GBPS)
    if block_speeds_gbps:
        speeds.update(block_speeds_gbps)
    tti_sec = tti_duration_ms / 1000.0

    total_bytes = sum(s["read"] + s["write"] for s in stage_stats.values())
    bus_req_gbps = (total_bytes / num_ttis / tti_sec) / 1e9
    bus_util = bus_req_gbps / axi_bus_capacity_gbps if axi_bus_capacity_gbps > 0 else 0.0

    # загрузка каждого блока: средняя (за симуляцию) и пиковая (самый нагруженный TTI)
    blocks = []
    for stage, s in stage_stats.items():
        req = ((s["read"] + s["write"]) / num_ttis / tti_sec) / 1e9
        peak_req = (s.get("peak", 0) / tti_sec) / 1e9
        sp = speeds.get(stage, 5.0)
        u = req / sp if sp > 0 else float("inf")
        pu = peak_req / sp if sp > 0 else float("inf")
        blocks.append((stage, req, peak_req, sp, u, pu))

    worst = max(blocks, key=lambda b: b[4])       # по средней загрузке
    worst_peak = max(blocks, key=lambda b: b[5])  # по пиковому TTI

    lines = [
        f"Шина AXI: требуется {bus_req_gbps:.2f} ГБ/с из {axi_bus_capacity_gbps} ГБ/с "
        f"(утилизация {bus_util*100:.1f}%) - шина и Shared RAM быстрые, узкое место ищем в блоках.",
        f"Самый загруженный блок: '{worst[0]}' - средняя загрузка {worst[4]*100:.0f}% "
        f"({worst[1]:.2f} ГБ/с из {worst[3]:.2f} ГБ/с).",
    ]
    if worst_peak[5] > 1.0:
        lines.append(
            f"ВЕРДИКТ: блок '{worst_peak[0]}' не успевает в пиковые TTI - требуется "
            f"{worst_peak[2]:.2f} ГБ/с при быстродействии {worst_peak[3]:.2f} ГБ/с "
            f"({worst_peak[5]*100:.0f}%). Пики сглаживает локальный SRAM-буфер перед этим "
            f"блоком (шина при этом загружена лишь на {bus_util*100:.0f}% - в ней проблемы нет)."
        )
    elif bus_util > 0.8:
        lines.append("ВЕРДИКТ: узкое место - сама шина AXI (утилизация > 80%). "
                     "Нужно расширять шину или поднимать тактовую частоту.")
    elif worst[4] > 0.8:
        lines.append(
            f"ВЕРДИКТ: шина в норме, но блок '{worst[0]}' близок к пределу ({worst[4]*100:.0f}%). "
            f"Рекомендуется локальный SRAM-буфер именно для этого блока, остальные - через общий Shared RAM."
        )
    else:
        lines.append("ВЕРДИКТ: ни шина, ни блоки не перегружены - Shared RAM через единую AXI-шину достаточна.")
    return "\n".join(lines)

def plot_resource_grid_heatmap(tti_slot, n_size_grid=273):
    # cursor идёт последовательно (как реальный планировщик), nan = свободный PRB
    grid = np.full(n_size_grid, np.nan)
    cursor = 0
    for ue in tti_slot.users:
        prbs = int(ue.format_prb.split('x')[0])
        end = min(cursor + prbs, n_size_grid)
        grid[cursor:end] = ue.ue_id
        cursor = end
        if cursor >= n_size_grid:
            break

    cmap = plt.cm.tab20
    cmap.set_bad(color='lightgrey')

    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(grid.reshape(1, -1), aspect='auto', cmap=cmap)
    ax.set_title(f'Resource Grid DL: распределение PRB между UE (TTI={tti_slot.tti_index})')
    ax.set_xlabel('PRB index')
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label='UE_ID')
    fig.tight_layout()
    return fig

def plot_users_dynamics(df, num_ttis, tti_duration_ms):
    users_per_tti = df[df['Direction'] == 'DL'].groupby('TTI')['UE_ID'].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(users_per_tti.index, users_per_tti.values, drawstyle='steps-post', color='#1f77b4', linewidth=2.5)
    ax.set_title(f'Динамика активных абонентов в соте (Симуляция {num_ttis * tti_duration_ms / 1000} сек)', fontsize=14, pad=15)
    ax.set_xlabel('Время (TTI)', fontsize=12)
    ax.set_ylabel('Количество активных пользователей', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 16)
    ax.set_xlim(0, num_ttis)
    fig.tight_layout()
    return fig

def plot_mcs_histogram(df):
    df_dl = df[df['Direction'] == 'DL']
    mcs_counts = df_dl['MCS'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(mcs_counts.index, mcs_counts.values, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('MCS Index')
    ax.set_ylabel('Количество TTI-аллокаций')
    ax.set_title('Распределение MCS (DL) - полная таблица 28 индексов')
    ax.set_xticks(range(0, 28))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

def plot_axi_waterfall(metrics):
    # накопление нагрузки на AXI-шину по блокам конвейера (среднее за TTI) - каждый блок кладёт/забирает из Shared RAM свою порцию - с каждым блоком растёт суммарный трафик шины
    stage_stats = metrics["stage_stats"]
    num_ttis = metrics["num_ttis"]
    order = [s for s in AXI_PDSCH_ORDER + AXI_PDCCH_ORDER if s in stage_stats]
    order += [s for s in stage_stats if s not in order]

    reads  = [stage_stats[s]["read"]  / num_ttis / 1024 for s in order]   # КБ/TTI
    writes = [stage_stats[s]["write"] / num_ttis / 1024 for s in order]
    cum = np.cumsum([r + w for r, w in zip(reads, writes)])

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(len(order))
    ax.bar(x, reads, color='#4c72b0', label='чтение из Shared RAM')
    ax.bar(x, writes, bottom=reads, color='#dd8452', label='запись в Shared RAM')
    ax.plot(x, cum, color='#222222', marker='o', linewidth=2, label='накопленная нагрузка на шину')
    for xi, ci in zip(x, cum):
        ax.annotate(f'{ci:.0f}', (xi, ci), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=8, color='#222222')

    # разделитель между трактом данных (PDSCH) и управляющим трактом (PDCCH)
    n_data = len([s for s in AXI_PDSCH_ORDER if s in stage_stats])
    if 0 < n_data < len(order):
        ax.axvline(n_data - 0.5, color='#999999', linestyle='--', linewidth=1)
        ax.text(n_data - 0.35, max(cum) * 0.5, 'управляющий тракт (PDCCH)',
                fontsize=8, color='#666666', rotation=90, va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('КБ / TTI (среднее за симуляцию)')
    ax.set_title('Накопление нагрузки на AXI-шину по блокам конвейера')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return fig

def plot_soc_load_diagram(metrics, config):
    # SoC-схема с блоками
    #Для каждого блока - чтение/запись КБ за TTI и требуемая скорость против его быстродействия. Цвет блока = загрузка - зелёный < 50%, жёлтый 50-80%, красный выше.
    stage_stats = metrics["stage_stats"]
    speeds = metrics["block_speeds_gbps"]
    num_ttis = metrics["num_ttis"]
    tti_sec = config.tti_duration_ms / 1000.0
    cap = metrics["axi_bus_capacity_gbps"]

    total_avg_kb = metrics["axi_avg_per_tti"] / 1024
    peak_kb = metrics["axi_max_per_tti"] / 1024

    fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_title(f'SoC 5G Downlink: нагрузка блоков через Shared RAM / AXI4 '
                 f'(симуляция {num_ttis} TTI × {config.tti_duration_ms} мс)', fontsize=12, pad=14)

    # Shared RAM
    ax.add_patch(FancyBboxPatch((22, 83), 56, 11, boxstyle='round,pad=0.4',
                                facecolor='#d6e4ff', edgecolor='#1a3c6e', linewidth=1.5))
    ax.text(50, 88.5, f'Shared RAM - через шину проходит {total_avg_kb:.1f} КБ/TTI',
            ha='center', va='center', fontsize=10)

    # AXI4 BUS 
    ax.add_patch(Rectangle((3, 63), 94, 8, facecolor='#222831', edgecolor='black'))
    ax.text(50, 67, f'AXI4 BUS   {total_avg_kb:.1f} КБ/TTI (пик {peak_kb:.1f})   '
                    f'утилизация {metrics["bus_util_avg_pct"]:.2f}% из {cap} ГБ/с',
            color='white', ha='center', va='center', fontsize=10)

    # блоки конвейера 
    stages = [s for s in AXI_PDSCH_ORDER + AXI_PDCCH_ORDER if s in stage_stats]
    n = len(stages)
    w, gap = 12.0, (94 - n * 12.0) / max(n - 1, 1)
    y0, h = 16, 30
    for i, stage in enumerate(stages):
        x = 3 + i * (w + gap)
        d = stage_stats[stage]
        r_kb = d["read"] / num_ttis / 1024
        w_kb = d["write"] / num_ttis / 1024
        req = ((d["read"] + d["write"]) / num_ttis / tti_sec) / 1e9
        peak_req = (d.get("peak", 0) / tti_sec) / 1e9
        sp = speeds.get(stage, 5.0)
        u = req / sp if sp > 0 else float('inf')
        pu = peak_req / sp if sp > 0 else float('inf')

        face = '#c8e6c9' if u < 0.5 else ('#fff3c4' if u < 0.8 else '#ffcdd2')
        ax.add_patch(Rectangle((x, y0), w, h, facecolor=face, edgecolor='#333333', linewidth=1.2))
        ax.annotate('', xy=(x + w / 2, 63), xytext=(x + w / 2, y0 + h),
                    arrowprops=dict(arrowstyle='<->', color='#555555', linewidth=1.2))

        name = stage if len(stage) <= 11 else stage.replace(' ', '\n', 1)
        ax.text(x + w / 2, y0 + h - 2.5, name, ha='center', va='top',
                fontsize=8, fontweight='bold')
        ax.text(x + w / 2, y0 + h * 0.52, f'Ч {r_kb:.1f} КБ\nЗ {w_kb:.1f} КБ/TTI',
                ha='center', va='center', fontsize=7.5)
        ax.text(x + w / 2, y0 + 4.5, f'треб {req:.2f}/{sp:.2f} ГБ/с',
                ha='center', fontsize=7.5, color='#b71c1c' if u > 0.8 else '#1b5e20')
        if pu > 1.0:
            ax.text(x + w / 2, y0 - 3.5, 'ПЕРЕГРУЗКА (пик)', ha='center',
                    fontsize=7.5, color='#b71c1c', fontweight='bold')

    # легенда загрузки 
    for j, (c, lbl) in enumerate([('#c8e6c9', 'загрузка < 50%'),
                                  ('#fff3c4', 'загрузка 50–80%'),
                                  ('#ffcdd2', 'загрузка > 80%')]):
        lx = 14 + j * 25
        ax.add_patch(Rectangle((lx, 5.5), 3, 3.5, facecolor=c, edgecolor='#333333'))
        ax.text(lx + 4, 7.2, lbl, fontsize=8, va='center')
    ax.text(85, 7.2, 'Ч - чтение, З - запись', fontsize=8, va='center', color='#555555')
    return fig

def run_simulation(config, num_ttis, axi_bus_capacity_gbps=7.5, block_speeds_gbps=None):
    # генерирует сценарий и считает все метрики. 
    _, dl_scenario, ul_scenario, harq_acc = generate_simulation_scenario(num_ttis=num_ttis, config=config)

    records = []
    for tti_dl, tti_ul in zip(dl_scenario, ul_scenario):
        for ue in tti_dl.users:
            records.append({
                "TTI": tti_dl.tti_index, "Direction": "DL", "UE_ID": ue.ue_id,
                "SNR_dB": ue.snr, "MCS": ue.mcs, "Resources": ue.format_prb,
                "PRBs": int(ue.format_prb.split('x')[0]),
                "Payload_Bits": ue.phy_sizes.tb_bits, "Codec": ue.phy_sizes.base_graph,
                "AXI_Load_Bytes": ue.phy_sizes.axi_bus_bytes,
                "PDCCH_AXI_Bytes": ue.pdcch_axi_bytes,
                "TB_ID": ue.tb_id, "HARQ_PID": ue.harq_pid, "HARQ_RV": ue.harq_rv,
                "Is_Retx": ue.is_retransmission,
                "HARQ_Soft_Bytes": 0,  # DL - передатчик gNB мягкие биты не хранит
            })
        for ue in tti_ul.users:
            records.append({
                "TTI": tti_ul.tti_index, "Direction": "UL", "UE_ID": ue.ue_id,
                "SNR_dB": ue.snr, "MCS": ue.mcs, "Resources": ue.format_prb,
                "PRBs": int(ue.format_prb.split('x')[0]),
                "Payload_Bits": ue.phy_sizes.tb_bits, "Codec": ue.phy_sizes.base_graph,
                "AXI_Load_Bytes": ue.phy_sizes.axi_bus_bytes,
                "PDCCH_AXI_Bytes": 0,
                "TB_ID": ue.tb_id, "HARQ_PID": ue.harq_pid, "HARQ_RV": ue.harq_rv,
                "Is_Retx": ue.is_retransmission,
                "HARQ_Soft_Bytes": ue.harq_soft_bytes,  # UL - приёмник gNB хранит мягкие биты
            })

    df = pd.DataFrame(records)
    df['Total_AXI_Bytes'] = df['AXI_Load_Bytes'] + df['PDCCH_AXI_Bytes']

    # трафик AXI по стадиям конвейера - сумма за симуляцию + пик за TTI.
    stage_stats = {}
    for tti_slot in dl_scenario + ul_scenario:
        per_tti = {}
        for ue in tti_slot.users:
            # транзакции трактa данных (PDSCH) + управляющего (PDCCH, только DL)
            for t in list(ue.phy_sizes.axi_transactions) + list(ue.pdcch_transactions):
                d = per_tti.setdefault(t.stage, [0, 0])
                d[0] += t.read_bytes
                d[1] += t.write_bytes
        for stage, (r, w) in per_tti.items():
            d = stage_stats.setdefault(stage, {"read": 0, "write": 0, "peak": 0})
            d["read"] += r
            d["write"] += w
            d["peak"] = max(d["peak"], r + w)

    total_time_sec = num_ttis * config.tti_duration_ms / 1000.0
    dl_payload_per_tti = df[df['Direction'] == 'DL'].groupby('TTI')['Payload_Bits'].sum()
    avg_kb_per_tti = (dl_payload_per_tti.mean() / 8) / 1024
    max_kb_per_tti = (dl_payload_per_tti.max() / 8) / 1024
    cell_throughput_mbps = (dl_payload_per_tti.sum() / 1e6) / total_time_sec

    user_traffic_bits = df.groupby(['UE_ID', 'Direction'])['Payload_Bits'].sum().unstack().fillna(0)
    user_traffic_kb = user_traffic_bits / 8 / 1024
    user_throughput_mbps = (user_traffic_bits / 1e6) / total_time_sec
    user_stats = pd.DataFrame({
        'Скачал всего (DL, КБ)': user_traffic_kb.get('DL', 0).round(2),
        'Отправил всего (UL, КБ)': user_traffic_kb.get('UL', 0).round(2),
        'Скорость DL (Мбит/с)': user_throughput_mbps.get('DL', 0).round(2),
        'Скорость UL (Мбит/с)': user_throughput_mbps.get('UL', 0).round(2),
    })

    axi_per_tti = df.groupby('TTI')['Total_AXI_Bytes'].sum()

    # требуемая скорость шины: байты за TTI / длительность TTI -> сравнение с ёмкостью
    tti_sec = config.tti_duration_ms / 1000.0
    bus_util_avg_pct = (axi_per_tti.mean() / tti_sec / 1e9) / axi_bus_capacity_gbps * 100
    bus_util_peak_pct = (axi_per_tti.max() / tti_sec / 1e9) / axi_bus_capacity_gbps * 100

    metrics = {
        "total_time_sec": total_time_sec,
        "total_records": len(df),
        "avg_kb_per_tti": avg_kb_per_tti,
        "max_kb_per_tti": max_kb_per_tti,
        "cell_throughput_mbps": cell_throughput_mbps,
        "user_stats": user_stats,
        "pdsch_axi_total": df['AXI_Load_Bytes'].sum(),
        "pdcch_axi_total": df['PDCCH_AXI_Bytes'].sum(),
        "axi_avg_per_tti": axi_per_tti.mean(),
        "axi_max_per_tti": axi_per_tti.max(),
        "codec_counts": df['Codec'].value_counts(),
        #  шина и быстродействие блоков SoC 
        "axi_bus_capacity_gbps": axi_bus_capacity_gbps,
        "block_speeds_gbps": {**DEFAULT_BLOCK_SPEEDS_GBPS, **(block_speeds_gbps or {})},
        "stage_stats": stage_stats,
        "bus_util_avg_pct": bus_util_avg_pct,
        "bus_util_peak_pct": bus_util_peak_pct,
        "buffer_recommendation": recommend_buffer_architecture(
            stage_stats, num_ttis, config.tti_duration_ms,
            axi_bus_capacity_gbps=axi_bus_capacity_gbps, block_speeds_gbps=block_speeds_gbps),
        #  HARQ-статистика 
        "harq_tb_total": harq_acc['tb_total'],
        "harq_retx_total": harq_acc['retx_total'],
        "harq_dropped": harq_acc['dropped'],
        "harq_bler_effective": (harq_acc['initial_nack'] / harq_acc['initial_tx']) if harq_acc['initial_tx'] else 0.0,
        "harq_avg_retx_per_tb": (harq_acc['retx_total'] / harq_acc['tb_total']) if harq_acc['tb_total'] else 0.0,
        "harq_buffer_max_bytes": harq_acc['buffer_max'],
        "harq_buffer_avg_bytes": (harq_acc['buffer_sum'] / harq_acc['buffer_count']) if harq_acc['buffer_count'] else 0.0,
        "dl_scenario": dl_scenario,
        "ul_scenario": ul_scenario,
        "num_ttis": num_ttis,
    }
    return df, metrics

def format_console_report(df, metrics, config):
    # текстовый отчёт в исходном консольном формате
    duplex_info = f"TDD [{config.tdd_pattern}]" if config.duplex == "TDD" else "FDD"
    lines = []
    lines.append(f"=== Конфигурация: {duplex_info} | BW: {config.bw_mhz}MHz | SCS: {config.scs_khz}kHz "
                  f"| μ={config.mu} | NSizeGrid: {config.n_size_grid} PRB | TTI: {config.tti_duration_ms}ms ===")
    lines.append(f"Всего записей в симуляции: {metrics['total_records']}")
    lines.append("\n--- Фрагмент базы данных (первые 10 операций) ---")
    lines.append(df.head(10).to_string(index=False))
    lines.append("\n--- Пропускная способность соты (Cell Capacity) ---")
    lines.append(f"Средний объем переданных данных (DL) за один TTI: {round(metrics['avg_kb_per_tti'], 2)} КБ")
    lines.append(f"Пиковый объем переданных данных (DL) за один TTI:  {round(metrics['max_kb_per_tti'], 2)} КБ")
    lines.append(f"Реальная средняя скорость соты (DL Cell Throughput): {round(metrics['cell_throughput_mbps'], 2)} Мбит/с")
    lines.append(f"\n--- Статистика по Абонентам за время симуляции ({metrics['total_time_sec']} сек) ---")
    lines.append(metrics['user_stats'].to_string())
    lines.append("\n--- Нагрузка на шину AXI (Hardware Performance) ---")
    lines.append(f"PDSCH AXI (сумма): {metrics['pdsch_axi_total']} байт")
    lines.append(f"PDCCH AXI (сумма): {metrics['pdcch_axi_total']} байт")
    lines.append(f"Средняя нагрузка: {round(metrics['axi_avg_per_tti'], 2)} байт / TTI")
    lines.append(f"Пиковая (Максимальная) нагрузка: {metrics['axi_max_per_tti']} байт / TTI")
    lines.append(f"Ёмкость шины AXI: {metrics['axi_bus_capacity_gbps']} ГБ/с → утилизация: "
                 f"средняя {metrics['bus_util_avg_pct']:.2f} % / пиковая {metrics['bus_util_peak_pct']:.2f} %")
    lines.append("\n--- Быстродействие блоков SoC (среднее за TTI) ---")
    speeds = metrics['block_speeds_gbps']
    tti_sec = config.tti_duration_ms / 1000.0
    num_ttis_r = metrics['num_ttis']
    order = [s for s in AXI_PDSCH_ORDER + AXI_PDCCH_ORDER if s in metrics['stage_stats']]
    order += [s for s in metrics['stage_stats'] if s not in order]
    lines.append(f"{'Блок':<18}{'Чтение':>9}{'Запись':>9}{'Треб.':>8}{'Есть':>7}{'Загрузка':>9}")
    lines.append(f"{'':<18}{'КБ/TTI':>9}{'КБ/TTI':>9}{'ГБ/с':>8}{'ГБ/с':>7}{'':>9}")
    for stage in order:
        s = metrics['stage_stats'][stage]
        r_kb = s['read'] / num_ttis_r / 1024
        w_kb = s['write'] / num_ttis_r / 1024
        req = ((s['read'] + s['write']) / num_ttis_r / tti_sec) / 1e9
        sp = speeds.get(stage, 5.0)
        lines.append(f"{stage:<18}{r_kb:>9.1f}{w_kb:>9.1f}{req:>8.2f}{sp:>7.2f}{req/sp*100:>8.0f}%")
    lines.append("\n--- HARQ (гибридный автоматический повтор) ---")
    lines.append(f"Транспортных блоков (TB): {metrics['harq_tb_total']} | Ретрансмиссий: {metrics['harq_retx_total']} | Отброшено после RV{HARQ_MAX_RETX}: {metrics['harq_dropped']}")
    lines.append(f"BLER первой передачи: {metrics['harq_bler_effective'] * 100:.1f} %")
    lines.append(f"Среднее число ретрансмиссий на TB: {round(metrics['harq_avg_retx_per_tb'], 2)}")
    lines.append(f"Soft buffer UL (приёмник gNB): пик {metrics['harq_buffer_max_bytes']:,} байт, средний {round(metrics['harq_buffer_avg_bytes']):,} байт")
    lines.append("\n--- Использование кодеков LDPC ---")
    lines.append(metrics['codec_counts'].to_string())
    lines.append("\n--- Рекомендация архитектуры буфера ---")
    lines.append(metrics['buffer_recommendation'])
    return "\n".join(lines)

# RE-level визуализация одного TTI (слота) 5G NR.
# Показывает каждый Resource Element (1 поднесущая × 1 OFDM-символ)
# с цветовой кодировкой по типу канала:
#   - серый      : пустой RE (свободен)
#   - чёрный     : PDCCH (управляющий канал, символы 0–1 - типовая CORESET)
#   - золотой    : DM-RS (опорный сигнал, символ 7 - типовая позиция)
#   - цветной    : PDSCH абонента (каждый UE - свой цвет из палитры tab20)
#
# Параметр max_prb ограничивает количество показываемых PRB - иначе при n_size_grid=273 картинка получается слишком вытянутой (по умолчанию 50)

# Цветовая палитра каналов для RE-level визуализации
COLOR_EMPTY  = '#f5f5f5'   # светло-серый - пустой RE
COLOR_PDCCH  = '#222222'   # чёрный - управляющий канал
COLOR_DMRS   = '#ffd700'   # золотистый - опорный сигнал (Demodulation Reference Signal)

# Палитра для абонентов (UE) - берём из matplotlib tab20
UE_PALETTE = plt.cm.tab20

def plot_re_level_grid(tti_slot, config, max_prb=50):
    """
    Строит RE-level визуализацию одного TTI
    Параметры:
        tti_slot : TtiSlot  - один интервал с списком абонентов (берётся из metrics["dl_scenario"])
        config : CellConfig - конфигурация соты (нужны n_size_grid, scs_khz)
        max_prb : int       - сколько PRB показать (считая с нулевого), а остальные обрезаются
            При n_size_grid=273 показывать все - картинка будет слишком вытянутой.
    Возвращает:
        matplotlib.figure.Figure - фигура для st.pyplot() или plt.show().
    """
    n_total_prb = config.n_size_grid
    n_show_prb = min(max_prb, n_total_prb)
    n_subcarriers = n_show_prb * 12
    n_symbols = 14  # 14 OFDM-символов в слоте 5G при Normal CP

    # Символы, занятые управляющим каналом и опорниками (PDSCH туда не пишем)
    PDCCH_SYMBOLS = [0, 1]          # CORESET - начало слота
    DMRS_SYMBOL   = 3               # типовая позиция front-loaded DM-RS
    PDSCH_SYMBOLS = [s for s in range(n_symbols)
                     if s not in PDCCH_SYMBOLS and s != DMRS_SYMBOL]
     
    # 1. Инициализируем матрицу RE
     
    # коды - 0 = пусто, -1 = PDCCH, -2 = DM-RS, >0 = UE_ID (PDSCH)
    grid = np.zeros((n_subcarriers, n_symbols), dtype=int)

    # PDCCH - на всех поднесущих, символы 0 и 1
    grid[:, PDCCH_SYMBOLS] = -1

    # DM-RS - на всех поднесущих, один символ
    grid[:, DMRS_SYMBOL] = -2
     
    # 2. Для каждого UE - заполняем его PDSCH область напрямую.
    # Не вызываем nrPDSCHIndices - мы уже знаем PRB-множество UE и какие символы отданы под PDSCH.
     
    cursor_prb = 0
    ue_on_screen = []
    retx_boxes = []   # (sc_start, sc_end) для пунктирных рамок ретрансмиссий

    for ue in tti_slot.users:
        n_prb = int(ue.format_prb.split('x')[0].strip())
        start_prb = cursor_prb
        end_prb = min(start_prb + n_prb, n_total_prb)
        cursor_prb = end_prb

        # Если UE целиком вне показанной области - пропускаем
        if start_prb >= n_show_prb:
            continue

        end_prb_show = min(end_prb, n_show_prb)
        sc_start = start_prb * 12
        sc_end   = end_prb_show * 12

        # Заполняем PDSCH только на символах данных (не PDCCH, не DM-RS)
        for sym in PDSCH_SYMBOLS:
            grid[sc_start:sc_end, sym] = ue.ue_id

        if ue.is_retransmission:
            retx_boxes.append((sc_start, sc_end))

        ue_on_screen.append(ue)
     
    # 3. Строим RGB-матрицу
     
    rgb = np.ones((n_subcarriers, n_symbols, 3))

    rgb[grid == 0]   = [0.96, 0.96, 0.96]   # пустые
    rgb[grid == -1]  = [0.13, 0.13, 0.13]   # PDCCH
    rgb[grid == -2]  = [1.0, 0.84, 0.0]     # DM-RS

    ue_ids = sorted({ue.ue_id for ue in ue_on_screen})
    ue_to_color = {ue_id: UE_PALETTE(i % 20)[:3] for i, ue_id in enumerate(ue_ids)}
    for ue_id in ue_ids:
        rgb[grid == ue_id] = ue_to_color[ue_id]
     
    # 4. Рисуем
     
    fig_h = max(6.0, n_show_prb / 8.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    ax.imshow(rgb, aspect='auto', interpolation='nearest',
              extent=[-0.5, n_symbols - 0.5, n_subcarriers - 0.5, -0.5])

    ax.set_xlabel('OFDM Symbol Index', fontsize=11)
    ax.set_ylabel('Subcarrier Index', fontsize=11)
    ax.set_title(
        f'RE-level Resource Grid - TTI #{tti_slot.tti_index}\n'
        f'{config.duplex} | SCS {config.scs_khz} kHz | '
        f'показано {n_show_prb} из {n_total_prb} PRB | '
        f'активных UE: {len(ue_on_screen)}',
        fontsize=11, pad=12
    )

    ax.set_xticks(range(n_symbols))
    ax.set_xticklabels([str(i) for i in range(n_symbols)], fontsize=8)

    step = max(1, n_show_prb // 15)
    prb_ticks = list(range(0, n_show_prb, step))
    sc_ticks = [p * 12 + 6 for p in prb_ticks]
    ax.set_yticks(sc_ticks)
    ax.set_yticklabels([f'PRB {p}' for p in prb_ticks], fontsize=8)

    # Тонкие линии между PRB
    for p in range(0, n_show_prb + 1, step):
        ax.axhline(y=p * 12 - 0.5, color='gray', linewidth=0.4, alpha=0.6)
    # Тонкие линии между символами
    for s in range(n_symbols + 1):
        ax.axvline(x=s - 0.5, color='gray', linewidth=0.3, alpha=0.5)
     
    # 4.1. Толстые разделители зон 
     
    # Граница между PDCCH и PDSCH (после символа 1)
    ax.axvline(x=1.5, color='#1a1a1a', linewidth=2.0, alpha=0.9)
    # Граница вокруг DM-RS (до и после символа 3)
    ax.axvline(x=2.5, color='#1a1a1a', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.axvline(x=3.5, color='#1a1a1a', linewidth=1.5, alpha=0.7, linestyle='--')
     
    # 4.2. Подписи зон сверху
     
    y_zone = -n_subcarriers * 0.04   # чуть ниже нижней границы
    ax.text(0.5, y_zone, 'PDCCH\n(Control)', ha='center', va='top',
            fontsize=8, fontweight='bold', color='#222222')
    ax.text(3.0, y_zone, 'DM-RS\n(Reference)', ha='center', va='top',
            fontsize=8, fontweight='bold', color='#996600')
    ax.text((4 + n_symbols - 1) / 2.0, y_zone, 'PDSCH (User Data)',
            ha='center', va='top', fontsize=8, fontweight='bold',
            color='#444444')
     
    # 4.3 Рамки ретрансмиссий HARQ - белый пунктир вокруг их PRB-блока
     
    for (sc_s, sc_e) in retx_boxes:
        ax.add_patch(Rectangle(
            (1.5, sc_s - 0.5),                      # от конца PDCCH-зоны
            (n_symbols - 0.5) - 1.5,                # до конца слота
            sc_e - sc_s,
            fill=False, edgecolor='white',
            linewidth=1.4, linestyle='--'))
     
    # 5. Легенда
     
    legend_elements = [
        Patch(facecolor=COLOR_EMPTY, edgecolor='gray', label='Empty'),
        Patch(facecolor=COLOR_PDCCH, label='PDCCH'),
        Patch(facecolor=COLOR_DMRS, label='DM-RS'),
    ]
    for ue in ue_on_screen:
        c = ue_to_color[ue.ue_id]
        rv_tag = f", retx RV{ue.harq_rv}" if ue.is_retransmission else ""
        legend_elements.append(
            Patch(facecolor=c, edgecolor='black', linewidth=0.5,
                  label=f'PDSCH UE #{ue.ue_id} (MCS {ue.mcs}, SNR {ue.snr} dB{rv_tag})')
        )

    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.95,
              ncol=1, title='Channels', title_fontsize=9)

    # Расширяем нижний отступ для подписей зон
    ax.margins(y=0.08)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # запуск напрямую из консоли
    config = CellConfig(bw_mhz=100, scs_khz=30, duplex="TDD", tdd_pattern="DDDSU")
    num_sim_ttis = 6000

    df, metrics = run_simulation(config, num_sim_ttis)
    report = format_console_report(df, metrics, config)
    print(report)

    df.to_csv('simulation_results.csv', index=False)
    print("\nДанные сохранены в файл simulation_results.csv")

    best_tti = max(metrics["dl_scenario"], key=lambda t: sum(
        int(ue.format_prb.split('x')[0]) for ue in t.users))
    plot_resource_grid_heatmap(best_tti, n_size_grid=config.n_size_grid)
    plt.show()

    plot_users_dynamics(df, num_sim_ttis, config.tti_duration_ms)
    plt.show()

    plot_mcs_histogram(df)
    plt.show()

    # RE-level визуализация для самого загруженного TTI
    plot_re_level_grid(best_tti, config, max_prb=50)
    plt.show()