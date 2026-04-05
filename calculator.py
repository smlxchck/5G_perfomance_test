import py3gpp
import numpy as np
import math
import random
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
    
    allocated_prb = max(1, math.ceil((packet_size_bytes * 8) / bits_per_prb))
    
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
    change_probability = 0.01  # шанс изменения (0.01 = 1%)
    max_cell_users = 15        # максимум пользователей в соте
    min_cell_users = 1         # минимум пользователей
    max_prb_per_ue = 40        # лимит ресурсов на одного (чтобы всем хватило места) так как 273 ресурсных блока


    # стартовый пул пользователей
    active_users = {i: random.uniform(10.0, 25.0) for i in range(1, start_users_count + 1)}
    
    for tti in range(num_ttis):
        current_tti = TtiSlot(tti_index=tti)
        current_prb = 0
        
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
            
            # размер пакета т.е. трафик
            payload = random.choice([50, 100, 300, 600, 1000])
            
            # вызываем расчет RE и AXI через py3gpp
            res = calculate_resources_for_packet(payload, mcs, start_prb=current_prb)
            
            # планировщик - ограничиваем PRB, чтобы не забить всю соту одним пользователем
            allocated_prbs = min(res["Allocated PRBs"], max_prb_per_ue)
            
            # проверка ресурсов соты (273 PRB)
            if current_prb + allocated_prbs > 273: 
                break 
                
            current_prb += allocated_prbs
            
            # запись в TTI
            current_tti.users.append(
                UeAllocation(ue_id, round(snr, 1), mcs, f"{allocated_prbs}x12x1", res["Phy Sizes"])
            )
            
        array_of_ttis_dl.append(current_tti)
        array_of_ttis_ul.append(TtiSlot(tti_index=tti))
        
    return config, array_of_ttis_dl, array_of_ttis_ul

# симуляция
config, dl_scenario, ul_scenario = generate_simulation_scenario(num_ttis=100) # каждый tti = 0.5 мс

print(f"=== Конфигурация: {config.duplex} | BW: {config.bw_mhz}MHz | SCS: {config.scs_khz}kHz | TTI: {config.tti_duration_ms}ms ===")
for tti in dl_scenario:
    print(f"\n[ Downlink TTI № {tti.tti_index} ] Пользователей: {len(tti.users)}")
    for ue in tti.users:
        print(f"  UE {ue.ue_id} | SNR: {ue.snr}dB -> MCS: {ue.mcs} | Ресурсы: {ue.format_prb}")
        print(f"      PHY: Payload {ue.phy_sizes.tb_bits}b -> CRC {ue.phy_sizes.tb_crc_bits}b -> {ue.phy_sizes.base_graph}")
        print(f"      AXI Load: {ue.phy_sizes.axi_bus_bytes} байт")