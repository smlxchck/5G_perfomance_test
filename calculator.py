import py3gpp
import numpy as np

def calculate_resources_for_packet(packet_size_bytes):
    ## настройка общей сетки базовой станции (carrier)
    carrier = py3gpp.nrCarrierConfig()
    carrier.NSizeGrid = 273        # 273 ресурсных блока (для полосы 100 МГц)
    carrier.SubcarrierSpacing = 30 # нумерология 30 кГц
    
    # настраиваем выделение ресурсов под нашего абонента (PDSCH - канал данных)
    pdsch = py3gpp.nrPDSCHConfig()
    
    ## логика планировщика
    # Чем больше пакет, тем больше ресурсных блоков (PRB) мы ему даем.
    # если 1 PRB может перенести 10 байт данных.
    # функция max(1, ...) гарантирует что даже для 1 байта мы выделим минимум 1 PRB.
    import math
    allocated_prb = max(1, math.ceil(packet_size_bytes / 10)) 

    # определение кодека (LDPC BG1 BG2)
    codec_type = "LDPC Base Graph 2 (для мелких пакетов)"
    # Сделаем порог переключения кодека меньше, чтобы на пакете 100 байт он переключился
    if packet_size_bytes >= 100: 
        codec_type = "LDPC Base Graph 1 (для крупных пакетов)"
    
    # исправление бага библиотеки 
    pdsch.NStartBWP = 0                       
    pdsch.NSizeBWP = carrier.NSizeGrid         
    pdsch.PRBSet = list(range(allocated_prb)) # передаем ресурсы как список чисел
    
    # выделяем символы в слоте
    pdsch.SymbolAllocation = [2, 12] 
    pdsch.Modulation = 'qpsk'
    
    ## вычисляем Resource Elements
    result_indices = py3gpp.nrPDSCHIndices(carrier, pdsch)
    
    # библиотека может вернуть просто массив индексов, либо кортеж (индексы, инфо).
    if isinstance(result_indices, tuple):
        ind = result_indices[0] # берем первый элемент
    else:
        ind = result_indices    # если вернулся только один массив
        
    num_re = len(ind) # количество загруженных Resouce Elements
    
    ## определение кодека (LDPC BG1 BG2) (3GPP TS 38.212)
    # по стандарту 3GPP если Transport Block Size - размер данных - маленький то BG2.
    # если большой то BG1.
    A = packet_size_bytes * 8  # Считаем размер в битах
    R = 0.5                    # Скорость кодирования (Code Rate) - можно менять от 0.1 до 0.9 (имитация связи)

    if A <= 292:
        codec_type = "LDPC Base Graph 2"
    elif A <= 3824 and R <= 0.67:
        codec_type = "LDPC Base Graph 2"
    elif R <= 0.25:
        codec_type = "LDPC Base Graph 2"
    else:
        codec_type = "LDPC Base Graph 1"
        
    ## cчитаем трафик для шины AXI
    # каждый Resource Elements обычно представляется комплексным числом I/Q
    # Если I и Q это 16-битные числа (2 байта), то 1 RE = 4 байта сырых данных.
    axi_bus_bytes = num_re * 4

    return {
        "Packet Size": f"{packet_size_bytes} байт",
        "Allocated PRBs": allocated_prb,
        "Resource Elements Used": num_re,
        "Codec": codec_type,
        "AXI Bus Load": f"{axi_bus_bytes} байт сырых I/Q данных"
    }

# сценарий c пакетами
packets_queue =[1000, 100, 50, 2, 1]

for pkt in packets_queue:
    result = calculate_resources_for_packet(pkt)
    print(result)