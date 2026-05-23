# Исследование и практическая реализация моделей генерации трафика для высоконагруженного тестирования базовых станций пятого поколеения и их компонентов


[Отчет о ходе работ](OHR.ipynb)

## Requirements 

Для работы [калькулятора](calculator.py) необходим Python 3.9+ и следующие библиотеки:
[py3gpp](https://github.com/catkira/py3gpp) - ядро для расчета частотно-временной сетки и параметров 5G NR.
[pandas](https://github.com/pandas-dev/pandas) - сбор, структурирование и аналитическая обработка результатов симуляции.
[numpy](https://github.com/numpy/numpy) - математические операции с массивами данных.
[matplotlib](https://github.com/matplotlib/matplotlib) - визуализация динамики абонентов в соте.

## Publications

 - https://www.researchgate.net/publication/325862061_5G_New_Radio_Unveiling_the_Essentials_of_the_Next_Generation_Wireless_Access_Technology
 - https://sharetechnote.com/html/5G/5G_PDSCH.html - Base Graph description
 - https://www.sharetechnote.com/html/5G/5G_ResourceGrid.html - Article about Resource Grid
 - https://www.techplayon.com/5g-nr-modulation-and-coding-scheme-modulation-and-code-rate/ - 5G Modulation and Coding Scheme
 - https://www.techplayon.com/5g-nr-numerology-subcarrier-spcaing-scs/ - Subcarrier Spacing in 5G (for flexible SCS in calculator)