# Скрипт (CLI) для анализа котировок с Yahoo.Finance

Предварительно в каталоге со скриптом создать файл tickers.txt с названием тикеров:
AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, BRK-B, UNH, JPM (каждый на отдельной строке)

usage:
tickers.py [-h] [-t TICKERS] [-s START_DATE]

options:
-h, --help          show this help message and exit
-t TICKERS, --tickers TICKERS
                    path to file, default 'tickers.csv'
-s START_DATE, --start_date START_DATE
                    start_date in format 'dd.mm.yy', default 01.01.20

Результат работы:
1. В каталоге /charts в файлах *.csv сохраняются данные котировок ('timestamp', 'adjclose', 'normalize')
1. Сохраняется файл charts_<дата>.png с графиком котировок (можно выводить окно график)