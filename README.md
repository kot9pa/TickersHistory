# Скрипт (CLI) для получения котировок с Yahoo.Finance
## Установка окружения
1. Установить python (3.12+) и pip
2. В рабочем каталоге установить пакет tickers_history (*.tar.gz), пример:
`pip install tickers_history-0.0.1.tar.gz`
3. Запустить командой `tickers_history`

## Использование скрипта
Предварительно в каталоге со скриптом создать файл tickers.txt с названием тикеров:
AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, BRK-B, UNH, JPM (каждый на отдельной строке)

usage:
tickers_history [-h] [-t TICKERS] [-s START_DATE]

options:
-h, --help          show this help message and exit
-t TICKERS, --tickers TICKERS
                    path to file, default 'tickers.txt'
-s START_DATE, --start_date START_DATE
                    start_date in format 'dd.mm.yy', default 01.01.20

Результат работы:
1. В каталоге /charts в файлах *.csv сохраняются данные котировок
2. Сохраняется файл charts_<дата>.png с графиком котировок
