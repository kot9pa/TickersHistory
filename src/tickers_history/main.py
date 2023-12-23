'''Main.py'''
import argparse
import os
import queue
import threading
import concurrent.futures
from functools import lru_cache
from datetime import datetime, timezone
from time import perf_counter
from dataclasses import dataclass, field
import requests
import pandas as pd
import matplotlib.pyplot as plt

@dataclass(order=True)
class Ticker:
    '''Ticker dataclass'''
    symbol: str = field(compare=False)
    timestamp: float
    adjclose: float = field(compare=False)
    normalize: float = field(default=0.0, init=False, compare=False)

class TickersHistory:
    '''TickersHistory class'''
    # точность нормализации (чем меньше, тем лучше работает кэширование)
    calc_accuracy = None
    tickers_path = 'charts'
    tickers_db = pd.DataFrame(columns=Ticker.__annotations__)

    def __init__(self, filename: str, start_date: str):
        self.tickers = tuple(self.get_ticker(filename))
        self.start_date = datetime.strptime(start_date, '%d.%m.%y')
        self.end_date = datetime.strptime(str(datetime.now().date()), '%Y-%m-%d')
        self.__barrier = threading.Barrier(len(self.tickers))
        self.__semaphore = threading.Semaphore(1)
        self.__file_queue = queue.Queue()

    def get_data(self):
        """Функция для формирования данных котировок"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for ticker in self.tickers:
                future = executor.submit(self.get_history_data, ticker, self.start_date, self.end_date)
                future.add_done_callback(self.__parsing_data)
        # сохраняем котировки на диск
        if not self.__file_queue.empty():
            threading.Thread(target=self.__save_history, daemon=True).start()
            self.__file_queue.join()
            return True
        return False

    def process_data(self, ticker):
        """Функция-обработчик для нормализации цены."""
        with self.__semaphore:
            try:
                df = pd.read_csv(os.path.join(self.tickers_path, f'{ticker}.csv'))
                start_rounded = round(df.at[0, 'adjclose'], self.calc_accuracy)
                for i, row in df.iterrows():
                    adjclose_rounded = round(row['adjclose'], self.calc_accuracy)
                    df.at[i, 'normalize'] = self.normalize(adjclose_rounded, start_rounded)
                self.tickers_db = pd.merge(self.tickers_db, df, how='outer')
                self.__file_queue.put((df, ticker))
                print(f"{threading.current_thread()} normalize.. OK")
            except Exception:
                print(f"{threading.current_thread()} normalize.. False")
                raise

        self.__barrier.wait()
        #print(self.normalize.cache_info())
        # сохраняем котировки на диск
        threading.Thread(target=self.__save_history, daemon=True).start()
        self.__file_queue.join()

    def plot_data(self):
        """Функция для построения графиков котировок"""
        self.tickers_db['timestamp'] = pd.to_datetime(self.tickers_db['timestamp'], unit='s')
        df = self.tickers_db.groupby('symbol', sort=True)
        plt.figure(figsize = [20, 8]) # type: ignore
        ax = plt.gca()
        df.plot(x='timestamp', y='normalize', ax=ax)
        plt.ylabel("Adjclose (normalize)")
        plt.legend(sorted(self.tickers), loc='upper left')
        #plt.show()
        plt.savefig(f'charts_{self.start_date.date()}.png')
        print(f"{threading.current_thread()} savefig to 'charts_{self.start_date.date()}.png'")

    def start(self):
        '''Start method'''
        # получаем котировки с помощью API сервиса yahoo.finance
        if self.get_data():
            # нормализуем цену (adjclose)
            threads = [threading.Thread(target=self.process_data, args=(ticker,), daemon=True)
                    for ticker in self.tickers]
            [thread.start() for thread in threads]
            try:
                [thread.join() for thread in threads]
                # создаем общий график всех котировок
                self.plot_data()
            except RuntimeError as err:
                print(err)
                raise

    @lru_cache(maxsize=1024)
    def normalize(self, adjclose: float, start: float = 100):
        '''Normalize adjclose'''
        return round(adjclose * 100 / start, self.calc_accuracy)

    @staticmethod
    def get_ticker(file: str):
        '''Get list tickers'''
        with open(file, encoding='utf-8') as wrapper:
            for line in wrapper:
                ticker = line.strip()
                yield ticker

    @staticmethod
    def get_history_data(ticker: str, start_date: datetime, end_date: datetime, interval: str = "1wk"):
        """Получает исторические данные для указанного тикера актива.
        :param ticker: str, тикер актива.
        :param start_date: str, дата начала периода в формате 'дд.мм.гг'.
        :param end_date: str, дата окончания периода в формате 'дд.мм.гг'.
        :param interval: str, интервал времени (неделя, день и т.д.) (необязательный, по умолчанию '1wk' - одна неделя).
        :return: str, JSON-строка с историческими данными.
        """
        try:
            per1 = int(start_date.replace(tzinfo=timezone.utc).timestamp())
            per2 = int(end_date.replace(tzinfo=timezone.utc).timestamp())
            params = {"period1": str(per1), "period2": str(per2),
                    "interval": interval, "includeAdjustedClose": "true"}
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            user_agent_key = "User-Agent"
            user_agent_value = "Mozilla/5.0"
            headers = {user_agent_key: user_agent_value}
            response = requests.get(url, headers=headers, params=params)
        except Exception:
            print(f'{threading.current_thread()} get {ticker=}.. False')
            raise
        if response.ok:
            print(f'{threading.current_thread()} get {ticker=}.. OK')
            return response.json()
        else:
            raise ValueError(f'{response.reason=} {response.status_code=}')

    def __parsing_data(self, future):
        """Функция-обработчик полученных котировок."""
        exception = future.exception()
        if exception is not None:
            print(f'An error occurred: {exception}')
            return
        else:
            try:
                result = future.result()
                symbol = result['chart']['result'][0]['meta']['symbol']
                timestamp = result['chart']['result'][0]['timestamp']
                adjclose = result['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
                data = pd.DataFrame(Ticker(symbol, t, a) for t, a in zip(timestamp, adjclose) if a)
                self.__file_queue.put((data, symbol))
                print(f'{threading.current_thread()} parsing.. OK')
            except Exception:
                print(f'{threading.current_thread()} parsing.. False')
                raise

    def __save_history(self):
        """Функция-поток для записи файлов из очереди на диск."""
        while True:
            data, filename = self.__file_queue.get()
            os.makedirs(self.tickers_path, exist_ok=True)
            with open(os.path.join(self.tickers_path, f'{filename}.csv'), 'w', newline='', encoding='utf-8') as f:
                data.to_csv(f, index=False)
            #print(f'{filename} was saved successfully')
            self.__file_queue.task_done()

def execute_main():
    '''Main method'''
    start = perf_counter()
    # создаём парсер аргументов и передаём их
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tickers", default='tickers.txt', required=False, help="path to file, default 'tickers.txt'")
    ap.add_argument("-s", "--start_date", default='01.01.20', required=False, help="start_date in format 'dd.mm.yy', default 01.01.20")
    args = vars(ap.parse_args())

    th = TickersHistory(args['tickers'], args['start_date'])
    print('Tickers:', *th.tickers)
    print(f'Start date: {th.start_date.date()}')
    print(f'End date: {th.end_date.date()}')
    if th.start_date <= th.end_date:
        print('\nRun main():')
        th.start()
        print(f'All tasks complete in {perf_counter() - start:.4f}s\n')

if __name__ == '__main__':
    execute_main()
