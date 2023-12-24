from src.tickers_history.main import TickersHistory

def test_normalize():
    th = TickersHistory('tickers.txt', '01.01.20')
    th.calc_accuracy = None
    assert th.normalize(100, 100) == 100
