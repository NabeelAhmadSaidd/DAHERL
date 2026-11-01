#File: preprocessing/feature_engineering.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def volatility(close: pd.Series, window: int = 20) -> pd.Series:
        returns = close.pct_change()
        return returns.rolling(window).std()

    @staticmethod
    def bollinger(close: pd.Series, window: int = 20):
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return upper, lower

    @staticmethod
    def vortex(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vortex Indicator (VI) – trend strength and direction."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True range
        tr = FeatureEngineer.atr(df, period=1)  # single‑period TR
        # VM+ and VM-
        vm_plus = (high - low.shift()).abs()
        vm_minus = (low - high.shift()).abs()

        sum_vm_plus = vm_plus.rolling(period).sum()
        sum_vm_minus = vm_minus.rolling(period).sum()
        sum_tr = tr.rolling(period).sum()

        vi_plus = sum_vm_plus / (sum_tr + 1e-10)
        vi_minus = sum_vm_minus / (sum_tr + 1e-10)

        # Vortex = max(VI+, VI-)
        vortex = pd.concat([vi_plus, vi_minus], axis=1).max(axis=1)
        return vortex

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD line and signal line."""
        ema_fast = FeatureEngineer.ema(close, fast)
        ema_slow = FeatureEngineer.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX) – measures trend strength.
        Values > 25 indicate a strong trend; < 20 indicates a weak trend.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 1. True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 2. Directional Movements (+DM and -DM)
        up_move = high - high.shift()
        down_move = low.shift() - low

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # 3. Smooth using rolling mean (Wilder's smoothing is more complex,
        #    but rolling mean is standard in many practical implementations)
        smoothed_tr = tr.rolling(period).mean()
        smoothed_pos = pd.Series(pos_dm).rolling(period).mean()
        smoothed_neg = pd.Series(neg_dm).rolling(period).mean()

        # 4. Directional Indicators (+DI and -DI)
        plus_di = 100 * (smoothed_pos / (smoothed_tr + 1e-10))
        minus_di = 100 * (smoothed_neg / (smoothed_tr + 1e-10))

        # 5. Directional Index (DX)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))

        # 6. ADX = smoothed DX
        adx = dx.rolling(period).mean()
        return adx

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]

        # Trend
        df["ema_20"] = self.ema(close, 20)
        df["ema_50"] = self.ema(close, 50)
        macd_line, signal_line = self.macd(close)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line

        # Momentum
        df["rsi_14"] = self.rsi(close)
        df["rsi_5"] = self.rsi(close, period=5)  
        df["vortex_14"] = self.vortex(df, 14)

        # Volatility
        df["atr_14"] = self.atr(df)
        df["volatility_20"] = self.volatility(close)

        # Mean reversion
        bb_upper, bb_lower = self.bollinger(close)
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower

        # Returns
        df["returns"] = close.pct_change(1)
        df["return_5"] = close.pct_change(5)

        df["adx"] = self.adx(df, period=14)

        # Log volume (handle zero volume by adding 1e-9)
        if "volume" in df.columns:
            df["log_volume"] = np.log(df["volume"] + 1e-9)
        else:
            df["log_volume"] = np.nan

        # Drop rows with NaN (first 50+ rows due to rolling windows)
        df = df.dropna().reset_index(drop=True)
        return df