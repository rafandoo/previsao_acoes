from typing import Any, Tuple, List, Union

import numpy as np
import pandas as pd
from numpy import ndarray


class Utils:

    def __init__(self) -> None:
        pass

    def getDfTicker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Return the rows of a dataframe with the given ticker.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            ticker (str): Ticker to filter by.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        return df[df['ticker'] == ticker]

    def createWindow(self, df: pd.DataFrame, size: int, ahead: int) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        This function creates temporal windows from a DataFrame (df). A time window consists of a sequence
        continuous collection of observations in the DataFrame. size specifies the size of the window, and ahead indicates how many observations to
        front must be anticipated. The function returns two lists, one containing the input windows (x) and the other containing
        the expected outputs (y).

        Parameters:
            df (pd.DataFrame): DataFrame to create the window from.
            size (int): Size of the window.
            ahead (int): Number of steps to shift the window.

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame]]: Tuple of lists of windows, one for input and one for output.
        """
        x, y = [], []

        for i in range(len(df) - size - ahead):
            x.append(df.iloc[i:i + size])
            y.append(df.iloc[i + size + ahead - 1])

        return x, y


    def normCols(self, dfX: pd.DataFrame, dfY: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize columns of a dataframe.

        Parameters:
            dfX (pd.DataFrame): Input Dataframe to normalize.
            dfY (pd.DataFrame): Output Dataframe to normalize.
            cols (List[str]): Columns to normalize.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Normalized dataframes.
        """
        normX = dfX[cols] / dfX[cols].iloc[0] - 1
        normY = dfY.to_frame().T[cols] / dfX[cols].iloc[0] - 1

        return normX, normY

        
    def denormCols(self, dfX: pd.DataFrame, dfNormX: pd.DataFrame, dfNormY: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs the opposite operation to normalization, that is, it denormalizes the normalized DataFrames (dfNormX and dfNormY).

        Parameters:
            dfX (pd.DataFrame): DataFrame with information on the share price at the time the time windows were created.
            dfNormX (pd.DataFrame): DataFrame with normalized input information.
            dfNormY (pd.DataFrame): DataFrame with normalized output information.
            cols (List[str]): Columns to be denormalized.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Returns the denormalized DataFrames.
        """
        dfDenormX = dfX[cols].iloc[0] * (dfNormX[cols] + 1)
        dfDenormY = dfX[cols].iloc[0] * (dfNormY[cols] + 1)

        return dfDenormX, dfDenormY

    def denormPreds(self, xWindows, normPreds, col):
        """
        This function transforms a model's normalized predictions back to the original scale
        of the data, using the information contained in the time windows used to make the predictions.

        Parameters:
            xWindows (List[pd.DataFrame]): List of time windows used to make the predictions.
            normPreds (ndarray): Normalized predictions.
            col (List[str]): Column to be denormalized.

        Returns:
            ndarray: Denormalized predictions.
        """
        preds = np.zeros_like(normPreds)
        for i, window in enumerate(xWindows):
            preds[i] = window[col].iloc[0] * (normPreds[i] + 1)
        return preds

    def createDataset(self, df: pd.DataFrame, windowSize: int, ahead: int, xCols: List[str], yCol: List[str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], ndarray, ndarray]:
        """
        Uses the previous functions to create datasets for training.
        Creates time windows, normalizes input and output columns, and returns numpy arrays
        ready for training. xCols is the list of input columns and yCol is the output column.
        output to be predicted.

        Parameters:
            df (pd.DataFrame): Input dataframe.
            windowSize (int): Size of the window.
            ahead (int): Number of steps to shift the window.
            xCols (List[str]): Columns to use as input data.
            yCol (List[str]): Column to use as output data.

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame], ndarray, ndarray]: Tuple of lists of windows, one for input and one for output, and the normalized numpy arrays for input and output data.
        """
        xWindows, yWindows = self.createWindow(df, windowSize, ahead)

        xCols = xCols if isinstance(xCols, list) else [xCols]
        xLst, yLst = [], []

        for xWindow, yWindow in zip(xWindows, yWindows):

            xNorm, yNorm = self.normCols(xWindow, yWindow, xCols)

            xLst.append(xNorm.to_numpy())
            yLst.append(yNorm[yCol].to_numpy().astype('float').squeeze())

        return xWindows, yWindows, np.stack(xLst), np.stack(yLst)


    def splitTrainTest(self, items, percent=0.8) -> Tuple[List[str], List[str]]:
        """
        Splits the datasets into training and testing sets. percent determines the percentage of
        data to be used for training, and the rest is used for testing. Returns a tuple containing
        the training and testing sets.

        Paramters:
            items (List[str]): List of items to be split.
            percent (float, optional): Percentage of data to be used for training. Defaults to 0.8.

        Returns:
            Tuple[List[str], List[str]]: Training and testing sets.
        """    
        results = []
        for item in items:
            split = int(len(item) * percent)
            results.append(item[:split])
            results.append(item[split:])

        return tuple(results)