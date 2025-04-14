import pandas as pd
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# globals.py
globdevice = torch.device("cpu")
userName = "Phil"
globalDebugCount = 0
tempOutputFile = f'C:\\Users\\{userName}\\Documents\\DataSphere AI\\DataSphere AI Code\\Inputs And Outputs\\tempOutput.txt'
filepathRawData = f'C:\\Users\{userName}\Documents\StockPredictor Project\DataFiles\RawDataFeb24-2\\'
filepathRawDataLive = f'C:\\Users\Phil\Documents\StockPredictor Project\DataFiles\RawDataLive\\'
dongo=3
EndMarker = "===================================================================================================="
FakeDocumentBasePath = r"C:/Users/Phil/Documents/DataSphere AI/DataSphere AI Not Code/Inputs And Outputs/Outputs/FakeDocuments/"
StateStorageDirectory = r"C:\Users\Phil\Documents\DataSphere AI\DataSphere AI Code v2\StateStorage"
LargeBlockAnalysisCompletedFile = os.path.join(StateStorageDirectory, "LargeBlockAnalysisCompleted.json")
IterativeAnalysisCompletedFile = os.path.join(StateStorageDirectory, "IterativeAnalysisCompleted.json")