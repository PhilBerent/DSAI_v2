import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import scipy.stats
from scipy.stats import chi2
import datetime
import bisect
import random
import clr
import time
from collections import defaultdict
import json
import globals as g
from globals import *
import gc

clr.AddReference('Python.Runtime')
clr.AddReference(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\System.dll")
from System import Array, Int32, Double

 
def cleanText(text: str) -> str:
    """
    Cleans a string by replacing common non-ASCII punctuation and
    potentially problematic characters with their closest ASCII equivalents.
    Also handles potential Mojibake patterns if they exist.
    """
    if not isinstance(text, str):
        text = str(text) # Ensure input is a string

    replacements = {
        # --- Stage 1: Fix potential Mojibake first ---
        # (If text might already be corrupted, e.g., UTF-8 read as Windows-1252)
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€“': '-',
        'â€”': '--', # Use double dash for em dash
        'â€¦': '...',
        'â€˜': "'",
        'â€': '"', # Common Mojibake for quote/apostrophe
        'Ã©': 'e', # Example Mojibake for é

        # --- Stage 2: Convert correct Unicode to simple ASCII ---
        '\u2018': "'",  # Left single quote -> ASCII apostrophe
        '\u2019': "'",  # Right single quote / Unicode Apostrophe -> ASCII apostrophe
        '\u201c': '"',  # Left double quote -> ASCII double quote
        '\u201d': '"',  # Right double quote -> ASCII double quote
        '\u2013': '-',  # En dash -> ASCII hyphen
        '\u2014': '--', # Em dash -> ASCII double hyphen
        '\u2026': '...', # Ellipsis -> ASCII triple dot
        '\u00a0': ' ',  # Non-breaking space -> ASCII space
        # Add more direct Unicode to ASCII mappings as needed
        '\u00e9': 'e',  # é -> e
        '\u00e8': 'e',  # è -> e
        '\u00ea': 'e',  # ê -> e
        # ... etc. for other common accented characters if desired
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Optional: Final check for any remaining non-ASCII after replacement
    # This is generally not needed if replacements are comprehensive and
    # the final write uses errors='ignore' or 'replace', but can be useful
    # for debugging the cleaning process itself.
    # text = text.encode('ascii', 'ignore').decode('ascii')

    return text.strip()

def find_modes(distances, bandwidth=None):
    # Flatten the distance matrix 
    flattened_distances = distances.flatten()

    # Compute the kernel density estimate
    if bandwidth is None:
        kde = gaussian_kde(flattened_distances)
    else:
        kde = gaussian_kde(flattened_distances, bw_method=bandwidth)

    # Define a range of distance values for evaluating the KDE
    x_vals = np.linspace(np.min(flattened_distances), np.max(flattened_distances), num=1000)

    # Evaluate the KDE at the specified distance values
    kde_vals = kde(x_vals)

    # Find local maxima in the KDE
    local_maxima_indices = argrelextrema(kde_vals, np.greater)
    local_maxima = x_vals[local_maxima_indices]

    return local_maxima

# functions that reads a matrix from a data file
def readMatrixFromFile(filename, filePath = 'testInputPath'):
    if filePath == 'testInputPath':
        filePath = gb.testFileInputPath
    filename = filePath + filename
    with open(filename, 'r') as f:
        lines = f.readlines()
        matrix = []
        for line in lines:
            matrix.append([float(x) for x in line.split()])
        matrix = np.array(matrix)
    return matrix


def write_tmat_to_file(tensor):
    # Convert tensor to NumPy array
    global tempOutputFile
    tensor_np = tensor.detach().numpy()

    # Open file for writing
    with open(tempOutputFile, 'w') as f:
        print("File opened successfully")

        if tensor.dim() == 1:
            # Write 1D vector elements separated by tabs
            for elem in tensor_np:
                f.write(str(float(elem)) + '\t')
            # End with a newline character
            f.write('\n')

        elif tensor.dim() == 2:
            # Loop over rows in tensor
            for row in tensor_np:
                # Write each element in row to file separated by tabs
                for elem in row:
                    f.write(str(float(elem)) + '\t')  # Convert tensor element to float before writing
                # End row with a newline character
                f.write('\n')

        else:
            raise ValueError("The input tensor should have either 1 or 2 dimensions.")

    # Close file
    f.close()

# function to write a list to a file with each element on a separate line
def WriteListToFile1(listToWrite, filename=tempOutputFile):
    with open(filename, 'w') as f:
        for item in listToWrite:
            f.write("%s\n" % item)

def WriteListToFile(listToWrite, filename=tempOutputFile):
    with open(filename, 'w') as f:
        if isinstance(listToWrite, np.ndarray):  # Check if input is a numpy array
            listToWrite = listToWrite.tolist()  # Convert numpy array to list
            
        if isinstance(listToWrite[0], list):  # Check if input is a 2D array
            for row in listToWrite:
                row_str = '\t'.join(map(str, row))
                f.write("%s\n" % row_str)
        else:  # Input is a 1D array
            for item in listToWrite:
                f.write("%s\n" % item)
            
def WriteNpArToFile(arToWrite, filename=g.tempOutputFile):
    with open(filename, "w") as f:
        if arToWrite.ndim == 1:
            np.savetxt(f, arToWrite)
        else:
            np.savetxt(f, arToWrite, fmt="%d", delimiter="\n")

def ObjectType(obj):
    if isinstance(obj, list):
        return "list"
    elif isinstance(obj, tuple):
        return "tuple"
    elif isinstance(obj, np.ndarray):
        return "NumPy array"
    elif isinstance(obj, torch.Tensor):
        return "PyTorch tensor"
    elif isinstance(obj, pd.DataFrame):
        return "Pandas DataFrame"
    elif isinstance(obj, Array[Int32]):
        return "Int32 Aray"
    elif isinstance(obj, Array[Double]):
        return "Double Aray"
    elif isinstance(obj, str):
        return "String"
    else:
        return type(obj).__name__
        
def GetDims(obj):
    if isinstance(obj, list):
        dims = GetDimsOfList(obj)
        return "(" + ", ".join(str(dim) for dim in dims) + ")"
    if isinstance(obj, tuple) or isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        dims = obj.shape if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor) else tuple(len(subobj) for subobj in obj)
        return "(" + ", ".join(str(dim) for dim in dims) + ")"
    elif isinstance(obj, pd.DataFrame):
        return "(" + str(len(obj.index)) + ", " + str(len(obj.columns)) + ")"
    elif isinstance(obj, Array[Int32]) or isinstance(obj, Array[Double]):
        return "(" + str(obj.GetLength(0)) + ")"
    else:
        return None

def skewness(x):
    res = ((x - x.mean())**3).mean() / (x.std()**3)
    return res

def kurtosis(x):
    res = ((x - x.mean())**4).mean() / (x.std()**4)
    return res - 3

def Normality_Score(x):
    return abs(kurtosis(x))

def GetDimsOfList(thisList):
    currList = thisList
    numdims = 0
    # create a list to which elements are added
    dimList=[]
    if isinstance(currList, list):
        thisDim = len(currList)
        dimList.append(thisDim)
        numdims += 1
    while isinstance(currList[0], list):
        thisDim = len(currList)
        dimList.append(thisDim)
        numdims += 1
        currList = currList[0]
    return dimList        

def GetTypeAndDims(obj):
    type_str = ObjectType(obj)
    dims_str = GetDims(obj)
    if type_str is not None and dims_str is not None:
        return type_str + ": " + dims_str
    else:
        return None
# def WriteTensorArray(obj):

import numpy as np
import torch
import pandas as pd

import numpy as np
import torch
import pandas as pd
from fileinput import filename

def WriteTensorListToFile(obj, filename = g.tempOutputFile):
    with open(filename, "w") as f:
        for elem in obj:
            f.write(str(elem.item()) + "\n")
    

def WriteToFile(obj, filename = g.tempOutputFile):
    type_str = ObjectType(obj)
    if type_str is None:
        return
    if isinstance(obj, np.ndarray) and obj.ndim == 1:
        with open(filename, "w") as f:
            for elem in obj:
                f.write(str(elem) + "\n")
    elif isinstance(obj, np.ndarray) and obj.ndim == 2:
        with open(filename, "w") as f:
            for row in obj:
                f.write("\t".join(str(elem) for elem in row) + "\n")
    elif isinstance(obj, np.ndarray) and obj.ndim == 3:
        with open(filename, "w") as f:
            for i in range(obj.shape[0]):
                if i > 0:
                    f.write("\n\n")
                for row in obj[i]:
                    f.write("\t".join(str(elem) for elem in row) + "\n")
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return
        # Check if the elements are single-value tensors
        if isinstance(obj[0], torch.Tensor) and obj[0].numel() == 1:
            with open(filename, "w") as f:
                for elem in obj:
                    f.write(str(elem.item()) + "\n")
        elif isinstance(obj[0], (list, tuple)):
            with open(filename, "w") as f:
                for row in obj:
                    f.write("\t".join(str(elem) for elem in row) + "\n")
        else:
            with open(filename, "w") as f:
                for elem in obj:
                    f.write(str(elem) + "\n")
    elif isinstance(obj, torch.Tensor) and obj.ndim == 1:
        with open(filename, "w") as f:
            # Convert the tensor to a string with each element followed by a newline
            tensor_string = '\n'.join(str(elem.item()) for elem in obj)
            f.write(tensor_string)
    elif isinstance(obj, torch.Tensor) and obj.ndim == 2:
        with open(filename, "w") as f:
            # Convert each row of the tensor to a tab-separated string and join them with newlines
            tensor_string = '\n'.join('\t'.join(str(elem.item()) for elem in row) for row in obj)
            f.write(tensor_string)    
    elif isinstance(obj, torch.Tensor) and obj.ndim == 3:
        with open(filename, "w") as f:
            B, T, V = obj.shape
            for i in range(B):
                if i > 0:
                    f.write("\n\n")
                # Convert each sub-tensor to string format
                tensor_string = '\n'.join('\t'.join(str(value.item()) for value in row) for row in obj[i])
                f.write(tensor_string)
    elif isinstance(obj, pd.DataFrame):
        obj.to_csv(filename, sep="\t", index=False)
    elif isinstance(obj, Array):
        rank = obj.Rank
        with open(filename, "w") as f:
            if rank == 1:
                p_list = list(obj)
                for elem in p_list:
                    f.write(str(elem) + "\n")
            elif rank == 2:
                for i in range(obj.GetLength(0)):  # Iterate over rows
                    for j in range(obj.GetLength(1)):  # Iterate over columns
                        f.write(str(obj[i, j]))
                        if j < obj.GetLength(1) - 1:  # Avoid adding space after the last element
                            f.write(" ")
                    f.write("\n")
            elif rank == 3:
                for k in range(obj.GetLength(0)):  # Iterate over depth (the matrices)
                    for i in range(obj.GetLength(1)):  # Iterate over rows
                        for j in range(obj.GetLength(2)):  # Iterate over columns
                            f.write(str(obj[k, i, j]))
                            if j < obj.GetLength(2) - 1:  # Avoid adding space after the last element
                                f.write(" ")
                        f.write("\n")
                    if k < obj.GetLength(0) - 1:  # Avoid adding an extra newline after the last matrix
                        f.write("\n")
    elif isinstance(obj, Array[Int32]) or isinstance(obj, Array[Double]):
        p_list = list(obj)
        with open(filename, "w") as f:
            for elem in p_list:
                f.write(str(elem) + "\n")
    elif isinstance(obj, str):
            with open(filename, "w") as f:
                f.write(obj)
    # else if the object is a callable_iterator, convert it to a list and write to file
    elif hasattr(obj, '__iter__') and not isinstance(obj, str):
        with open(filename, "w") as f:
            for elem in list(obj):
                f.write(str(elem) + "\n")
    else:
        return

# write a function that takes the input of two torch tensors the first of dimensions l x m and the second of dimension p x m and outputs a tensor of dimensions l x p where each entry (i, j) is  the corrlation between the ith row of the first tensor and the jth row of the second tensor
def corr_matrix(tensor1, tensor2, epsilon=1e-8):
    T1, D1 = tensor1.shape
    T2, D2 = tensor2.shape
    assert D1 == D2, "Both sequences should have the same feature dimension."

    seq_i_centered = tensor1 - tensor1.mean(dim=-1, keepdim=True) + epsilon
    seq_j_centered = tensor2 - tensor2.mean(dim=-1, keepdim=True) + epsilon

    seq_i_norm = torch.norm(seq_i_centered, p=2, dim=-1, keepdim=True)
    seq_j_norm = torch.norm(seq_j_centered, p=2, dim=-1, keepdim=True)

    corr_mat = torch.matmul(seq_i_centered, seq_j_centered.transpose(-1, -2)) / (seq_i_norm * seq_j_norm.transpose(-1, -2))
    corr_mat = corr_mat.clamp(min=-1, max=1)
    return corr_mat

# function to unflatten 2D array
def UnFlatten2DArray(InputArray, T1, T2):
    outputMat = torch.zeros((T1, T2))
    for i in range(T1):
        for j in range(T2):
            arrayIndex = i * T2 + j
            outputMat[i, j] = InputArray[arrayIndex]
    return outputMat


def UnFlatten3DMatrix(InputArray, T1, T2, T3):
    outputMat = torch.zeros((T1, T2, T3))
    for i in range(T1):
        for j in range(T2):
            for k in range(T3):
                arrayIndex = i * T2*T3 + j*T3 + k
                outputMat[i, j, k] = InputArray[arrayIndex]
    return outputMat

# Function to compare two 3d tensors return true if they are equal and false otherwise
def Compare3DTensors(tensor1, tensor2):
    T1, T2, T3 = tensor1.shape
    for i in range(T1):
        for j in range(T2):
            for k in range(T3):
                if tensor1[i, j, k] != tensor2[i, j, k]:
                    return False
    return True

def softmin(a, b, c, gamma=1.0):
            ta= torch.tensor(a)
            tb= torch.tensor(b)
            tc= torch.tensor(c)
            maxV = torch.max(torch.max(ta, tb), tc)
            amod= torch.exp(-(maxV - ta)/gamma) 
            bmod= torch.exp(-(maxV - tb)/gamma)
            cmod= torch.exp(-(maxV - tc)/gamma)
            retval=-gamma*(maxV + torch.log(amod+bmod+cmod))
            # return  - self.gamma * torch.log(torch.exp(-a/self.gamma) 
            #                                         + torch.exp(-b/self.gamma) 
            #                                         + torch.exp(-c/self.gamma))
            return retval

def softmin1(a, b, c, gamma):
    ta= torch.tensor(a)
    tb= torch.tensor(b)
    tc= torch.tensor(c)
    max_val = torch.max(torch.max(ta, tb), tc)
    return - max_val + gamma * (torch.log(
        torch.exp(-(ta - max_val)/gamma)
        + torch.exp(-(tb - max_val)/gamma)
        + torch.exp(-(tc - max_val)/gamma)
    ))

def softmin2(a, b, c, gamma):
    ta= torch.tensor(a)
    tb= torch.tensor(b)
    tc= torch.tensor(c)
    max_val = torch.max(torch.max(ta, tb), tc)
    ta_minMax = ta - max_val
    tb_minMax = tb - max_val
    tc_minMax = tc - max_val
    ta_minMax_exp = torch.exp(-ta_minMax/gamma)
    tb_minMax_exp = torch.exp(-tb_minMax/gamma)
    tc_minMax_exp = torch.exp(-tc_minMax/gamma)
    sum_exp = ta_minMax_exp + tb_minMax_exp + tc_minMax_exp
    log_sum_exp = torch.log(sum_exp)
    gamma_log_sum_exp = -gamma * log_sum_exp
    retVal = max_val + gamma_log_sum_exp
    return retVal

def softmin3(a, b, c, gamma):
    ta= torch.tensor(a)
    tb= torch.tensor(b)
    tc= torch.tensor(c)
    max_val = torch.max(torch.max(ta, tb), tc)
    ta_ovMax = ta/max_val
    tb_ovMax = tb/max_val
    tc_ovMax = tc/max_val
    ta_ovMax_exp = torch.exp(-ta_ovMax/gamma)
    tb_ovMax_exp = torch.exp(-tb_ovMax/gamma)
    tc_ovMax_exp = torch.exp(-tc_ovMax/gamma)
    sum_exp = ta_ovMax_exp + tb_ovMax_exp + tc_ovMax_exp
    log_sum_exp = torch.log(sum_exp)
    gamma_log_sum_exp = -gamma * log_sum_exp
    retVal = max_val * gamma_log_sum_exp
    return retVal

# function UnFlatten3dTensor which converts a 1d array to a 3d tensor with dimensions T1 x T2 x T3 where the index of the output Tensor[i][j][k] is i*T2*T3 + j*T3 + k. Input array will have length T1xT2xT3  and Function returns a tensor with dimensions T1 x T2 x T3
def UnFlatten3dTensor(InputArray, T1, T2, T3):
    outputTensor = torch.zeros((T1, T2, T3))
    for i in range(T1):
        for j in range(T2):
            for k in range(T3):
                arrayIndex = i * T2*T3 + j*T3 + k
                outputTensor[i, j, k] = InputArray[arrayIndex]
    return outputTensor

def GetTensorFromCSharpArrayX(InputArray):
    outputTensor= torch.zeros(len(InputArray))
    for i in range(len(InputArray)):
        outputTensor[i] = InputArray[i]
    return outputTensor

def GetTensorFromCSharpAr(InputArray):
    # Convert the C# double[] array to a Python list
    python_list = list(InputArray)
    
    # Convert the Python list to a torch tensor
    tensor = torch.tensor(python_list, dtype=torch.float64)
    return tensor

def GetNumpyArFromCSharpAr(InputArray):
    # Convert the C# double[] array to a Python list
    python_list = list(InputArray)
    
    # Convert the Python list to a NumPy array
    array = np.array(python_list, dtype=np.float64)
    return array

def Write3DMatToFile(mat3d, blankLinesBetweenMatrixes=1, filename = g.tempOutputFile):
    if mat3d.is_cuda:
        mat3dNp = mat3d.detach().cpu().numpy()
    else:
        mat3dNp = mat3d.detach().numpy()

    # convert batchred to a string array where each block of size block_length x encoding_dim is written
    # as block_length strings of length encoding_dim and there are blankLinesBetweenMatrixes blank lines between each block
    # this is the format that the C# code expects
    mat3dStr = []
    numMats = mat3dNp.shape[0]
    for i in range(numMats):
        for j in range(mat3dNp.shape[1]):
            linestring = ''
            for k in range(mat3dNp.shape[2]-1):
                linestring += str(mat3dNp[i,j,k]) + '\t'
            linestring += str(mat3dNp[i,j,mat3dNp.shape[2]-1])
            mat3dStr.append(linestring)
        if i < numMats-1:
            for k in range(blankLinesBetweenMatrixes):
                mat3dStr.append('')
            
    WriteToFile(mat3dStr, filename)

# Function to read a 3D Mat from a text file where each row of a 2d matrix is on one line, each element of each row is
# seperated by ' ' characters, and there are 1 or more blank lines between each 2d matrix
def Read3DMatFromFile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize a 3D list
    threeD_list = []
    twoD_list = []

    for line in lines:
        if line.strip() == '':
            if twoD_list:
                # Add the 2D list to the 3D list and initialize a new 2D list
                threeD_list.append(np.array(twoD_list))
                twoD_list = []
        else:
            # Add the row to the 2D list
            twoD_list.append(list(map(float, line.split())))

    # Add the last 2D list if not empty
    if twoD_list:
        threeD_list.append(np.array(twoD_list))

    # Convert the 3D list to a PyTorch tensor
    threeD_tensor = torch.stack([torch.from_numpy(a) for a in threeD_list])
    threeD_tensor=threeD_tensor.float()
    threeD_tensor=threeD_tensor.to(g.globdevice)
    
    return threeD_tensor
    
# Function to test if 2 3D tensors are equal
def Are3DTensorsEqual(t1, t2, tol=1e-6):
    if t1.shape != t2.shape:
        return False
    else:
        for i in range(t1.shape[0]):
            for j in range(t1.shape[1]):
                for k in range(t1.shape[2]):
                    if abs(t1[i,j,k] - t2[i,j,k]) > tol:
                        return False
        return True

def Are1DTensorsEqual(t1, t2, tol=1e-6):
    if t1.shape != t2.shape:
        return False
    else:
        for i in range(t1.shape[0]):
            if t1[i] != t2[i]:
                return False
        return True

def NpArraysAreEqual(Array1, Array2, tolerance=0.01):
    return np.allclose(Array1, Array2, atol=tolerance)

def ListsAreEqual(List1, List2, tolerance=0.01):
    # convert lists to numpy arrays
    Array1 = np.array(List1)
    Array2 = np.array(List2)
    return np.allclose(Array1, Array2, atol=tolerance)

# function to check if tensors are equal
def TensorsAreEqual(t1, t2, tol=1e-6):
    if t1.shape != t2.shape:
        return False
    return torch.all(torch.abs(t1 - t2) <= tol)

# function to check if bool tensors are equal
def BoolTensorsAreEqual(t1, t2):
    if t1.shape != t2.shape:
        return False
    return torch.all(t1 == t2)
    
# function to write 3d Mat to file as a 1D array
def Write3DMatToFile1D(mat3d, filename):
    mat3dNp = mat3d.detach().numpy()
    batchRedFlat = mat3dNp.astype(np.float64).flatten()
    batchRedFlatList = batchRedFlat.tolist()
    WriteToFile(batchRedFlatList, filename)

# function to read a 1D array from file
def ReadFromFile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def Read2DMatFromFile(filename):
    # Open the file for reading
    with open(filename, 'r') as file:
        # Read lines from the file
        lines = file.readlines()
        # Split each line into a list of floats
        data = [[float(value) for value in line.split()] for line in lines]
        # Convert the list of lists into a PyTorch FloatTensor
        tensor = torch.tensor(data, dtype=torch.float32)
        tensor=tensor.to(g.globdevice)
    
    return tensor

def Read2DMatFromFileSep(fileName, separator="\t"):
    # Initialize an empty list to hold the 2D array
    mat2D = []
    
    # Open the file for reading
    with open(fileName, 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip leading/trailing whitespace and split the line by the separator
            row = line.strip().split(separator)
            # Append the row to the 2D array
            
            mat2D.append(row)
    
    return mat2D

def WriteDictToFile(dictionary, filename=g.tempOutputFile, ensureAscii=False):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(dictionary, file, ensure_ascii=ensureAscii, indent=2)        
        
def ReadDictFromFile(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

# function to read 1D array from file and convert to 3D Mat
def Read3DMatFromFile1D(filename, T1, T2, T3):
    batchRedFlatList = ReadFromFile(filename)
    batchRedFlat = np.array(batchRedFlatList, dtype=np.float64)
    batchRed = batchRedFlat.reshape(T1, T2, T3)
    batchRedTensor = torch.tensor(batchRed, dtype=torch.float64)
    return batchRedTensor

def compute_sd_over_mnAbs(input):
    # Compute the standard deviation across the batch and time dimensions
    sd = torch.std(input, dim=[0, 1])
    # Compute the mean absolute values across the batch and time dimensions
    mnabs = torch.mean(torch.abs(input), dim=[0, 1])
    # Divide sd by mnabs for each variable and then take the average
    avg_sd_over_mnabs = torch.mean(sd / mnabs)
    return avg_sd_over_mnabs 

def normalize_per_featurex(input):
    # Find the minimum and maximum values per feature across all sequences in the batch.
    min_val, _ = input.min(dim=0)
    min_val, _ = min_val.min(dim=0, keepdim=True)  # further min across the second dimension
    max_val, _ = input.max(dim=0)
    max_val, _ = max_val.max(dim=0, keepdim=True)  # further max across the second dimension

    # Normalize the input tensor based on the min and max values
    return (input - min_val) / (max_val - min_val + 1e-8)

def windows_to_unix_path(windows_path):
    # Extract the drive letter and convert it to the relevant WSL path
    drive_letter = windows_path[0].lower()
    wsl_drive_path = f"/mnt/{drive_letter}"

    # Replace backslashes with forward slashes and remove the drive letter
    unix_path = windows_path.replace('\\', '/')[2:]
    # Prepend the WSL drive path
    unix_path = wsl_drive_path + unix_path
    # Escape spaces for Unix
    unix_path = unix_path.replace(' ', '\\ ')

    return unix_path


def blockStartsSelect(dataLength, blockLength, numBlocksToGet, randomSeed=23):
    if numBlocksToGet * blockLength > dataLength:
        raise ValueError("Can't select that many non-overlapping blocks from the provided data length.")

    block_starts = []
    potential_starts = list(range(0, dataLength - blockLength + 1))
    # initialize random
    random.seed(randomSeed)

    for _ in range(numBlocksToGet):
        # Randomly select a start point from the potential_starts list
        chosen_start = random.choice(potential_starts)
        
        # Remove that start point from the potential list
        potential_starts.remove(chosen_start)
        
        # Use bisect to determine where this start point would fit in our sorted list
        i = bisect.bisect_left(block_starts, chosen_start)
        
        # Insert the chosen start into the block_starts list
        bisect.insort_left(block_starts, chosen_start)
        
        # Remove all start points from the potential_starts list that would result in an overlap
        # with the block we just added.
        start_range = max(0, chosen_start - blockLength + 1)
        end_range = min(chosen_start + blockLength, dataLength - blockLength + 1)
        for j in range(start_range, end_range):
            if j in potential_starts:
                potential_starts.remove(j)
        
        # If we run out of potential starts before selecting the desired number of blocks, raise an error.
        if len(potential_starts) == 0 and len(block_starts) < numBlocksToGet:
            raise ValueError("Can't select any more non-overlapping blocks. Try selecting fewer blocks or reducing block length.")
    
    random.shuffle(block_starts)
    return block_starts

# function which takes a tensor and returns the middle elements which make up 'midPct' percent of the tensor
def GetMiddleElements(tensor, midPct):
    numElements = tensor.shape[0]
    numElementsToGet = int(numElements * midPct)
    startElement = int((numElements - numElementsToGet)/2)
    endElement = startElement + numElementsToGet
    return tensor[startElement:endElement]

def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)  # Set seed for numpy
    torch.manual_seed(seed_value)  # Set seed for pytorch
    torch.cuda.manual_seed_all(seed_value)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Use deterministic algorithms for GPU operations
    torch.backends.cudnn.benchmark = False  # Do not use optimized algorithms for GPU operations

# function that takes a tensor with any number of dimensions as input and returns x% of the elements of the tensor which are randomly selected on the first dimension. The function should return a tensor with the same dimensions as the input tensor and the same elements as the input tensor except that x% of the elements are randomly selected and the rest are not added to the output tensor
def RandomlySelectElements(tensor, pctToSelect):
    numElements = tensor.shape[0]
    numElementsToSelect = int(numElements * pctToSelect)
    # select a sample of the elements without replacement
    selectedIndices = np.random.choice(numElements, numElementsToSelect, replace=False)
    # sort the indices
    selectedIndices.sort()
    # select the elements of the tensor with the selected indices
    selectedElements = tensor[selectedIndices]
    return selectedElements

def compute_l2_penalty(model):
    l2_penalty = 0.0
    total_params = 0
    for param in model.parameters():
        l2_penalty += torch.norm(param, 2) ** 2
        total_params += param.numel()
    return l2_penalty, total_params

def evaluate_l2_for_lambdas(model):
    _, total_params = compute_l2_penalty(model)
    lambdas = [0.001, 0.05] + list(np.logspace(np.log10(0.05), np.log10(100), 10))
    
    print(f"Total parameters in the model: {total_params}")
    for lam in lambdas:
        l2_penalty, _ = compute_l2_penalty(model)
        l2_value = lam * l2_penalty
        print(f"For lambda = {lam:.5f}, L2 penalty value = {l2_value.item():.5f}")
    
def count_parameters(model):
    nupparam = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {nupparam}')
    return nupparam
    
import math

def max_entropy(categories):
    H_max = -math.log2(1/categories)
    return H_max

    return H_max

def calculate_entropy(sequence):
    # Ensure the sequence is a PyTorch tensor
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence)

    sumSequence = torch.sum(sequence)
    proportions = sequence / (sumSequence+.00001)
    logProps = torch.log(proportions)
    entropy = -torch.sum(proportions * logProps)

    return entropy

def calculate_entropyPropMaxPenalty(sequence):
    # Ensure the sequence is a PyTorch tensor
    n = torch.tensor(sequence.shape[0])
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence)

    sumSequence = torch.sum(sequence)
    proportions = sequence / (sumSequence+.00001)
    logProps = torch.log(proportions)
    entropy = -torch.sum(proportions * logProps)
    max_entropy = torch.log(n)
    entropyPenalty = max_entropy/entropy-1

    return entropyPenalty

def calculate_entropyPropMax(sequence, max_entropy=-1):
    # Ensure the sequence is a PyTorch tensor
    n = torch.tensor(sequence.shape[0])
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence)

    sumSequence = torch.sum(sequence)
    proportions = sequence / (sumSequence+.00001)
    logProps = torch.log(proportions)
    entropy = -torch.sum(proportions * logProps)
    if max_entropy == -1:
        max_entropy = torch.log(n)
    propMax = entropy/max_entropy

    return propMax

def scaledvariance_penalty(sequence):
    mean_val = torch.mean(sequence)
    variance = torch.mean((sequence - mean_val) ** 2)
    scaledvar=  variance/mean_val**2
    return scaledvar

def gini_coefficient(sequence):
    sorted_seq = torch.sort(sequence)[0]
    n = sequence.shape[0]
    index = torch.arange(1, n + 1)
    gini = (torch.sum((2 * index - n  - 1) * sorted_seq)) / (n * torch.sum(sorted_seq))
    return gini

def powered_entropyPenalty(sequence, power=2):
    powered_seq = torch.pow(sequence, power)
    normalized_seq = powered_seq / torch.sum(powered_seq)
    entropy = -torch.sum(normalized_seq * torch.log(normalized_seq))
    max= torch.log(torch.tensor(sequence.shape[0])) 
    entropyPenalty = max/entropy-1
    return entropyPenalty

def GetSoftMin(tensor, beta=100):
    probabilities = F.softmin(tensor * beta, dim=0)
    soft_min = torch.sum(probabilities*tensor)
    return soft_min

def GetSoftMax(tensor, beta=100):
    probabilities = F.softmax(tensor * beta, dim=0)
    soft_max = torch.sum(probabilities*tensor)
    return soft_max

def MaxValueSigmoid(tensorVal, maxFunctionVal, gain=.5):
    # note that a mult of 1 gives a maximum value of .5 so for a target of 't' we need to multiply by 2/t
    gain= torch.tensor(gain)
    mult= torch.tensor(maxFunctionVal*2)
    tensorMult = 10/mult
    expTerm = torch.exp(-tensorVal * tensorMult * gain)
    bottomTerm = expTerm + 1
    fraction = 1/bottomTerm
    temp=fraction -.5
    res =  temp*mult
    return res

def MinValAsymptoticPenalty(val, target, k=3):
    # if value is greater than target return value asymptotic to 0
    # if value is smaller than target rises to a value near 1  at value = 0
    # still has decent slope at this level.
    # only use if target is positive and val will always be positive
    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val)
    targetOver2 = target/2
    diff = (val-targetOver2)
    mult = k/targetOver2
    bottom = 1 + torch.exp(-mult*diff) 
    penalty =-1/bottom+1
    return penalty

def ZeroBelowMinValueRelu(x, minValue):
    reluShortFall = F.relu(x - (minValue - .00001))
    val = reluShortFall+minValue*reluShortFall/(reluShortFall+.000000001)
    return val

def ZeroBelowAbsMinValueRelu(x, minValue):
    absx = torch.abs(x)
    reluShortFall = F.relu(absx - (minValue - .00001))
    val = reluShortFall+minValue*reluShortFall/(reluShortFall+.000000001)
    val = val * x/(absx+.000000001)
    return val

def SmallValBelowAbsMinValueRelu(x, minValue, smallVal):
    absx = torch.abs(x)
    reluShortFall = F.relu(absx - (minValue - .00001))
    oneZeroInOut = reluShortFall/(reluShortFall+.000000001)
    zeroOneInOut = 1 - oneZeroInOut
    val = reluShortFall + minValue*oneZeroInOut + smallVal*zeroOneInOut
    val = val * x/(absx+.000000001)
    return val

def MaxValueRelu(x, maxValue):
    excess = x - maxValue
    val = x - F.relu(excess)
    return val

def MaxValQuadraticPenalty(input, max, target, valAtTarget):
    # use to get a penalty if a value is greater than max.
    # input at max or below returns 0
    # at 'target' returns valAtTarget. Target should be greater than max
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    valMinusMax = F.relu(input - max)
    targetMinusMax = target - max
    frac = valMinusMax/targetMinusMax
    fracSq = frac**2
    penalty =  valAtTarget * fracSq
    return penalty

def MinValQuadraticPenalty(input, min, target, valAtTarget):
    # use to get a penalty if a value is less than min.
    # input at min or above returns 0
    # at 'target' returns valAtTarget. Target should be less than min
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    minMinusVal = F.relu(min - input)
    targetMinusMin = target - min
    frac = minMinusVal/targetMinusMin
    fracSq = frac**2
    penalty =  valAtTarget * fracSq
    return penalty

def MinValAdd(input, min, penalty):
    # add a penalty of 'penalty' if input is less than min otherwise return input
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    minMinusVal = min - input
    minMinusValRelu = F.relu(min - input)
    mult = minMinusValRelu/(minMinusVal+.0000001)
    penalty = penalty * mult
    return input + penalty

def MaxValueQuadratic(tensorVal, maxFunctionVal):
    # note that a mult of 1 gives a maximum value of .5 so for a target of 't' we need to multiply by 2/t
    maxFunctionVal= torch.tensor(maxFunctionVal)
    tv= tensorVal*2
    res = (-1/(4*maxFunctionVal))*tv**2+tv
    return res
    
def GetSoftMedian(tensor):
    n = tensor.numel()
    if n % 2 == 1:
        # Odd number of elements
        values, _ = tensor.flatten().topk(n // 2 + 1)
        return values[-1]
    else:
        # Even number of elements
        values, _ = tensor.flatten().topk(n // 2 + 1)
        return (values[-1] + values[-2]) / 2.0

def GetSoftPercentile(tensor, percentile):
    n = tensor.numel()
    position = int((1-percentile)  * n)
    values, _ = tensor.flatten().topk(position + 1)

    if percentile * n == position:
        # If the percentile is exactly at an integer position
        return (values[-1] + values[-2]) / 2.0
    else:
        return values[-1]

def GetSoftPercentileUpper(tensor, percentile):
    # Ensure tensor is 2D
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D.")
    
    # Get the size of the tensor
    n = tensor.size(0)
    if tensor.size(1) != n:
        raise ValueError("Input tensor must be square.")
    
    # Create a mask for the upper triangle without the diagonal
    mask = torch.triu(torch.ones(n, n, device=tensor.device), diagonal=1).bool()
    
    # Apply the mask to get the upper triangular elements (excluding the diagonal)
    upper_tri_elements = tensor[mask]
    
    # Calculate the position for the percentile
    num_elements = upper_tri_elements.numel()
    position = int((1-percentile) * num_elements)
    
    # Top-k operation to get the percentile value
    values, _ = upper_tri_elements.topk(position + 1)
    
    # Handle the case where the percentile is exactly on an integer position
    if percentile * num_elements == position:
        return (values[-1] + values[-2]) / 2.0
    else:
        return values[-1]

def GetPercentile(tensor, percentile):
    return torch.quantile(tensor.flatten(), percentile)

def PrintParamList(model):
    param_list = list(model.parameters())
    named_children_list = list(model.named_children())

    # Extract names
    param_names = []
    for name, child in named_children_list:
        for _ in child.parameters():
            param_names.append(name)

    # Print names and shapes
    for name, param in zip(param_names, param_list):
        print(f"Name: {name}, Shape: {param.shape}")
        
 
def write_params_and_grads(obj, filename=None):
    if filename is None:
        for name, param in obj.named_parameters():
            print(f"Parameter {name}:")
            print("Value:", param.data)
            print("Gradient:", param.grad)
    else:
        with open(filename, "w") as f:
            # Write headings "Parameter"  "Value" and "Gradient" at the top of the file at 0, 20 and 40 characters
            f.write(f"Parameter{' ' * 20}Value{' ' * 20}Gradient\n")
            for name, param in obj.named_parameters():
                f.write(f"{name}{' ' * (20 - len(name))}{param.data}{' ' * (20 - len(str(param.data)))}{param.grad}\n")

def hasGradients(obj):
    for name, param in obj.named_parameters():
        if param.grad is not None:
            print(f"Parameter {name}: HasGrad = True")
        else:
            print(f"Parameter {name}: HasGrad = False")

import numpy as np

def GradToArray(obj, ParamName):
    # Find the parameter with the given name
    for name, param in obj.named_parameters():
        if name == ParamName:
            # Check if the parameter has a gradient
            if param.grad is not None:
                return param.grad.cpu()
            else:
                return None
    raise ValueError(f"No parameter named {ParamName} in the given object.")


def kurtosis_of_columns(tensor):
    B, T, C = tensor.shape
    n = B * T  # Sample size

    # Reshape tensor to group by columns (flattening batches and time steps)
    flat_tensor = tensor.reshape(n, C)
    
    mean = torch.mean(flat_tensor, dim=0)
    var = torch.var(flat_tensor, unbiased=True, dim=0)  # Using unbiased variance
    std = torch.sqrt(var)

    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    term2 = torch.sum(((flat_tensor - mean) / std) ** 4, dim=0)
    term3 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    
    kurt = term1 * term2 - term3

    return kurt

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = torch.tensor(quantiles)
        
        # Getting the theoretical quantiles for the standard normal distribution
        self.theoretical_quantiles = torch.tensor(scipy.stats.norm.ppf(quantiles)).float()
        self.a=3
            
    def soft_quantile(self, x, q):
        # Soft quantile computation
        ret = torch.mean(torch.relu(x - q) - torch.relu(-x + q))
        return ret
            
    def forward(self, x):
        # Assuming x is already standardized (zero mean, unit variance)
        empirical_quantiles = [self.soft_quantile(x, q) for q in self.quantiles]
        losslist = [(emp - theo)**2 for emp, theo in zip(empirical_quantiles, self.theoretical_quantiles)]
        loss = sum([(emp - theo)**2 for emp, theo in zip(empirical_quantiles, self.theoretical_quantiles)])
        return loss
    
    def printBins(self):
        for quant, theo in zip(self.quantiles, self.theoretical_quantiles):
            print(f"Quantile: {quant:.2f}, BinVal: {theo:.2f}")

class NormalDistributionChecker1D(nn.Module):
    def __init__(self, quantiles):
        super(NormalDistributionChecker1D, self).__init__()
        
        # self.quantiles = torch.tensor(quantiles)
        # self.numQuantiles = len(quantiles)
        # self.numBins = self.numQuantiles + 1
        # self.quantile_ZScores = torch.tensor(scipy.stats.norm.ppf(quantiles)).float()
        # self.expectedPercentages = torch.tensor([quantiles[0]] + 
        #     [quantiles[i] - quantiles[i-1] for i in range(1, self.numQuantiles)] + 
        #     [1 - quantiles[-1]]).float()
        # self.chiSquaredPValue = DifferentiableChiSquaredPValue(self.numQuantiles, quantiles)
        
        self.quantiles = torch.tensor(quantiles)
        self.numQuantiles = len(quantiles)
        self.numBins = self.numQuantiles+1
        self.quantile_ZScores = torch.tensor(scipy.stats.norm.ppf(quantiles)).float()
        self.expectedPercentages = torch.tensor([quantiles[0]] + [quantiles[i] - quantiles[i-1] 
            for i in range(1, self.numQuantiles)] + [1 - quantiles[-1]]).float()
        self.chiSquaredPValue = DifferentiableChiSquaredPValue(self.numQuantiles, self.quantiles)

        
    def forward(self, x):
        # Assume x is a one-dimensional tensor
        n = x.size(0)

        # Calculate the Z-scores for the input data 
        mean = x.mean()
        std = x.std()
        z_scores = (x - mean) / std
        
        # Compute the expected counts for each bin
        expected_counts = n * self.expectedPercentages
        
        actual_counts = torch.zeros(self.numBins)
        count_lower = torch.sigmoid(100 * (self.quantile_ZScores[0] - z_scores)).sum()
        actual_counts[0] = count_lower
        total = count_lower
        
        for i in range(1, self.numBins - 1):
            count_upper = torch.sigmoid(100 * (self.quantile_ZScores[i] - z_scores)).sum()
            count = count_upper - count_lower
            actual_counts[i] = count
            total += count
            count_lower = count_upper
        
        remainder = n - total
        actual_counts[self.numBins - 1] = remainder
        
        probNonNorm = self.chiSquaredPValue(actual_counts, expected_counts)
        
        return probNonNorm
    
class NormalDistributionChecker2D(nn.Module):
    def __init__(self, quantiles):
        super(NormalDistributionChecker2D, self).__init__()
        
        self.quantiles = torch.tensor(quantiles)
        self.numQuantiles = len(quantiles)
        self.numBins = self.numQuantiles+1
        self.quantile_ZScores = torch.tensor(scipy.stats.norm.ppf(quantiles)).float()
        self.expectedPercentages = torch.tensor([quantiles[0]] + [quantiles[i] - quantiles[i-1] 
            for i in range(1, self.numQuantiles)] + [1 - quantiles[-1]]).float()
        self.chiSquaredPValue = DifferentiableChiSquaredPValue(self.numQuantiles, self.quantiles)
        
    def forward(self, x):
        T, C = x.shape
        nonNormStats = torch.zeros(C)
        
        for c in range(C):
            column_data = x[:, c]
            n = column_data.size(0)

            # Calculate the Z-scores for the input data 
            mean = column_data.mean()
            std = column_data.std()
            z_scores = (column_data - mean) / std

            # Compute the expected counts for each bin
            expected_counts = n * self.expectedPercentages

            actual_counts = torch.zeros(self.numBins)
            count_lower = torch.sigmoid(100 * (self.quantile_ZScores[0] - z_scores)).sum()
            actual_counts[0] = count_lower
            total = count_lower

            for i in range(1, self.numBins - 1):
                count_upper = torch.sigmoid(100 * (self.quantile_ZScores[i] - z_scores)).sum()
                count = count_upper - count_lower
                actual_counts[i] = count
                total += count
                count_lower = count_upper

            remainder = n - total
            actual_counts[self.numBins - 1] = remainder
            
            probNonNorm = self.chiSquaredPValue(actual_counts, expected_counts)
            nonNormStats[c] = probNonNorm

        return nonNormStats

class DifferentiableChiSquaredPValue(nn.Module):
    # returns approx probability with which we can reject the null hypothesis that actual and expected have the same distribution
    def __init__(self, df: int, target_p_values: torch.Tensor):
        super(DifferentiableChiSquaredPValue, self).__init__()

        # Precompute the chi-squared statistics for the target p-values
        critical_values = torch.tensor([chi2.ppf(1 - p, df) for p in target_p_values])
        self.maxCriticalValue = torch.max(critical_values)
        
        self.register_buffer('critical_values', critical_values)
        self.register_buffer('target_p_values', target_p_values)

    def forward(self, actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
        # Compute the chi-squared statistic
        chi2_statistic = torch.sum((actual - expected)**2 / (expected + 1e-7))
        # Differentiable interpolation
        p_value_estimate = 1- self._differentiable_interpolate(chi2_statistic, self.critical_values, self.target_p_values)
        excess_chi2_statistic = torch.relu((chi2_statistic - self.maxCriticalValue)/100.0)
        ret = p_value_estimate + excess_chi2_statistic

        return ret

    def _differentiable_interpolate(self, x, x_vals, y_vals):
        """Differentiable linear interpolation function."""
        weights = torch.nn.functional.softmax(-torch.abs(x - x_vals), dim=0)
        return torch.sum(weights * y_vals)

    
# function to write the gradients of the parameters of a model to a file if a file is specified if not print them to the console. The function should print a heading Name, Value, Gradient seperated by tabs and then for each parameter in the model print the name of the parameter, the value of the parameter, and the gradient of the parameter on the same line for each parameter in the model seperated by tabs
def WriteParamsAndGrads(model, filename=None):
    if filename is None:
        for name, param in model.named_parameters():
            print(f"Parameter {name}: Value: {param.data}, Gradient: {param.grad}")
    else:
        with open(filename, "w") as f:
            # Write headings "Parameter"  "Value" and "Gradient" at the top of the file at 0, 20 and 40 characters
            f.write(f"Parameter\tValue\tGradient\n")
            for name, param in model.named_parameters():
                f.write(f"{name}\t{param.data}\t{param.grad}\n")


def WriteGradients(model, outputFile):
    with open(outputFile, 'w') as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # For single value parameters
                if param.ndimension() == 0:
                    grad = param.grad.item() if param.grad is not None else 'None'
                    f.write(f"Parameter {name}: Value: {param.data.item()}, Gradient: {grad}\n")

                # For 2D parameters
                elif param.ndimension() == 2:
                    f.write(f"Parameter {name}\n")
                    f.write("Values\t" + "\t" * (param.size(1) - 1) + "\tGradients\n")
                    for i in range(param.size(0)):
                        values = '\t'.join(f"{v.item():.4f}" for v in param[i])
                        # Check the gradient for each row
                        if param.grad is not None:
                            gradients = '\t'.join(f"{g.item():.4f}" for g in param.grad[i])
                        else:
                            gradients = 'None'
                        f.write(f"{values}\t\t{gradients}\n")

                # For 3D parameters
                elif param.ndimension() == 3:
                    f.write(f"Parameter {name}\n")
                    for k in range(param.size(0)):
                        f.write(f"Slice {k}:\n")
                        f.write("Values\t" + "\t" * (param.size(2) - 1) + "\tGradients\n")
                        for i in range(param.size(1)):
                            values = '\t'.join(f"{v.item():.4f}" for v in param[k][i])
                            # Check the gradient for each [k][i] row
                            if param.grad is not None:
                                gradients = '\t'.join(f"{g.item():.4f}" for g in param.grad[k][i])
                            else:
                                gradients = 'None'
                            f.write(f"{values}\t\t{gradients}\n")
                        f.write("\n")  # Separate the slices by a newline

                f.write("\n")  # Separate parameters by a newline

def WriteGradients2(model, outputFile):
    with open(outputFile, 'w') as f:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # For single value parameters
                if param.ndimension() == 0:
                    grad = param.grad.item() if param.grad is not None else 'None'
                    ratio = grad / (abs(param.data.item()) + 1e-8) if grad != 'None' else 'None'
                    f.write(f"Parameter {name}: Value: {param.data.item()}, Gradient: {grad}, Ratio: {ratio:.4e}\n")

                # For 2D parameters
                elif param.ndimension() == 2:
                    f.write(f"Parameter {name}\n")
                    # Heading with appropriate tab spaces
                    f.write("Values" + "\t" * (param.size(1)+1) + "Gradients" + "\t" * (param.size(1)+1) + "Gradient/AbsValue\n")
                    for i in range(param.size(0)):
                        values = '\t'.join(f"{v.item():.4f}" for v in param[i])
                        gradients = 'None\t' * param.size(1)
                        ratios = 'None\t' * param.size(1)
                        if param.grad is not None:
                            gradients = '\t'.join(f"{g.item():.4f}" for g in param.grad[i])
                            ratios = '\t'.join(f"{(g / (abs(v.item()) + 1e-8)).item():.4e}" for v, g in zip(param[i], param.grad[i]))
                        # Adding an extra tab after values to align the columns
                        f.write(f"{values}\t\t{gradients}\t\t{ratios}\n")

                # For 3D parameters
                elif param.ndimension() == 3:
                    f.write(f"Parameter {name}\n")
                    for k in range(param.size(0)):
                        f.write(f"Slice {k}:\n")
                        # Heading with appropriate tab spaces
                        f.write("Values" + "\t" * (param.size(2)+1) + "Gradients" + "\t" * (param.size(2)+1) + "Gradient/AbsValue\n")
                        for i in range(param.size(1)):
                            values = '\t'.join(f"{v.item():.4f}" for v in param[k][i])
                            gradients = 'None\t' * param.size(2)
                            ratios = 'None\t' * param.size(2)
                            if param.grad is not None:
                                gradients = '\t'.join(f"{g.item():.4f}" for g in param.grad[k][i])
                                ratios = '\t'.join(f"{(g / (abs(v.item()) + 1e-8)).item():.4e}" for v, g in zip(param[k][i], param.grad[k][i]))
                            # Adding an extra tab after values to align the columns
                            f.write(f"{values}\t\t{gradients}\t\t{ratios}\n")
                        f.write("\n")  # Separate the slices by a newline

                f.write("\n")  # Separate parameters by a newline

def GetPctAboveMarkAbsCorrelation(Input, maxAbsCorr = -1, daysToUse = -1):
    B, T, C = Input.shape
    total_rows = B * T
    
    # if days to use is smaller than T then reduce each block to the last daysToUse days
    if daysToUse > 0 and daysToUse < T:
        inputAdj = Input[:, -daysToUse:, :]
        total_rows = B * daysToUse
    else:
        inputAdj = Input

    # Flatten the first two dimensions of the input tensor to treat them as 'rows'
    flat_Input = inputAdj.reshape(-1, C)

    # Normalize the data
    mean = flat_Input.mean(dim=1, keepdim=True)
    std = flat_Input.std(dim=1, keepdim=True, unbiased=True)
    norm_Input = (flat_Input - mean) / std

    # Compute the correlation matrix by multiplying the normalized matrix with its transpose
    corr_matrix = torch.mm(norm_Input, norm_Input.t())/(C-1)

    # Fill the diagonal of the correlation matrix with zeros before computing the average
    torch.diagonal(corr_matrix).fill_(0)

    # Compute the average of the absolute values of the upper triangular part of the matrix, excluding the diagonal
    abs_corr = torch.abs(corr_matrix)
    # make abs_cor require gradient if it doesnt alreasy
    if not abs_corr.requires_grad:
        abs_corr.requires_grad = True
        
    upper_tri_indices = torch.triu_indices(row=abs_corr.size(0), col=abs_corr.size(1), offset=1, device=inputAdj.device )
    avg_abs_corr = abs_corr[upper_tri_indices[0], upper_tri_indices[1]].mean()
    percentAboveMark=-1
    if(maxAbsCorr > -1):
        totalVals = (total_rows * total_rows)         
        numAboveLevel = countAboveLevel(abs_corr, maxAbsCorr)
        percentAboveMark = numAboveLevel / totalVals

    return avg_abs_corr, percentAboveMark


def countAboveLevel(tensor, level, beta=100):
    adjusted_tensor = beta * (tensor - level)
    
    # Sigmoid function to approximate the step function
    # The result will be close to 1 for elements above the level and close to 0 otherwise
    count = torch.sigmoid(adjusted_tensor).sum()
    return count

def soft_percentile_count(tensor, percentile, steps=100):
    # Define the range for the cutoff search
    min_val, max_val = tensor.min(), tensor.max()
    delta = (max_val - min_val) / steps

    # Function to softly count the elements above a threshold
    def soft_count(threshold):
        # Use a sigmoid to create a soft count of values greater than the threshold
        return torch.sigmoid(10 * (tensor - threshold)).sum()

    # Find the threshold that gives us the desired percentile count
    start, end = min_val, max_val
    for _ in range(steps):
        mid = (start + end) / 2
        count = soft_count(mid)
        if count / tensor.numel() < percentile:
            start = mid
        else:
            end = mid
    
    return (start + end) / 2

def GetBottomValuesAvUtil(InputTensor, numBottomValues):
    # Flatten the tensor first
    flattened = InputTensor.view(-1)
    # Now use topk on the flattened tensor
    bottom_values, _ = torch.topk(flattened, numBottomValues, largest=False)
    # Calculate the average of the bottom values
    avBottom = torch.mean(bottom_values)
    return avBottom

# Custom differentiable percentile function
def differentiable_quantile(x, q):
    # Assuming x is your input tensor and q is the quantile in [0, 1]
    assert 0 <= q <= 1

    # Smooth approximation of the step function
    # The scale factor can be adjusted for a sharper or smoother approximation
    scale = 100.0
    
    # Compute the threshold that would be the qth quantile
    kth_value = q * (x.numel() - 1)
    target_value = x.view(-1).sort().values[int(kth_value)]

    # Apply the sigmoid to approximate the indicator function
    weights = torch.sigmoid(scale * (x - target_value))
    
    # Compute the weighted sum of the tensor
    quantile_value = torch.sum(weights * x) / torch.sum(weights)
    
    return quantile_value

def MaskSmallValues(input, threshold, scale=100.0):
    weights = F.sigmoid((input - threshold) * scale)
    maskedValues = input * weights
    return maskedValues, weights

def MaskVerySmallValuesAsLarge(tensor, threshold=0.001, high_value=100, scale_factor=10):
    # Creating a mask for values below the threshold
    mask = torch.sigmoid(scale_factor/threshold * (threshold - tensor))
    # Applying the transformation: For values below the threshold, change to high_value; otherwise, keep original
    modified_tensor = mask * high_value + (1 - mask) * tensor
    return modified_tensor

def MaskBelowAbsVal(input, threshold):
    input = input
    intputMinTh= input-threshold
    weightsPosElements = F.relu(intputMinTh)/(intputMinTh+.0000001)
    weightsNegElements = F.relu(-intputMinTh)/(-intputMinTh+.0000001)
    weights= weightsPosElements + weightsNegElements
    maskedValues = input * weights
    return maskedValues, weights

def NumColsBelowMinElements(input, minval, minElementsPerCol, scale=100.0):
    input = input
    numcols = input.shape[1]
    sd= torch.std(input)
    temp = torch.ones_like(input)
    weights = F.sigmoid((minval-input) * (scale/sd)*2)
    OneZeroMat = temp * (1-weights)
    sumColumnsOfOneZeroMat = torch.sum(OneZeroMat, dim=0)
    
    ret=NumElementsBelowMark(sumColumnsOfOneZeroMat, minElementsPerCol*.9)
    return ret

def NumColsBelowMinElementsSoftOneZero(input, minval, minElementsPerCol, minCols, scale=100.0):
    input = input
    numcols = input.shape[1]
    sd= torch.std(input)
    temp = torch.ones_like(input)
    weights = F.sigmoid((minval-input) * (scale/sd)*2)
    OneZeroMat = temp * (1-weights)
    sumColumnsOfOneZeroMat = torch.sum(OneZeroMat, dim=0)
    
    colsWithLessThanMinElAtMark=NumElementsBelowMark(sumColumnsOfOneZeroMat, minElementsPerCol*.9)
    ret= F.sigmoid((colsWithLessThanMinElAtMark- minCols-.2) * scale)
    
    return ret

def NumElementsBelowMark(input, threshold, scale=100.0):
    input = input
    sd= torch.std(input)
    numEl = input.shape[0]
    temp = torch.ones_like(input)
    weights = F.sigmoid((input - threshold) * (scale/sd)*2)
    oneZeroVec = temp * (1-weights)
    ret = torch.sum(oneZeroVec, dim=0)
    
    ret=sum(oneZeroVec)
    return ret

def MeasureExecutionTime(func, num_tests, randomSeed=-1, *args, **kwargs):
    if randomSeed > -1:
        set_seed(randomSeed)
    
    start_time = time.time()
    last_result = None
    for _ in range(num_tests):
        last_result = func(*args, **kwargs)
    end_time = time.time()
    average_time = (end_time - start_time) / num_tests
    
    return average_time, last_result

def gini_coefficient(tensor):
    n = tensor.numel()
    if n == 0:
        return 0.0  # To prevent division by zero
    
    # Compute mean of the tensor
    mean_obs = torch.mean(tensor).item()
    
    # Calculate the sum of absolute differences
    sum_abs_diffs = torch.sum(torch.abs(tensor.unsqueeze(0) - tensor.unsqueeze(1))).item()
    
    # Calculate the Gini coefficient using the provided formula
    gini = (sum_abs_diffs / (n * (n - 1) * mean_obs))/2
    
    return gini

def WriteParamsToFile(model, file_name = g.tempOutputFile):
    categorized_params = defaultdict(list)
    with open(file_name, 'w') as file:
        for name, param in model.named_parameters():
            # Splitting the name by '.' to get the module name
            module_name = name.split('.')[0]
            categorized_params[module_name].append((name, param.shape))

        for module, params in categorized_params.items():
            file.write(f"\n{module} Parameters:\n")
            for name, shape in params:
                file.write(f"  {name}: {shape}\n")

def custom_sigmoid(x, scale, limit):
    # Apply the sigmoid function to input x
    sigmoid_output = torch.sigmoid(x*scale)
    
    # Scale the output from (0, 1) to (0, 2*limit)
    scaled_output = 2 * limit * sigmoid_output
    
    # Shift the output to center it around 0, resulting in (-limit, limit)
    shifted_output = scaled_output - limit
    
    return shifted_output

def sigmoidMinMax(x, min, max, scale=10):
    # Apply the sigmoid function to input x
    range = max-min
    x = torch.sigmoid((x-min-(range/2))*scale/range)*range
    x = x+min
    
    return x

def gini_coefficient(tensor):
    n = tensor.numel()
    if n == 0:
        return 0.0  # To prevent division by zero
    
    # Compute mean of the tensor
    mean_obs = torch.mean(tensor).item()
    
    # Calculate the sum of absolute differences
    sum_abs_diffs = torch.sum(torch.abs(tensor.unsqueeze(0) - tensor.unsqueeze(1))).item()
    
    # Calculate the Gini coefficient using the provided formula
    gini = (sum_abs_diffs / (n * (n - 1) * mean_obs))/2
    
    return gini

def MergeTensors(tensorList, width):
    # Flatten each tensor to 1D
    flattened_tensors = [tensor.flatten() for tensor in tensorList]
    
    # Concatenate all flattened tensors into a single 1D tensor
    concatenated_tensor = torch.cat(flattened_tensors)
    
    # Calculate the total number of elements
    total_elements = concatenated_tensor.size(0)
    
    # Calculate the required number of zeros to add to make the length divisible by width
    remainder = total_elements % width
    if remainder != 0:
        zeros_needed = width - remainder
        # Create a tensor of zeros with the required number of elements
        zeros_tensor = torch.zeros(zeros_needed, dtype=concatenated_tensor.dtype)
        # Append zeros to the concatenated tensor
        concatenated_tensor = torch.cat((concatenated_tensor, zeros_tensor))
    
    # Calculate the new total number of elements after padding
    total_elements_padded = concatenated_tensor.size(0)
    # Calculate the height of the 2D tensor
    height = total_elements_padded // width
    
    # Reshape the 1D tensor into a 2D tensor with the specified width
    reshaped_tensor = concatenated_tensor.view(height, width)
    
    return reshaped_tensor

def getDFFromFile(fileNameOrTicker, filePath=filepathRawData, isLive=False):
    # get the path and tickerlist[] from the command line
    # specifiy the path as C:\Users\Phil\Documents\Python Projects\Get Data\DataFiles\
    if isLive:
        path = filepathRawDataLive
        fileNameOrTicker = fileNameOrTicker + "_Live"
    else:
        path = filePath

    # get the file name
    fileNameWithPath = path + fileNameOrTicker + ".csv"
    # read the data from the file into a dataframe
    df = pd.read_csv(fileNameWithPath)
    # read the first row of the dataframe as titles for each column
    # df.columns = df.iloc[0]
    # convert the date column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'].str[:10], format='%Y-%m-%d')
    # Set the date column as the index
    df = df.set_index('Date')
    # convert the columns to numeric
    df = df.apply(pd.to_numeric)

    return df    

def UniqueRandomValues(numToGet, maxValue, minValue=0):
    # Ensure n is not greater than m to avoid repeats
    if numToGet + minValue > maxValue+1:
        raise ValueError("n must be less than or equal to m to avoid repeats.")
    
    # Generate a random permutation of integers from 0 to m-1, then select the first n integers
    indices = torch.randperm(maxValue+1)[:numToGet]
    
    return indices
    
def GetTopElements(listIn, numToGet):
    # Pair each element with its index and sort by the element value in descending order
    sorted_pairs = sorted(enumerate(listIn), key=lambda x: x[1], reverse=True)
    
    # Get the top n elements and their indices
    top_n = sorted_pairs[:numToGet]
    
    # Optionally, you can format the output to separate indices and elements
    indices, elements = zip(*top_n)  # This step separates the indices and elements into two tuples

    return list(indices), list(elements)

from datetime import datetime

from datetime import datetime

from datetime import datetime

def parse_datetime(date_str):
    formats = [
        "%Y-%m-%d %I:%M:%S%p", "%Y-%m-%d %I:%M%p", "%Y-%m-%d",
        "%d-%m-%Y %I:%M:%S%p", "%d-%m-%Y %I:%M%p", "%d-%m-%Y",
        "%m/%d/%Y %I:%M:%S%p", "%m/%d/%Y %I:%M%p", "%m/%d/%Y",
        "%Y%m%d%I%M%S%p", "%Y%m%d%I%M%p", "%Y%m%d",
        "%b %d, %Y %I:%M:%S%p", "%b %d, %Y %I:%M%p", "%b %d, %Y",
        "%B %d, %Y %I:%M:%S%p", "%B %d, %Y %I:%M%p", "%B %d, %Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
        "%b %d, %Y %H:%M:%S", "%b %d, %Y %H:%M",
        "%B %d, %Y %H:%M:%S", "%B %d, %Y %H:%M"
    ]
    
    # Normalize AM/PM case
    date_str = date_str.replace("am", "AM").replace("pm", "PM")

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt), True
        except ValueError:
            continue
    
    return None, False  # Indicate failure

def deep_getsizeof(obj, seen=None):
    """Recursively find the memory footprint of a Python object."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid infinite recursion
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(deep_getsizeof(v, seen) for v in obj.values())
        size += sum(deep_getsizeof(k, seen) for k in obj.keys())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_getsizeof(i, seen) for i in obj)
    return size

def WriteJSONToMindMapOld(jsonObj, filename=g.tempOutputFile):
    def recurse(obj, level=0, key_name=None, lines=None):
        if lines is None:
            lines = []

        indent = '\t' * level

        # Handle dictionary
        if isinstance(obj, dict):
            if key_name is not None:
                lines.append(f"{indent}{key_name}")
                indent += '\t'
            for k, v in obj.items():
                recurse(v, level + 1, k, lines)

        # Handle list
        elif isinstance(obj, list):
            if key_name is not None:
                lines.append(f"{indent}{key_name}")
            for i, item in enumerate(obj, 1):
                node_label = f"{key_name}_{i}" if key_name else f"item_{i}"
                recurse(item, level + 1, node_label, lines)

        # Handle scalar
        else:
            if key_name is not None:
                lines.append(f"{indent}{key_name}: {obj}")
            else:
                lines.append(f"{indent}{obj}")

        return lines

    lines = recurse(jsonObj)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# function to convert dictionary object to a mindmap
def WriteDictToMindMapOld(dictObj, filename=g.tempOutputFile):
    def recurse(obj, level=0, key_name=None, lines=None):
        if lines is None:
            lines = []

        indent = '\t' * level

        if isinstance(obj, dict):
            for k, v in obj.items():
                lines.append(f"{indent}{k}")
                recurse(v, level + 1, None, lines)

        elif isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                label = f"item_{i}"
                lines.append(f"{indent}{label}")
                recurse(item, level + 1, None, lines)

        else:
            lines.append(f"{indent}{obj}")

        return lines

    lines = recurse(dictObj)
    with open(filename, 'w', encoding='utf-8') as f:
        # Convert to ASCII-safe form
        f.write('\n'.join(lines).encode('ascii', 'backslashreplace').decode('ascii'))

def WriteJSONToMindMap(jsonObj, filename=g.tempOutputFile): # Kept original default filename
    """
    Converts a JSON-like object (dict/list structure) to a hierarchical
    text format, replacing common non-ASCII punctuation with ASCII
    equivalents using cleanText() before writing.
    Writes the result using ASCII encoding.
    """
    def recurse(obj, level=0, key_name=None, lines=None):
        if lines is None:
            lines = []

        indent = '\t' * level

        # Handle dictionary
        if isinstance(obj, dict):
            if key_name is not None:
                # Clean the key name itself before appending
                clean_key_name = cleanText(key_name) # Use cleanText
                lines.append(f"{indent}{clean_key_name}")
            # This version puts the key on its own line, then recurses
            # Adjust indentation for children
            # child_indent = '\t' * (level + 1) # Not strictly needed if children indent themselves
            for k, v in obj.items():
                 # Pass the original key 'k' as the name for the value 'v'
                recurse(v, level + 1, k, lines)

        # Handle list
        elif isinstance(obj, list):
            if key_name is not None:
                 # Clean the key name itself before appending
                clean_key_name = cleanText(key_name) # Use cleanText
                lines.append(f"{indent}{clean_key_name}")
            # This version puts the list name on its own line, then recurses items
            # child_indent = '\t' * (level + 1) # Not strictly needed
            for i, item in enumerate(obj): # Using 0-based index for label
                # Create a label like "parent_key[0]", "parent_key[1]", etc.
                node_label = f"{key_name}[{i}]" if key_name else f"item[{i}]"
                 # Clean the generated label before passing it down
                clean_node_label = cleanText(node_label) # Use cleanText
                 # Pass cleaned label for the list item
                recurse(item, level + 1, clean_node_label, lines)

        # Handle scalar (string, number, bool, None)
        else:
            # Convert to string and clean the scalar value
            obj_str = str(obj)
            clean_obj_str = cleanText(obj_str) # Use cleanText

            if key_name is not None:
                 # Clean the key name before appending the line
                clean_key_name = cleanText(key_name) # Use cleanText
                lines.append(f"{indent}{clean_key_name}: {clean_obj_str}")
            else: # Safeguard for root scalar
                 lines.append(f"{indent}{clean_obj_str}")

        return lines

    try:
        lines = recurse(jsonObj)
        # Join lines (already cleaned individually)
        full_output_string = '\n'.join(lines)

        # Write the resulting string (should be mostly ASCII now)
        # Using ASCII encoding is safe because we replaced problematic chars.
        # errors='ignore' will drop any remaining non-ASCII chars we didn't clean.
        with open(filename, 'w', encoding='ascii', errors='ignore') as f:
             f.write(full_output_string)
        # print(f"Mind map data (ASCII cleaned) written to {filename}")

    except Exception as e:
        print(f"Error writing JSON to mind map {filename}: {e}")

def WriteDictToMindMap(dictObj, filename=g.tempOutputFile): # Changed default filename
    def recurse(obj, level=0, key_name=None, lines=None):
        if lines is None:
            lines = []

        indent = '\t' * level

        if isinstance(obj, dict):
            for k, v in obj.items():
                # Clean the key before appending
                clean_key = cleanText(k) # Use the revised function
                lines.append(f"{indent}{clean_key}")
                recurse(v, level + 1, None, lines)

        elif isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                label = f"item_{i}" # Label is already ASCII
                lines.append(f"{indent}{label}")
                recurse(item, level + 1, None, lines)

        else:
            # Clean the scalar value before appending
            obj_str = str(obj)
            clean_obj_str = cleanText(obj_str) # Use the revised function
            lines.append(f"{indent}{clean_obj_str}")

        return lines

    try:
        lines = recurse(dictObj)
        full_output_string = '\n'.join(lines) # Lines are already cleaned

        with open(filename, 'w', encoding='ascii', errors='ignore') as f:
            f.write(full_output_string)

    except Exception as e:
        print(f"Error writing dictionary to mind map {filename}: {e}")

def WriteStructToMindMap(structObj, filename=g.tempOutputFile):
    def recurse(obj, level=0, key_name=None, lines=None):
        if lines is None:
            lines = []

        indent = '\t' * level

        if isinstance(obj, dict):
            # Iterate through dictionary items
            for k, v in obj.items():
                # Clean the key before appending it as a node
                clean_key = cleanText(k)
                lines.append(f"{indent}{clean_key}")
                # Recurse for the value, passing None as key_name, as the value
                # doesn't inherently carry the key's name in this format.
                recurse(v, level + 1, None, lines)

        elif isinstance(obj, list):
            # Iterate through list items
            for i, item in enumerate(obj, 1):
                # Create a generic label for the list item
                label = f"item_{i}" # Label is already ASCII
                lines.append(f"{indent}{label}")
                # Recurse for the item, passing None as key_name
                recurse(item, level + 1, None, lines)

        else:
            obj_str = str(obj)
            clean_obj_str = cleanText(obj_str)
            # Append the cleaned scalar value, indented
            lines.append(f"{indent}{clean_obj_str}")

        return lines

    try:
        lines = recurse(structObj, level=0) # Initial level is 0
        full_output_string = '\n'.join(lines) # Lines are already cleaned

        with open(filename, 'w', encoding='ascii', errors='ignore') as f:
            f.write(full_output_string)

    except Exception as e:
        print(f"Error writing structure to mind map {filename}: {e}")

def WriteDictOrJsonToMM(Obj, filename=g.tempOutputFile):
    if isinstance(Obj, dict or list):
        WriteStructToMindMap(Obj, filename)
    elif isinstance(Obj, str):
        try:
            parsed = json.loads(Obj)
            WriteJSONToMindMap(parsed, filename)
        except json.JSONDecodeError:
            raise ValueError("Input string is not valid JSON.")
    else:
        raise TypeError("Input must be a dict or JSON string.")

def count_chars_in_dict(data) -> int:
    """Recursively counts characters in all keys and string-convertible values of a nested dictionary."""
    total = 0

    if isinstance(data, dict):
        for key, value in data.items():
            total += len(str(key))+3+len(data)-1  # +1 for colon and +1 for comma
            # count items in dict
            total += count_chars_in_dict(value)  # recursive step
    elif isinstance(data, list):
        for item in data:
            total += count_chars_in_dict(item)+2+len(data)-1  # +2 for brackets and +1 for comma
    else:
        total += len(str(data))

    return total

def has_string(input_to_search, string_to_look_for):
    def search(obj, target, path):
        # Convert non-str targets to string when checking
        if isinstance(obj, str):
            if target in obj:
                return True, ", ".join(map(str, path))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                found, loc = search(value, target, path + [key])
                if found:
                    return True, loc
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                found, loc = search(item, target, path + [idx])
                if found:
                    return True, loc
        elif isinstance(obj, (int, float, bool)):
            if target in str(obj):
                return True, ", ".join(map(str, path))
        return False, ""

    return search(input_to_search, string_to_look_for, [])
