# -*- coding: utf-8 -*-
"""myBoxDetection

**SETUP**
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5  # clone
# %cd yolov5
# %pip install -qr requirements.txt  # install

import torch
import utils
display = utils.notebook_init()  # checks
# %matplotlib inline
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # for image display
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from skimage.util.dtype import img_as_int
import glob
from pathlib import Path
import random
import imgaug.augmenters as iaa
import math
import shutil
import os

"""**INITIALIZATION OF WORK PARAMETERS**"""

# IMAGE AUGMENTATION
max_angle, min_angle, step_angle = 40, 2, 2                  # Rotation
min_gamma, max_gamma, step_gamma = 1, 30, 2                  # Gamma correction, keep integer numbers
min_boxFilter, max_boxFilter, step_boxFilter = 10, 40, 10    # Box Filter, max_boxFilter suggest is 50

# DATASET TRAIN-TEST-VAL SPLIT
testSize, valSize = 0.2, 0.2

# YOLO 
number_of_epochs = 30
batch_size = 32                                             # keep a multiple of 8

"""**FUNCTIONS**"""

def ListImageCoordBox(listRows):

  listData = []

  # COLUMNS : CoordX, CoordY, WidthBox, HeightBox
  for line in listRows:
      listAux = []
      listAux.append(line[0])             
      listAux.append(line[1])
      listAux.append(line[0] + line[2])
      listAux.append(line[1])
      listAux.append(line[0] + line[2])
      listAux.append(line[1] + line[3])
      listAux.append(line[0])
      listAux.append(line[1] + line[3])   

      listAux.append(line[4]) # class

      listData.append(listAux)

  return listData

# ---------------------------------------------------------------------

def ListCustomAux(df_start, img_name):

  output = [] # CoordX, CoordY, WidthBox, HeightBox
  col_searching = df_start['image_name']

  for indexRow in range(0, len(df_start), 1):

    if col_searching[indexRow].find(img_name) >= 0:
       row = df_start.values[indexRow, 0:5]
       output.append(row)

  return output

# --------------------------------------------------------------------

def RowYoloFormat(img, coordinates):
      
      centreX = ((coordinates[0] + coordinates[2]) / 2)
      centreY = ((coordinates[1] + coordinates[5]) / 2)
      width = ((coordinates[0] - centreX)*2)
      height = ((coordinates[1] - centreY)*2)
      centreX_norm = round(centreX / img.shape[1], 5)
      centreY_norm = round(centreY / img.shape[0], 5)
      height_norm = round(abs(width / img.shape[1]), 5)
      width_norm = round(abs(height / img.shape[0]), 5)

      classBox = coordinates[8]

      out = str(classBox) + ' ' + str(centreX_norm) + ' '+ str(centreY_norm) + ' ' + str(height_norm) + ' ' + str(width_norm)

      return out

# ------------------------------------------------------------------------------

def ConvertStrToList(input):
    out = list(input.split(" "))
    return out

def AugmentationRotation(img, indexImg, indexLenList, labels_original, classBox,
              min_angle, max_angle, step_angle,
              imagesStart_labelsStart_AuxPath):

  for indexAngle in range(min_angle, max_angle, step_angle):

      rows, cols, ch = img.shape

      rotationMatrix = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), indexAngle, 1)
      img_rotated = cv2.warpAffine(img, rotationMatrix, (cols, rows))
            
      nameSavingImageRot = Path(indexImg).stem + '_rot' + str(indexAngle) + '.jpg'
      pathSaveImgRot = imagesStart_labelsStart_AuxPath + nameSavingImageRot
      cv2.imwrite(pathSaveImgRot,img_rotated)

      labels_rotated =  labels_original @ rotationMatrix.T
            
      labels_rotated = np.reshape(labels_rotated, (1, 10)) # 2x5 to 1x10
      labels_rotated = labels_rotated[0]
      labels_rotated[8] = classBox

      row_yoloFormat_rot = RowYoloFormat(img, labels_rotated[:9])

      pathSaveLabelsRotated = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_rot' + str(indexAngle) + '.txt'

      if indexLenList == 0:
         txtLabelsRotated = open(pathSaveLabelsRotated, "w") 
      else:
         txtLabelsRotated = open(pathSaveLabelsRotated, "a") 

      txtLabelsRotated.write(row_yoloFormat_rot) 
      txtLabelsRotated.write('\n')
      txtLabelsRotated.close()
      
  return 0

# -------------------------------------------------------------------------------

def AugmentationGammaCorrection(img, indexImg, indexLenList, labels_original, classBox,
                                min_gamma, max_gamma, step_gamma,
                                imagesStart_labelsStart_AuxPath):

  for indexGamma in range(min_gamma, max_gamma, step_gamma):

      rows, cols, ch = img.shape

      invGamma = 1 / (indexGamma/10)
      table = [((i / 255) ** invGamma) * 255 for i in range(256)]
      table = np.array(table, np.uint8)
      img_GammaCorrected = cv2.LUT(img, table)
            
      nameSavingImageGammaCorr = Path(indexImg).stem + '_gammaCorr' + str(indexGamma) + '.jpg'
      pathSaveImgGammaCorr = imagesStart_labelsStart_AuxPath + nameSavingImageGammaCorr
      cv2.imwrite(pathSaveImgGammaCorr,img_GammaCorrected)

      labels_gammaCorr = labels_original[:4, :2]
      labels_gammaCorr = np.reshape(labels_gammaCorr, (1, 8))
      labels_gammaCorr = list(labels_gammaCorr[0])
      labels_gammaCorr[8] = classBox
      
      row_yoloFormat_gammaCorr = RowYoloFormat(img, labels_gammaCorr)

      pathSaveLabelsGammaCorr = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_gammaCorr' + str(indexGamma) + '.txt'

      if indexLenList == 0:
         txtLabelsGammaCorr = open(pathSaveLabelsGammaCorr, "w") 
      else:
         txtLabelsGammaCorr = open(pathSaveLabelsGammaCorr, "a") 
         
      txtLabelsGammaCorr.write(row_yoloFormat_gammaCorr) 
      txtLabelsGammaCorr.write('\n')
      txtLabelsGammaCorr.close()
      
  return 0

# -------------------------------------------------------------------------------

def AugmentationBoxFilter(img, indexImg, indexLenList, labels_original, classBox,
                          min_boxFilter, max_boxFilter, step_boxFilter,
                          imagesStart_labelsStart_AuxPath):

      for indexBoxFilter in range(min_boxFilter, max_boxFilter, step_boxFilter):

          rows, cols, ch = img.shape

          img_boxFiltered = cv2.boxFilter(img, -1, (indexBoxFilter,indexBoxFilter))
                
          nameSavingImageBoxFilt = Path(indexImg).stem + '_BoxFilter' + str(indexBoxFilter) + '.jpg'
          pathSaveImgBoxFilt = imagesStart_labelsStart_AuxPath + nameSavingImageBoxFilt
          cv2.imwrite(pathSaveImgBoxFilt,img_boxFiltered)

          labels_boxFilt = labels_original[:4, :2]
          labels_boxFilt = np.reshape(labels_boxFilt, (1, 8))
          labels_boxFilt = list(labels_boxFilt[0])
          labels_boxFilt[8] = classBox
          
          row_yoloFormat_boxFilt = RowYoloFormat(img, labels_boxFilt)

          pathSaveLabelsBoxFilt = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_BoxFilter' + str(indexBoxFilter) + '.txt'

          if indexLenList == 0:
            txtLabelsBoxFilter = open(pathSaveLabelsBoxFilt, "w") 
          else:
            txtLabelsBoxFilter = open(pathSaveLabelsBoxFilt, "a") 

          txtLabelsBoxFilter.write(row_yoloFormat_boxFilt)
          txtLabelsBoxFilter.write('\n') 
          txtLabelsBoxFilter.close()
          
      return 0

# -----------------------------------------------------------------------------

def AugmentationRotationAndBoxFilter(img, indexImg, indexLenList, labels_original, classBox,
                                     min_angle, max_angle, step_angle,
                                     min_boxFilter, max_boxFilter, step_boxFilter,
                                     imagesStart_labelsStart_AuxPath):
  
    for indexBoxFilter in range(min_boxFilter, max_boxFilter, step_boxFilter):

        for indexAngle in range(min_angle, max_angle, step_angle):

          rows, cols, ch = img.shape

          img_boxFiltered = cv2.boxFilter(img, -1, (indexBoxFilter,indexBoxFilter))

          rotationMatrix = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), indexAngle, 1)
          img_rotated_boxed = cv2.warpAffine(img_boxFiltered, rotationMatrix, (cols, rows))
          nameSavingImageRotBox = Path(indexImg).stem + '_rot' + str(indexAngle) + \
          '_BoxFilter' + str(indexBoxFilter) + '.jpg'
          pathSaveImgRotBox = imagesStart_labelsStart_AuxPath + nameSavingImageRotBox
          cv2.imwrite(pathSaveImgRotBox,img_rotated_boxed)

          labels_rotated =  labels_original @ rotationMatrix.T
          labels_rotated = np.reshape(labels_rotated, (1, 10)) # 2x5 to 1x10
          labels_rotated = labels_rotated[0]
          labels_rotated[8] = classBox
          row_yoloFormat_rotBox = RowYoloFormat(img, labels_rotated[:9])

          pathSaveLabelsRotatedBoxed = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_rot' + str(indexAngle) + \
          '_BoxFilter' + str(indexBoxFilter) + '.txt'

          if indexLenList == 0:
             txtLabelsRotatedBoxed = open(pathSaveLabelsRotatedBoxed, "w") 
          else:
            txtLabelsRotatedBoxed = open(pathSaveLabelsRotatedBoxed, "a") 

          txtLabelsRotatedBoxed.write(row_yoloFormat_rotBox) 
          txtLabelsRotatedBoxed.write('\n')
          txtLabelsRotatedBoxed.close()

    return 0

# -------------------------------------------------------------------------------

def AugmentationRotationAndGammaCorrection(img, indexImg, indexLenList, labels_original, classBox,
                                           min_angle, max_angle, step_angle,
                                           min_gamma, max_gamma, step_gamma,
                                           imagesStart_labelsStart_AuxPath):
  
    for indexGamma in range(min_gamma, max_gamma, step_gamma,):

        for indexAngle in range(min_angle, max_angle, step_angle):

          rows, cols, ch = img.shape

          invGamma = 1 / (indexGamma/10)
          table = [((i / 255) ** invGamma) * 255 for i in range(256)]
          table = np.array(table, np.uint8)
          img_GammaCorrected = cv2.LUT(img, table)

          rotationMatrix = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), indexAngle, 1)
          img_rotated_gamma = cv2.warpAffine(img_GammaCorrected, rotationMatrix, (cols, rows))
          nameSavingImageRotGamma = Path(indexImg).stem + '_rot' + str(indexAngle) + \
          '_gammaCorr' + str(indexGamma) + '.jpg'
          pathSaveImgRotGamma = imagesStart_labelsStart_AuxPath + nameSavingImageRotGamma
          cv2.imwrite(pathSaveImgRotGamma, img_rotated_gamma)

          labels_rotated =  labels_original @ rotationMatrix.T
          labels_rotated = np.reshape(labels_rotated, (1, 10)) # 2x5 to 1x10
          labels_rotated = labels_rotated[0]
          labels_rotated[8] = classBox
          row_yoloFormat_rotGamma = RowYoloFormat(img, labels_rotated[:9])

          pathSaveLabelsRotatedGamma = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_rot' + str(indexAngle) + \
          '_gammaCorr' + str(indexGamma) + '.txt'

          if indexLenList == 0:
             txtLabelsRotatedGamma = open(pathSaveLabelsRotatedGamma, "w") 
          else:
            txtLabelsRotatedGamma = open(pathSaveLabelsRotatedGamma, "a") 

          txtLabelsRotatedGamma.write(row_yoloFormat_rotGamma)
          txtLabelsRotatedGamma.write('\n') 
          txtLabelsRotatedGamma.close()

    return 0

# -------------------------------------------------------------------------------

def AugmentationBoxFilterAndGammaCorrection(img, indexImg, indexLenList, labels_original, classBox,
                                            min_gamma, max_gamma, step_gamma,
                                            min_boxFilter, max_boxFilter, step_boxFilter,
                                            imagesStart_labelsStart_AuxPath):
  
    for indexGamma in range(min_gamma, max_gamma, step_gamma):

      for indexBoxFilter in range(min_boxFilter, max_boxFilter, step_boxFilter):

          rows, cols, ch = img.shape

          img_boxFiltered = cv2.boxFilter(img, -1, (indexBoxFilter,indexBoxFilter))

          invGamma = 1 / (indexGamma/10)
          table = [((i / 255) ** invGamma) * 255 for i in range(256)]
          table = np.array(table, np.uint8)
          img_GammaCorrectedBoxFiltered = cv2.LUT(img_boxFiltered, table)
                
          nameSavingImageBoxFiltGamma = Path(indexImg).stem + '_BoxFilter' + str(indexBoxFilter) + \
          '_gammaCorr' + str(indexGamma) + '.jpg'
          pathSaveImgBoxFiltGamma = imagesStart_labelsStart_AuxPath + nameSavingImageBoxFiltGamma
          cv2.imwrite(pathSaveImgBoxFiltGamma,img_GammaCorrectedBoxFiltered)

          labels_boxFiltGamma = labels_original[:4, :2]
          labels_boxFiltGamma = np.reshape(labels_boxFiltGamma, (1, 8))
          labels_boxFiltGamma = list(labels_boxFiltGamma[0])
          labels_boxFiltGamma[8] = classBox
          row_yoloFormat_boxFiltGamma = RowYoloFormat(img, labels_boxFiltGamma)

          pathSaveLabelsBoxFiltGamma = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_BoxFilter' + str(indexBoxFilter) +\
          '_gammaCorr' + str(indexGamma) + '.txt'

          if indexLenList == 0:
            txtLabelsBoxFilterGamma = open(pathSaveLabelsBoxFiltGamma, "w") 
          else:
            txtLabelsBoxFilterGamma = open(pathSaveLabelsBoxFiltGamma, "a") 

          txtLabelsBoxFilterGamma.write(row_yoloFormat_boxFiltGamma) 
          txtLabelsBoxFilterGamma.write('\n')
          txtLabelsBoxFilterGamma.close()
          
      return 0

def YoloLabelsListAndImagesSaving(DataCoordinates,
                                  max_angle, min_angle, step_angle,
                                  min_gamma, max_gamma, step_gamma,
                                  min_boxFilter, max_boxFilter, step_boxFilter,
                                  images_path, imagesStart_labelsStart_AuxPath):
  
  DataCoordinates.drop(columns=['image_width', 'image_height'], inplace=True)
  DataCoordinates = DataCoordinates.reindex(columns=['bbox_x','bbox_y','bbox_width','bbox_height','label_name','image_name'])
  number_images = 0

  for indexImg in images_path: # how many images

    remaining = len(images_path) - number_images
    print(f'Images still to be processed: {remaining}/{len(images_path)}')
    number_images = number_images + 1

    img = cv2.imread(indexImg)
    name_img = Path(indexImg).stem + '.jpg'

    nameSavingImage = Path(indexImg).stem + '_basic.jpg'
    pathSaveImg1 = imagesStart_labelsStart_AuxPath + nameSavingImage
    cv2.imwrite(pathSaveImg1,img)
    
    dataRows = ListCustomAux(DataCoordinates, name_img)

    listFinalBox = ListImageCoordBox(dataRows)

    pathSaveLabelsBasic = imagesStart_labelsStart_AuxPath + Path(indexImg).stem + '_basic.txt'
    txtLabelsBasic = open(pathSaveLabelsBasic, "w") 
    
    for indexLenList in range(0, len(listFinalBox), 1):  # how many boxs for picture

        listAux = listFinalBox[indexLenList]
        row_yoloFormat_basic = RowYoloFormat(img, listAux)

        if indexLenList > 0:
           txtLabelsBasic.write('\n')
        
        txtLabelsBasic.write(row_yoloFormat_basic)

        # MANDATORY: add "1" like the last element to maintain the correctness of the calculations with the rotation matrix
        labels_original = np.float32([[listAux[0], listAux[1], 1],
                                      [listAux[2], listAux[3], 1],
                                      [listAux[4], listAux[5], 1],
                                      [listAux[6], listAux[7], 1],
                                      [listAux[0], listAux[1], 1]]) # to visualize rectangle
        
        classBox = listAux[8]
        """
        AugmentationRotation(img, indexImg, indexLenList, labels_original, classBox,
                             min_angle, max_angle, step_angle,
                             imagesStart_labelsStart_AuxPath)
        
        AugmentationGammaCorrection(img, indexImg, indexLenList, labels_original, classBox,
                                    min_gamma, max_gamma, step_gamma,
                                    imagesStart_labelsStart_AuxPath)
        
        AugmentationBoxFilter(img, indexImg, indexLenList, labels_original, classBox,
                              min_boxFilter, max_boxFilter, step_boxFilter,
                              imagesStart_labelsStart_AuxPath)
        """
        AugmentationRotationAndBoxFilter(img, indexImg, indexLenList, labels_original, classBox,
                                     min_angle, max_angle, step_angle,
                                     min_boxFilter, max_boxFilter, step_boxFilter,
                                     imagesStart_labelsStart_AuxPath)  
        
        AugmentationRotationAndGammaCorrection(img, indexImg, indexLenList, labels_original, classBox,
                                           min_angle, max_angle, step_angle,
                                           min_gamma, max_gamma, step_gamma,
                                           imagesStart_labelsStart_AuxPath) 
        
        AugmentationBoxFilterAndGammaCorrection(img, indexImg, indexLenList, labels_original, classBox,
                                                min_gamma, max_gamma, step_gamma,
                                                min_boxFilter, max_boxFilter, step_boxFilter,
                                                imagesStart_labelsStart_AuxPath)
        
         
  return 0

def myTrainTestValSplit(ImagesAuxPath, indicesTrain, indicesVal, indicesTest, dirFinal):

    for indexIndTrain in range(0, len(indicesTrain), 1):
        img = cv2.imread(ImagesAuxPath[indexIndTrain])
        
        name_img = Path(ImagesAuxPath[indexIndTrain]).stem 
        pathImg_saveTrain = dirFinal + '/images/train/' + name_img + '.jpg'
        cv2.imwrite(pathImg_saveTrain, img)

        pathTxt_saveTrain = imagesStart_labelsStart_AuxPath + name_img + '.txt'
        dest_name_txt = dirFinal + '/labels/train/'
        shutil.copy2(pathTxt_saveTrain, dest_name_txt) # copy and paste

    for indexIndVal in range(0, len(indicesVal), 1):
        img = cv2.imread(ImagesAuxPath[indexIndVal])
        
        name_img = Path(ImagesAuxPath[indexIndVal]).stem 
        pathImg_saveVal = dirFinal + '/images/val/' + name_img + '.jpg'
        cv2.imwrite(pathImg_saveVal, img)

        pathTxt_saveVal = imagesStart_labelsStart_AuxPath + name_img + '.txt'
        dest_name_txt = dirFinal + '/labels/val/'
        shutil.copy2(pathTxt_saveVal, dest_name_txt) # copy and paste

    for indexIndTest in range(0, len(indicesTest), 1):
        img = cv2.imread(ImagesAuxPath[indexIndTest])
        
        name_img = Path(ImagesAuxPath[indexIndTest]).stem 
        pathImg_saveTest = dirFinal + '/images/test/' + name_img + '.jpg'
        cv2.imwrite(pathImg_saveTest, img)

        pathTxt_saveTest = imagesStart_labelsStart_AuxPath + name_img + '.txt'
        dest_name_txt = dirFinal + '/labels/test/'
        shutil.copy2(pathTxt_saveTest, dest_name_txt) # copy and paste
      
    return 0

# --------------------------------------------------------------------------------

def IndicesTrainTestVal(ImagesAuxPath, testSize, valSize):
    testSizeConfirmed = int(len(ImagesAuxPath) * testSize)
    valSizeConfirmed = int(len(ImagesAuxPath) * valSize)
    trainSizeConfirmed = len(ImagesAuxPath) - testSizeConfirmed - valSizeConfirmed

    StartIndices = np.arange(0, len(ImagesAuxPath))

    indicesTest = []

    while len(indicesTest) < testSizeConfirmed:
        index = np.random.randint(low = 0, high = len(ImagesAuxPath))
        if index not in indicesTest:
            indicesTest.append(index)

    AuxIndices = StartIndices

    indDel = []
    for indexAux in range(0, len(AuxIndices), 1):
        for indexTest in range(0, len(indicesTest), 1):
            if AuxIndices[indexAux] == indicesTest[indexTest]:
              indDel.append(indexAux) 
    AuxIndices = np.delete(AuxIndices, indDel, axis = 0)

    indicesVal = []
    while len(indicesVal) < valSizeConfirmed:
        index = random.choice(AuxIndices)
        if index not in indicesVal:
            indicesVal.append(index)

    indicesTrain = AuxIndices
    indDel = []
    for indexAux in range(0, len(indicesTrain), 1):
        for indexVal in range(0, len(indicesVal), 1):
            if indicesTrain[indexAux] == indicesVal[indexVal]:
              indDel.append(indexAux)
    indicesTrain = np.delete(indicesTrain, indDel, axis = 0)

    return indicesTrain.tolist(), indicesVal, indicesTest

# ------------------------------------------------------------------------------
def LabelsToDataframe(pathLabelsOut):

    p_ZY = []

    filesLabelsOut = glob.glob(pathLabelsOut)

    columns_names = ['name', 'class', 'distZ', 'distY']
    basic_list = [[0]*len(columns_names)]
    dataframeClassBoxes = pd.DataFrame(basic_list, columns=columns_names)
    row_number = 0

    for indexFiles in range(0, len(filesLabelsOut), 1):

        p_ZY_aux = []
        
        with open(filesLabelsOut[indexFiles]) as f:
             
             linesToRead = f.readlines()

             for indexReadLine in range(0, len(linesToRead), 1):
                
                 listOut = ConvertStrToList(linesToRead[indexReadLine])  # Class, CentreX, CentreY, Height, Width
                 classBox = float(listOut[0])
                 centreX_norm = float(listOut[1])
                 centreY_norm = float(listOut[2])
                 height_norm = float(listOut[3])
                 width_norm = float(listOut[4])

                 relative_distance = math.sqrt(2) - math.sqrt(pow(width_norm,2) + pow(height_norm,2))
                 centreY_norm2 = centreY_norm - 0.5
                 out = Path(filesLabelsOut[indexFiles]).stem, classBox, relative_distance, centreY_norm2
                
                 p_ZY_aux.extend(list(out))

                 dataframeClassBoxes.loc[row_number] = list(out)
                 row_number = row_number + 1

             p_ZY.append(p_ZY_aux) 

    return dataframeClassBoxes, p_ZY

def MAIN_PrepareData(ImagesAuxPath, testSize, valSize, dirFinal):
  indicesTrain, indicesVal, indicesTest = IndicesTrainTestVal(ImagesAuxPath, testSize, valSize)
  myTrainTestValSplit(ImagesAuxPath, indicesTrain, indicesVal, indicesTest, dirFinal)

  pathSel = glob.glob(dirFinal + '/images/*.jpg')
  
  return 'MAIN_PrepareData done'

# --------------------------------------------------------------------------------

def MAIN_ImagesLabelsForYolo(csvName, ImagesPath, imagesStart_labelsStart_AuxPath,
                             max_angle, min_angle, step_angle,
                             min_gamma, max_gamma, step_gamma,
                             min_boxFilter, max_boxFilter, step_boxFilter):

    DataCSV = pd.read_csv(csvName)

    images_path = glob.glob(ImagesPath)

    YoloLabelsListAndImagesSaving(DataCSV, max_angle + step_angle, min_angle, step_angle,
                                  min_gamma, max_gamma + min_gamma, step_gamma,
                                  min_boxFilter, max_boxFilter, step_boxFilter,
                                  images_path, imagesStart_labelsStart_AuxPath)
    
    pathSel = glob.glob(imagesStart_labelsStart_AuxPath + '/*.jpg')
    
    noticeNumberImages = 'Number of starting images: ' + str(len(images_path)) + '; Number of finishing images: ' + str(len(pathSel))
    return noticeNumberImages

"""**PATHS SETTING**"""

try:
    shutil.rmtree('/content/start')
    shutil.rmtree('/content/TRAINING_VAL_TEST_DATA')
except OSError as e:
    pass
    
os.makedirs('/content/start')
!unzip -q /content/input_data.zip -d /content/start/input_data

dirAux = '/content/start/aux'
os.makedirs(dirAux)

dirFinal = '/content/TRAINING_VAL_TEST_DATA'
os.makedirs(dirFinal)

os.makedirs(dirFinal + '/labels')
os.makedirs(dirFinal + '/images')

os.makedirs(dirFinal + '/labels/train')
os.makedirs(dirFinal + '/labels/val')
os.makedirs(dirFinal + '/labels/test')
os.makedirs(dirFinal + '/images/train')
os.makedirs(dirFinal + '/images/val')
os.makedirs(dirFinal + '/images/test')

imagesStart_path = '/content/start/input_data/*.jpg'
coordStart_path = '/content/coord.csv'
imagesStart_labelsStart_AuxPath = dirAux + '/'
ImagesAuxPath_Path = imagesStart_labelsStart_AuxPath + '*.jpg'
ImagesAuxPath = glob.glob(ImagesAuxPath_Path)

"""**IMAGES AND LABELS FOR YOLO CREATION**"""

MAIN_ImagesLabelsForYolo(coordStart_path, imagesStart_path, imagesStart_labelsStart_AuxPath,
                         max_angle, min_angle, step_angle,
                         min_gamma, max_gamma, step_gamma,
                         min_boxFilter, max_boxFilter, step_boxFilter)

"""**TRAIN TEST SPLIT**"""

ImagesAuxPath = glob.glob(ImagesAuxPath_Path)                # update of the path content
MAIN_PrepareData(ImagesAuxPath, testSize, valSize, dirFinal)
dataframeClassBoxes_true, p_ZY_true = LabelsToDataframe('/content/TRAINING_VAL_TEST_DATA/labels/test/*.txt')

"""**TRAIN with hyperparameters optimization - VALIDATION - TEST**"""

!python train.py --img 640 --batch batch_size --epochs number_of_epochs --data /content/custom_data.yaml --weights yolov5s.pt --cache --exist-ok

!python val.py --weights /content/yolov5/runs/train/exp/weights/best.pt --data /content/custom_data.yaml --img 640 --half --exist-ok

!python my_detect.py --weights /content/yolov5/runs/train/exp/weights/best.pt --img 640 --conf confidenceResults --source /content/TRAINING_VAL_TEST_DATA/images/test --save-txt --exist-ok

#!python val.py --weights /content/best.pt --data /content/custom_data.yaml --img 640 --half --exist-ok

!python my_detect.py --weights /content/best.pt --img 640 --conf 0.5 --source /content/TRAINING_VAL_TEST_DATA/images/test --save-txt --exist-ok

"""**PREDICTED LABELS TO DATAFRAME**"""

dataframeClassBoxes_pred, p_ZY_pred = LabelsToDataframe('/content/yolov5/runs/detect/exp/labels/*.txt')

"""**VISUALIZING MAPS**"""

def ImageDF(dataframeClassBoxes, name_fig):
    df_mask = dataframeClassBoxes['name'] == name_fig
    positions = np.flatnonzero(df_mask)
    image_df = dataframeClassBoxes.iloc[positions]

    return image_df

# ------------------------------------------------------------------------------

def CoordinatesForPlot(dataframeClassBoxes, name_fig, indexClass, image_df):

    df_mask = image_df['class'] == indexClass
    positions = np.flatnonzero(df_mask)
    class_df = image_df.iloc[positions]

    list_DistY = class_df['distY'].tolist()
    list_DistZ = class_df['distZ'].tolist()

    if len(list_DistY) > 0 and len(list_DistZ) > 0:
       if indexClass == 0:
          label_legend = 'unrec'
       else:
          label_legend = 'N°' + str(indexClass)
       return list_DistY, list_DistZ, label_legend
    else:
       return 0, 0, 'not'

PredLabels = glob.glob('/content/yolov5/runs/detect/exp/labels/*.txt')
TrueLabels = glob.glob('/content/TRAINING_VAL_TEST_DATA/labels/test/*.txt')

if len(PredLabels) != len(TrueLabels):
  print(len(TrueLabels)- len((PredLabels)), ' images without detection')

total_accuracy = 0

for indexNameImage in range(0, len(PredLabels), 1):

    pX_Rover, pY_Rover = 0, 0
    numberImagesOk = 0
    name_fig = Path(PredLabels[indexNameImage]).stem
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    image_df_pred = ImageDF(dataframeClassBoxes_pred, name_fig)
    ax1.set_title('Predictions'), ax1.set_xlim(-0.5, 0.5), ax1.set_ylim(-0.1, 1.5)
    ax1.scatter(pX_Rover, pY_Rover, label = 'rover')

    image_df_true = ImageDF(dataframeClassBoxes_true, name_fig)
    ax2.set_title('Reality'), ax2.set_xlim(-0.5, 0.5), ax2.set_ylim(-0.1, 1.5)
    ax2.scatter(pX_Rover, pY_Rover, label = 'rover')

    totNumberImages = 0

    for indexClass in range(0, 16, 1):  

        list_DistY_pred, list_DistZ_pred, label_legend_pred = CoordinatesForPlot(dataframeClassBoxes_pred, name_fig, indexClass, image_df_pred)
        list_DistY_true, list_DistZ_true, label_legend_true = CoordinatesForPlot(dataframeClassBoxes_true, name_fig, indexClass, image_df_true)
               
        if label_legend_pred != 'not' and label_legend_true != 'not':                
           ax1.scatter(list_DistY_pred, list_DistZ_pred, label = label_legend_pred)
           ax2.scatter(list_DistY_true, list_DistZ_true, label = label_legend_true)
 
           totNumberImages = totNumberImages + len(list_DistY_true)
           
           for indexComparison in range(0, len(list_DistY_true), 1): 
               
               if len(list_DistY_true) == len(list_DistY_pred) and \
                  list_DistY_true[indexComparison] - 0.1 < list_DistY_pred[indexComparison] < list_DistY_true[indexComparison] + 0.1 and \
                  list_DistZ_true[indexComparison] - 0.1 < list_DistZ_pred[indexComparison] < list_DistZ_true[indexComparison] + 0.1:
                  
                      numberImagesOk = numberImagesOk + 1
    
    ax1.legend(loc = 'best'), ax2.legend(loc = 'best')
    #ax2.label_outer() 
    accuracyLenTot = round(numberImagesOk/totNumberImages, 3) * 100    # imperfect accuracy: if number of box seen is equal to real number of box
    total_accuracy = total_accuracy + accuracyLenTot
    name_plt = name_fig + ', accuracy: ' + str(accuracyLenTot) + '%'
    fig.suptitle(name_plt)
    
    nameSavefig = '/content/ImgResult/' + name_fig + '.jpg'
    #plt.savefig(nameSavefig)

print('Total accuracy: ', round(total_accuracy/len(PredLabels),2), ' %')

"""**SAVE** **RESULTS**"""

#!zip -r /content/file.zip /content/yolov5/runs/detect/exp/labels
!zip -r /content/fileImgResult.zip /content/ImgResult