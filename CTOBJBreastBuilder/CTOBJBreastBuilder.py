import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import importlib
import SimpleITK as sitk
import sitkUtils
import numpy as np
import CTOBJBreastBuilderFunctions.PectoralSideModule as PectoralSideModule

import math
import random
import sys
import time
import qt
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtk.util import numpy_support

#
# CTOBJBreastBuilder
#

class CTOBJBreastBuilder(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CT-OBJ Breast Builder" # TODO make this more human readable by adding spaces
        self.parent.categories = ["NCTU"]
        self.parent.dependencies = []
        self.parent.contributors = ["NCTU Computer Graphics Laboratory"] # replace with "Firstname Lastname (Organization)"
        self.parent.helpText = ""
        #self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = "" # replace with organization, grant and thanks.

#
# CTOBJBreastBuilderWidget
#

class CTOBJBreastBuilderWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CTOBJBreastBuilderLogic()
        self.logic.initiate(self)
        # Instantiate and connect widgets ...

        # CT相關UI

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Create Chestwall"
        #parametersCollapsibleButton.setFont(qt.QFont("Times", 12))
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # input volume selector
        #
        self.inputCTSelector = slicer.qMRMLNodeComboBox()
        self.inputCTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputCTSelector.selectNodeUponCreation = True
        self.inputCTSelector.addEnabled = False
        self.inputCTSelector.removeEnabled = False
        self.inputCTSelector.noneEnabled = False
        self.inputCTSelector.showHidden = False
        self.inputCTSelector.showChildNodeTypes = False
        self.inputCTSelector.setMRMLScene( slicer.mrmlScene )
        self.inputCTSelector.setToolTip( "Pick the input to the algorithm." )
        parametersFormLayout.addRow("Input Volume: ", self.inputCTSelector)

        #
        # Pectoral Smoothing Iterations Spin Box
        #
        self.pectoralSmoothingIterationSpinBox = qt.QSpinBox()
        self.pectoralSmoothingIterationSpinBox.setRange(0, 20000)
        self.pectoralSmoothingIterationSpinBox.setSingleStep(500)
        self.pectoralSmoothingIterationSpinBox.setValue(4000)
        parametersFormLayout.addRow("Pectoral Smoothing Iterations: ", self.pectoralSmoothingIterationSpinBox)

        #
        # Estimate Volume Button
        #
        self.createChestWallButton = qt.QPushButton("Create Chestwall")
        self.createChestWallButton.toolTip = "Run the algorithm."
        self.createChestWallButton.enabled = False
        self.createChestWallButton.setFont(qt.QFont("Times", 15, qt.QFont.Black))
        parametersFormLayout.addRow(self.createChestWallButton)

        # OBJ相關UI
        
        # Default Parameters
        self.targetColor = qt.QColor("#4573a0")

        # Parameters Area
        parametersCollapsibleButton_2 = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton_2.text = "Create Breasts"
        self.layout.addWidget(parametersCollapsibleButton_2)
        # Layout within the dummy collapsible button
        parametersFormLayout_2 = qt.QFormLayout(parametersCollapsibleButton_2)

        # input model selector
        self.inputModelSelector = slicer.qMRMLNodeComboBox()
        self.inputModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.inputModelSelector.selectNodeUponCreation = True
        self.inputModelSelector.addEnabled = False
        self.inputModelSelector.removeEnabled = False
        self.inputModelSelector.noneEnabled = False
        self.inputModelSelector.showHidden = False
        self.inputModelSelector.showChildNodeTypes = False
        self.inputModelSelector.setMRMLScene( slicer.mrmlScene )
        self.inputModelSelector.setToolTip( "Model" )
        parametersFormLayout_2.addRow("Input OBJ Model: ", self.inputModelSelector)

        # input texture selector
        self.OBJTextureSelector = slicer.qMRMLNodeComboBox()
        self.OBJTextureSelector.nodeTypes = ["vtkMRMLVectorVolumeNode"]
        self.OBJTextureSelector.addEnabled = False
        self.OBJTextureSelector.removeEnabled = False
        self.OBJTextureSelector.noneEnabled = False
        self.OBJTextureSelector.showHidden = False
        self.OBJTextureSelector.showChildNodeTypes = False
        self.OBJTextureSelector.setMRMLScene(slicer.mrmlScene)
        self.OBJTextureSelector.setToolTip("Color image containing texture image.")
        parametersFormLayout_2.addRow("Texture: ", self.OBJTextureSelector)

        # inpute color selector
        self.colorButton = qt.QPushButton()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        parametersFormLayout_2.addRow("Marker Color:", self.colorButton)

        # Texture Button
        self.textureButton = qt.QPushButton("Apply Texture")
        self.textureButton.toolTip = "Paste the texture onto the model."
        self.textureButton.enabled = True
        parametersFormLayout_2.addRow(self.textureButton)

        # Apply Button
        self.preprocessButton = qt.QPushButton("Remove Tape")
        self.preprocessButton.toolTip = "Remove tapes from the model."
        self.preprocessButton.enabled = True
        parametersFormLayout_2.addRow(self.preprocessButton)

        # Select Breasts
        self.breastButton = qt.QPushButton("Finish Select Breasts")
        self.breastButton.toolTip = "Click after breasts are selected."
        self.breastButton.enabled = True
        parametersFormLayout_2.addRow(self.breastButton)
        parametersFormLayout_2.addRow(" ", None)

        #input segment editor
        self.inputSegmenationSelector = slicer.qMRMLNodeComboBox()
        self.inputSegmenationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.inputSegmenationSelector.selectNodeUponCreation = True
        self.inputSegmenationSelector.addEnabled = False
        self.inputSegmenationSelector.removeEnabled = False
        self.inputSegmenationSelector.noneEnabled = False
        self.inputSegmenationSelector.showHidden = False
        self.inputSegmenationSelector.showChildNodeTypes = False
        self.inputSegmenationSelector.setMRMLScene( slicer.mrmlScene )
        self.inputSegmenationSelector.setToolTip( "Segmentation" )
        parametersFormLayout_2.addRow("Input Segmenation: ", self.inputSegmenationSelector)

        # Setup Button
        self.setupButton = qt.QPushButton("Setup")
        self.setupButton.toolTip = "Change segmentation lookout"
        self.setupButton.enabled = True
        parametersFormLayout_2.addRow(self.setupButton)

        # Transform Button
        self.transformButton = qt.QPushButton("Transform")
        self.transformButton.toolTip = "Transform"
        self.transformButton.enabled = True
        parametersFormLayout_2.addRow(self.transformButton)

        # Model to Segment Button
        self.exportButton = qt.QPushButton("Export")
        self.exportButton.toolTip = "Export input model to segment"
        parametersFormLayout_2.addRow(self.exportButton)

        # Testing
        self.breastvolumeButton = qt.QPushButton("Closed Breast")
        self.breastvolumeButton.toolTip = ""
        parametersFormLayout_2.addRow(self.breastvolumeButton)
        parametersFormLayout_2.addRow(" ", None)
        
        #
        # Resampling Widget
        #
        self.segmentSelector = slicer.qMRMLNodeComboBox()
        self.segmentSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentSelector.selectNodeUponCreation = True
        self.segmentSelector.addEnabled = False
        self.segmentSelector.removeEnabled = False
        self.segmentSelector.noneEnabled = False
        self.segmentSelector.showHidden = False
        self.segmentSelector.showChildNodeTypes = False
        self.segmentSelector.setMRMLScene( slicer.mrmlScene )
        self.segmentSelector.setToolTip( "Pick the input to the algorithm." )
        parametersFormLayout_2.addRow("Segmentation : ", self.segmentSelector)
        
        self.oversamplingFactorSpinBox = qt.QDoubleSpinBox()
        self.oversamplingFactorSpinBox.setRange(0.05, 2.0)
        self.oversamplingFactorSpinBox.setSingleStep(0.05)
        self.oversamplingFactorSpinBox.setValue(0.5)

        self.segmentationGeometryButton = qt.QPushButton("Change Sampling")
        self.segmentationGeometryButton.toolTip = "Press to change sampling factor."
        self.segmentationGeometryButton.enabled = True
        self.layout.addWidget(self.segmentationGeometryButton)

        container = qt.QHBoxLayout()
        container.addWidget(self.oversamplingFactorSpinBox)
        container.addWidget(self.segmentationGeometryButton)
        parametersFormLayout_2.addRow("Over Sampling Factor : ", container)

        #
        # Editting Segmentation
        #
        segmentationEditorCollapsibleButton = ctk.ctkCollapsibleButton()
        segmentationEditorCollapsibleButton.text = "Editting Segmentation"
        segmentationEditorCollapsibleButton.setFont(qt.QFont("Times", 12))
        segmentationEditorCollapsibleButton.collapsed = True
        self.layout.addWidget(segmentationEditorCollapsibleButton)

        # Layout within the dummy collapsible button
        segmentationEditorFormLayout = qt.QFormLayout(segmentationEditorCollapsibleButton)

        self.segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.segmentationEditorWidget.setMaximumNumberOfUndoStates(10)
        self.parameterSetNode = None
        self.selectParameterNode()
        self.segmentationEditorWidget.setSwitchToSegmentationsButtonVisible(False)
        self.segmentationEditorWidget.setUndoEnabled(True)
        self.segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentationEditorFormLayout.addWidget(self.segmentationEditorWidget)
        
        #
        # Calculate Statistics Button
        #
        self.statButton = qt.QPushButton("Calculate Statistics")
        self.statButton.toolTip = "Calculate statistics."
        self.statButton.enabled = True
        self.statButton.setFont(qt.QFont("Times", 24, qt.QFont.Black))
        segmentationEditorFormLayout.addWidget(self.statButton)


        # connections
        # 防呆
        self.inputCTSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeSelect)
        self.inputModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOBJInputDataSelect)
        self.OBJTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOBJInputDataSelect)
        
        # 按鈕行為
        self.createChestWallButton.connect('clicked(bool)', self.onEstimateButton)
        self.colorButton.connect("clicked(bool)", self.onSelectColor)
        self.textureButton.connect("clicked(bool)", self.onTextureButton)
        self.preprocessButton.connect("clicked(bool)", self.onPreprocessButton)
        self.breastButton.connect("clicked(bool)", self.onBreastButton)
        self.setupButton.connect("clicked(bool)", self.onSetupButton)
        self.transformButton.connect("clicked(bool)", self.onTransformButton)
        self.exportButton.connect("clicked(bool)", self.onExportButton)
        self.breastvolumeButton.connect("clicked(bool)", self.onBreastvolumeButton)
        self.statButton.connect('clicked(bool)', self.calculateStatistics)

        # Refresh Apply button state
        self.onInputVolumeSelect()
        self.onOBJInputDataSelect()

    def onSelectColor(self):
        self.targetColor = qt.QColorDialog.getColor()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        self.colorButton.update()

    def onTextureButton(self):
        self.logic.showTextureOnModel(self.inputModelSelector.currentNode(), self.OBJTextureSelector.currentNode())

    def onPreprocessButton(self):
        self.logic.startPreprocessing(self.inputModelSelector.currentNode(), self.OBJTextureSelector.currentNode(), self.targetColor)

    def onBreastButton(self):
        self.logic.truncateBreastPolyData()
        
    def finishPreProcessing(self):
        self.breastButton.enabled = True
        self.logic.setupFiducialNodeOperation()

    def onSetupButton(self):
        # setting current segment node    
        
        current = slicer.util.getNode("Segmentation")
        self.inputSegmenationSelector.setCurrentNode(current)
        self.inputSegmenationSelector.setMRMLScene(slicer.mrmlScene)
        self.logic.transformSetup(self.inputSegmenationSelector.currentNode())

    def onTransformButton(self):
        self.logic.performTransform(self.inputModelSelector.currentNode(), self.inputSegmenationSelector.currentNode())

    def onExportButton(self):
        self.logic.changeType()

    def onBreastvolumeButton(self):
        self.logic.Breastvolume()

    def setPoint(self):
        self.markupPointWidget.setCurrentNode(self.pointSelector.currentNode())

    def cleanup(self):
        pass

    def onInputVolumeSelect(self):
        self.createChestWallButton.enabled = self.inputCTSelector.currentNode()
    
    def onOBJInputDataSelect(self):
        self.textureButton.enabled = self.inputModelSelector.currentNode() and self.OBJTextureSelector.currentNode()

    def onEstimateButton(self):
        logic = CTOBJBreastBuilderLogic()
        logic.createChestWall(self.inputCTSelector.currentNode(), self.pectoralSmoothingIterationSpinBox.value)
    
    def onSegmentationGeometryButton(self):
        segmentationNode = self.segmentSelector.currentNode()

        #Create desired geometryImageData with overSamplingFactor
        segmentationGeometryLogic = slicer.vtkSlicerSegmentationGeometryLogic()
        segmentationGeometryLogic.SetInputSegmentationNode(segmentationNode)
        segmentationGeometryLogic.SetSourceGeometryNode(segmentationNode)
        segmentationGeometryLogic.SetOversamplingFactor(self.oversamplingFactorSpinBox.value)
        segmentationGeometryLogic.CalculateOutputGeometry()
        geometryImageData = segmentationGeometryLogic.GetOutputGeometryImageData()

        segmentIDs = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)

        for index in range(segmentIDs.GetNumberOfValues()):
            currentSegmentID = segmentIDs.GetValue(index)
            currentSegment = segmentationNode.GetSegmentation().GetSegment(currentSegmentID)

            currentLabelmap = currentSegment.GetRepresentation("Binary labelmap")

            success = slicer.vtkOrientedImageDataResample.ResampleOrientedImageToReferenceOrientedImage(currentLabelmap, geometryImageData, currentLabelmap, False, True)

            if not success:
                print("Segment {}/{} failed to be resampled".format(segmentationNode.GetName(), currentSegmentID))

        segmentationNode.Modified()

        print("Finish Resolution Changing")

    def onReload(self):
        importlib.reload(PectoralSideModule)
        ScriptedLoadableModuleWidget.onReload(self)

    def calculateStatistics(self):
        from SegmentStatistics import SegmentStatisticsLogic
        segStatLogic = SegmentStatisticsLogic()

        segStatLogic.getParameterNode().SetParameter("Segmentation", self.segmentationEditorWidget.segmentationNodeID())
        self.segmentationEditorWidget.segmentationNode().CreateDefaultDisplayNodes()
        segStatLogic.computeStatistics()

        resultsTableNode = slicer.vtkMRMLTableNode()
        slicer.mrmlScene.AddNode(resultsTableNode)
        segStatLogic.exportToTable(resultsTableNode)
        segStatLogic.showTable(resultsTableNode)

    def selectParameterNode(self):
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if self.parameterSetNode == segmentEditorNode:
            # nothing changed
            return
        self.parameterSetNode = segmentEditorNode
        self.segmentationEditorWidget.setMRMLSegmentEditorNode(self.parameterSetNode)

    def saveState(self):
        self.segmentationEditorWidget.saveStateForUndo()

#
# CTOBJBreastBuilderLogic
#

class CTOBJBreastBuilderLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def initiate(self, widget):
        self.widget = widget
        self.createdModelNodes = []
        self.markerName = "Reference_Breast_Position"
        self.mainModelNode = None
        self.chestWallName = "Poggers"
        self.numofbreast = 0

    def HasImageData(self,volumeNode):
        """This is an example logic method that
        returns true if the passed in volume
        node has valid image data
        """
        if not volumeNode:
            logging.debug('HasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('HasImageData failed: no image data in volume node')
            return False
        return True

    # Actual algorithm
    def createChestWall(self, inputVolume, pectoralSmoothingIterations=4000):

        logging.info('Processing started')
        inputImage = sitkUtils.PullVolumeFromSlicer(inputVolume) #對應:sitkUtils.PushVolumeFromSlicer
        direction = inputImage.GetDirection()
        inputImage = sitk.Flip(inputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0]) # 根據旋轉矩陣確保所有病人的座標方向一致

        #把一部分不需要的資料消除
        originImage = inputImage
        truncatedImage = self.TruncateUnecessaryPart(inputImage)

        result = PectoralSideModule.EvaluatePectoralSide(truncatedImage, pectoralSmoothingIterations)

        vtk_result = self.sitkImageToVtkOrientedImage(result)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName('test_seg')
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtk_result, "Poggers")


    def TruncateUnecessaryPart(self, image):
        imageSize = image.GetSize()
        truncated = image[:, :, imageSize[2] // 5:]

        return truncated

    def sitkImageToVtkOrientedImage(self, img):
        imgNode = sitkUtils.PushVolumeToSlicer(img)
        vtkImage = imgNode.GetImageData()

        vtkOrientedImage = slicer.vtkOrientedImageData()
        vtkOrientedImage.DeepCopy(vtkImage)
        dir = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        imgNode.GetIJKToRASDirections(dir)
        vtkOrientedImage.SetDirections([dir[0], dir[1], dir[2]])
        vtkOrientedImage.SetOrigin(imgNode.GetOrigin())
        vtkOrientedImage.SetSpacing(imgNode.GetSpacing())

        slicer.mrmlScene.RemoveNode(imgNode)
        return vtkOrientedImage

    ###以下是transform部分###
    def transformSetup(self, segmentationNode):

        segmentationDisplayNode = segmentationNode.GetDisplayNode()
        segmentation = segmentationNode.GetSegmentation()

        segmentId1 = segmentation.GetSegmentIdBySegmentName("Bone")
        segmentationDisplayNode.SetSegmentOpacity3D(segmentId1, 1.0)
        segmentation.GetSegment(segmentId1).SetColor(0.0,0.8,0.0)

        segmentId2 = segmentation.GetSegmentIdBySegmentName("Skin")
        segmentationDisplayNode.SetSegmentOpacity3D(segmentId2, 0.5)
        segmentation.GetSegment(segmentId2).SetColor(1.0,1.0,1.0)

        # volume to seg
        # 這段不需要，因為已經有test_seg
        #self.chestWallName = labelmapVolumeNode.GetName()
        #seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        #slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, seg)
        #seg.CreateClosedSurfaceRepresentation()

        # 把在test_seg的胸壁加入到Segmentation
        chestWallNode = slicer.util.getNode("test_seg")
        segmentation_s = chestWallNode.GetSegmentation()

        sourceSegmentId = segmentation_s.GetSegmentIdBySegmentName(self.chestWallName)
        segmentation.CopySegmentFromSegmentation(segmentation_s, sourceSegmentId)

        slicer.mrmlScene.RemoveNode(chestWallNode)

        # change fiducial node name and set to blue
        fiducialNode = slicer.util.getNode("Fiducials")

        ### 修改
        color = fiducialNode.GetDisplayNode().GetSelectedColor()
        if color==(1.0,0.0,0.0):
            #代表選到segnode(想要是1)
            fiducialNode.SetName("seg_fid")
            next_fiducialNode = slicer.util.getNode("Fiducials")
            next_fiducialNode.SetName("model_fid")
            next_fiducialNode.GetDisplayNode().SetSelectedColor([0.0,0.0,1.0])
        else:
            #代表選到modelnode(想要是2)
            fiducialNode.SetName("model_fid")
            fiducialNode.GetDisplayNode().SetSelectedColor([0.0,0.0,1.0])
            next_fiducialNode = slicer.util.getNode("Fiducials")
            next_fiducialNode.SetName("seg_fid")

        return True
    
    def performTransform(self, modelNode, segNode):

        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        transformNode.SetName("Registration Transform")
        parameters = {}
    
        f1 = slicer.util.getNode("seg_fid")
        f1.GetMarkupsDisplayNode().SetTextScale(0.0)
        f2 = slicer.util.getNode("model_fid")
        f2.GetMarkupsDisplayNode().SetTextScale(0.0)

        parameters["saveTransform"] = transformNode.GetID()
        parameters["movingLandmarks"] = f2.GetID()
        parameters["fixedLandmarks"] = f1.GetID()
        fiduciaReg = slicer.modules.fiducialregistration
        slicer.cli.runSync(fiduciaReg, None, parameters)

        tmatrix = vtk.vtkMatrix4x4()
        saveTransformNode = slicer.util.getNode(parameters["saveTransform"])
        saveTransformNode.GetMatrixTransformToParent(tmatrix)
        #transformMatrix = slicer.util.arrayFromVTKMatrix(tmatrix)

        f2.SetAndObserveTransformNodeID(transformNode.GetID())

        color = f1.GetDisplayNode().GetSelectedColor()
        if color==(1.0,0.0,0.0):
            print("color=red")
            modelNode.SetAndObserveTransformNodeID(transformNode.GetID())
            print("transform:",self.numofbreast)
            for i in range(self.numofbreast):
                breast = slicer.util.getNode("MergedPolyData_"+str(i))
                breast.SetAndObserveTransformNodeID(transformNode.GetID())
        else:
            # 移動segmentation比較方便
            print("color=",color)
            segNode.ApplyTransformMatrix(tmatrix)
    

    def changeType(self):    
        ### export model to seg ###
        print("export:",self.numofbreast)
        for i in range(self.numofbreast):
            modelNode = slicer.util.getNode("MergedPolyData_"+str(i))
            seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "MergedPolyData_"+str(i)+"-segmentation")
            slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, seg)
            seg.CreateBinaryLabelmapRepresentation()
        
        return

    def Breastvolume(self):
        ### 移動test_seg到胸部的segmentation
        chestWallNode = slicer.util.getNode("Segmentation")
        segmentation_s = chestWallNode.GetSegmentation()
        sourceSegmentId = segmentation_s.GetSegmentIdBySegmentName(self.chestWallName)

        for n in range(self.numofbreast):
            mergedbreastNode = slicer.util.getNode("MergedPolyData_"+str(n)+"-segmentation")
            segmentation_d = mergedbreastNode.GetSegmentation()
            segmentation_d.CopySegmentFromSegmentation(segmentation_s, sourceSegmentId)
            
            ### 為segmentation增加一個相對應的labelmapvolume
            #segmentationNode = slicer.util.getNode("MergedPolyData_"+str(i)+"-segmentation")
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            labelmapVolumeNode.SetName("labelmapvolume_"+str(n))
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(mergedbreastNode, labelmapVolumeNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)

            # 換data type to scalar volume
            outputvolumenode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            outputvolumenode.SetName("scalarvolume_"+str(n))
            sef = slicer.modules.volumes.logic().CreateScalarVolumeFromVolume(slicer.mrmlScene, outputvolumenode, labelmapVolumeNode)

            image = sitkUtils.PullVolumeFromSlicer(outputvolumenode)
            resVolume = sitk.Cast(image, sitk.sitkInt16)

            # (281黃, 206綠, 268紅)
            image_shape = image.GetSize()
            for i in range(int(image_shape[2]*2/3)):
                find_img = image[0:image_shape[0], 0:image_shape[1], i]
                # 2d array.shape=(1024,1024)
                # 總共會有三個值0,1,2
                # 測試方法 array[array==1]
                array = sitk.GetArrayFromImage(find_img)
                tmp = array[array==1]
                if(tmp.shape==0):
                    continue

                for j in range(image_shape[0]):
                    check = False
                    for k in range(int(image_shape[1]//3)):
                        # 先確定碰到點
                        if check==False and array[k][j]==1:
                            check = True
                        # 再往後長  
                        elif check==True and array[k][j]==0:
                            array[k][j]=1

        
                res = sitk.GetImageFromArray(array)
                resVolume = sitk.Paste(
                            destinationImage = resVolume,
                            sourceImage = sitk.JoinSeries(res),
                            sourceSize = [image_shape[0]-1, image_shape[1]-1, 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, i])
            
            vtkt = self.sitkImageToVtkOrientedImage(resVolume)
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            if n==0:
                segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkt, "closed_breast_"+str(n), [1.0, 1.0, 0.0])
            else:
                segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkt, "closed_breast_"+str(n), [0.0, 0.0, 1.0])

            # show in 3d
            segmentationNode.CreateClosedSurfaceRepresentation()

        return
    

    ###以下是膠帶部分###
    def showTextureOnModel(self, modelNode, textureImageNode):
        modelDisplayNode = modelNode.GetDisplayNode()
        modelDisplayNode.SetBackfaceCulling(0)
        textureImageFlipVert = vtk.vtkImageFlip()
        textureImageFlipVert.SetFilteredAxis(1)
        textureImageFlipVert.SetInputConnection(textureImageNode.GetImageDataConnection())
        modelDisplayNode.SetTextureImageDataConnection(textureImageFlipVert.GetOutputPort())

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)

        self.createdModelNodes.append(modelNode)

        return modelNode

    def convertTextureToPointAttribute(self, modelNode, textureImageNode):
        polyData = modelNode.GetPolyData()
        textureImageFlipVert = vtk.vtkImageFlip()
        textureImageFlipVert.SetFilteredAxis(1)
        textureImageFlipVert.SetInputConnection(
            textureImageNode.GetImageDataConnection())
        textureImageFlipVert.Update()
        textureImageData = textureImageFlipVert.GetOutput()
        pointData = polyData.GetPointData()
        tcoords = pointData.GetTCoords()
        numOfPoints = pointData.GetNumberOfTuples()
        assert numOfPoints == tcoords.GetNumberOfTuples(), "Number of texture coordinates does not equal number of points"
        textureSamplingPointsUv = vtk.vtkPoints()
        textureSamplingPointsUv.SetNumberOfPoints(numOfPoints)
        for pointIndex in range(numOfPoints):
            uv = tcoords.GetTuple2(pointIndex)
            textureSamplingPointsUv.SetPoint(pointIndex, uv[0], uv[1], 0)

        textureSamplingPointDataUv = vtk.vtkPolyData()
        uvToXyz = vtk.vtkTransform()
        textureImageDataSpacingSpacing = textureImageData.GetSpacing()
        textureImageDataSpacingOrigin = textureImageData.GetOrigin()
        textureImageDataSpacingDimensions = textureImageData.GetDimensions()
        uvToXyz.Scale(textureImageDataSpacingDimensions[0] / textureImageDataSpacingSpacing[0],
                    textureImageDataSpacingDimensions[1] / textureImageDataSpacingSpacing[1], 1)
        uvToXyz.Translate(textureImageDataSpacingOrigin)
        textureSamplingPointDataUv.SetPoints(textureSamplingPointsUv)
        transformPolyDataToXyz = vtk.vtkTransformPolyDataFilter()
        transformPolyDataToXyz.SetInputData(textureSamplingPointDataUv)
        transformPolyDataToXyz.SetTransform(uvToXyz)
        probeFilter = vtk.vtkProbeFilter()
        probeFilter.SetInputConnection(transformPolyDataToXyz.GetOutputPort())
        probeFilter.SetSourceData(textureImageData)
        probeFilter.Update()
        rgbPoints = probeFilter.GetOutput().GetPointData().GetArray('ImageScalars')

        colorArray = vtk.vtkDoubleArray()
        colorArray.SetName('Color')
        colorArray.SetNumberOfComponents(3)
        colorArray.SetNumberOfTuples(numOfPoints)
        for pointIndex in range(numOfPoints):
            rgb = rgbPoints.GetTuple3(pointIndex)
            colorArray.SetTuple3(
                pointIndex, rgb[0]/255., rgb[1]/255., rgb[2]/255.)
            colorArray.Modified()
            pointData.AddArray(colorArray)

        pointData.Modified()
        polyData.Modified()

    def extractSelection(self, modelNode, targetColor, threshold):
        colorData = vtk_to_numpy(
            modelNode.GetPolyData().GetPointData().GetArray("Color"))
        colorData = np.sum(np.abs(colorData - targetColor), axis=1) / 3

        return np.asarray(np.where(colorData < threshold))[0]

    def reduceAndCleanPolyData(self, modelNode):
        # triangulate
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(modelNode.GetPolyData())
        triangleFilter.Update()

        # decimate
        decimateFilter = vtk.vtkDecimatePro()
        decimateFilter.SetInputConnection(triangleFilter.GetOutputPort())
        decimateFilter.SetTargetReduction(0.33)
        decimateFilter.PreserveTopologyOn()
        decimateFilter.BoundaryVertexDeletionOff()
        decimateFilter.Update()

        # clean
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(decimateFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        # relax
        relaxFilter = vtk.vtkWindowedSincPolyDataFilter()
        relaxFilter.SetInputConnection(cleanFilter.GetOutputPort())
        relaxFilter.SetNumberOfIterations(10)
        relaxFilter.BoundarySmoothingOn()
        relaxFilter.FeatureEdgeSmoothingOn()
        relaxFilter.SetFeatureAngle(120.0)
        relaxFilter.SetPassBand(0.001)
        relaxFilter.NonManifoldSmoothingOn()
        relaxFilter.NormalizeCoordinatesOn()
        relaxFilter.Update()

        # normal
        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(relaxFilter.GetOutputPort())
        normalFilter.ComputePointNormalsOn()
        normalFilter.SplittingOff()
        normalFilter.Update()

        '''
        # alignCenter
        polyData = normalFilter.GetOutput()
        points_array = vtk_to_numpy(polyData.GetPoints().GetData())
        center = points_array.sum(axis=0) / points_array.shape[0]
        np.copyto(points_array, points_array - center)
        polyData.GetPoints().GetData().Modified()
        '''
        polyData = normalFilter.GetOutput()
        modelNode.SetAndObservePolyData(polyData)

    def setupFiducialNodeOperation(self):
        # Create fiducial node
        fiducialNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLMarkupsFiducialNode())
        fiducialNode.SetName(self.markerName)

        placeModePersistence = 1
        slicer.modules.markups.logic().StartPlaceMode(placeModePersistence)

    def deletePoint(self, polyData, delPointIds):
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_vtk(delPointIds))
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, polyData)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(extractSelection.GetOutput())
        geometryFilter.Update()

        return geometryFilter.GetOutput()
    
    def startPreprocessing(self, modelNode, textureImageNode, targetColor):
        """
        Run the actual algorithm
        """
        print("----Start Processing----")
        startTime = time.time()
        print("Start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)) + "\n")

        self.mainModelNode = modelNode

        newPolyData = vtk.vtkPolyData()
        newPolyData.DeepCopy(modelNode.GetPolyData())
        self.modifidedModelNode = self.createNewModelNode(newPolyData, "Modified_Model")

        newModelNode = self.modifidedModelNode

        
        # 轉換顏色格式:QColor -> np.array
        targetColor = np.array([targetColor.redF(), targetColor.greenF(), targetColor.blueF()])
        print("Selected Color: {}".format(targetColor))

        # 取得vtkMRMLModelNode讀取的檔案
        fileName = modelNode.GetStorageNode().GetFileName()
        print("OBJ File Path: {}\n".format(fileName))

        print("Origin Model points: {}".format(
            self.modifidedModelNode.GetPolyData().GetNumberOfPoints()))
        print("Origin Model cells: {}\n".format(
            self.modifidedModelNode.GetPolyData().GetNumberOfCells()))

        # 產生點的顏色資料
        self.convertTextureToPointAttribute(newModelNode, textureImageNode)
        
        # 取出顏色於範圍內的點id
        delPointIds = self.extractSelection(newModelNode, targetColor, 0.18)

        # 刪除顏色符合的點
        newModelNode.SetAndObservePolyData(self.deletePoint(newModelNode.GetPolyData(), delPointIds))

        
        # 處理PolyData (降低面數、破洞處理......)
        self.reduceAndCleanPolyData(newModelNode)

        print("Modified Model points: {}".format(newModelNode.GetPolyData().GetNumberOfPoints()))
        print("Modified Model cells: {}\n".format(newModelNode.GetPolyData().GetNumberOfCells()))

        modelNode.GetDisplayNode().VisibilityOff()
        newModelNode.GetDisplayNode().VisibilityOn()

        self.widget.finishPreProcessing()
        
        print("\n----Complete Processing----")
        stopTime = time.time()
        print("Complete time: " +
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stopTime)))
        logging.info('Processing completed in {0:.2f} seconds\n'.format(
            stopTime - startTime))

        return True

    def truncateBreastPolyData(self):
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SwitchToViewTransformMode()
        # also turn off place mode persistence if required
        interactionNode.SetPlaceModePersistence(0)

        fiducialNode = slicer.util.getNode(self.markerName)
        numFids = fiducialNode.GetNumberOfFiducials()
        self.numofbreast = numFids
        print("numofbreast:",self.numofbreast)
        last_point = 1
        if numFids>=2:
            last_point = 2

        breastPos = []
        # Use the last 2 fiducials as two breast positions
        for i in range(numFids - last_point, numFids):
            ras = [0, 0, 0]
            fiducialNode.GetNthFiducialPosition(i, ras)
            breastPos.append(ras)
            # the world position is the RAS position with any transform matrices applied
            world = [0, 0, 0, 0]
            fiducialNode.GetNthFiducialWorldCoordinates(i, world)
            #print(i, ": RAS =", ras, ", world =", world)

        slicer.mrmlScene.RemoveNode(fiducialNode)

        wallGenerator = BreastWallGenerator()

        for i in range(self.numofbreast):
            # 尋找最接近選取點的mesh
            connectFilter = vtk.vtkPolyDataConnectivityFilter()
            connectFilter.SetInputData(self.modifidedModelNode.GetPolyData())
            connectFilter.SetExtractionModeToClosestPointRegion()
            connectFilter.SetClosestPoint(breastPos[i])
            connectFilter.Update()

            rawBreastPolyData = connectFilter.GetOutput()
            self.createNewModelNode(connectFilter.GetOutput(), "Breast_{}".format(i))

            mergedPolyData, edgePolydata, wallMesh = wallGenerator.generateWall(rawBreastPolyData, True)
            self.createNewModelNode(mergedPolyData, "MergedPolyData_" + str(i))
            self.createNewModelNode(edgePolydata, "edgePolydata_" + str(i))
            self.createNewModelNode(wallMesh, "wallMesh_" + str(i))
    
class BreastWallGenerator():
    def generateWall(self, breastPolyData, ripEdge):
        refinedBreastPolyData = self.refineBreastPolyData(breastPolyData, 50)

        if ripEdge:
            rippedBreastPolyData = refinedBreastPolyData
            for _ in range(3):  # 藉由直接移除n層boundary減少突出邊緣
                _, edgeIds = self.extractBoundaryPoints(rippedBreastPolyData)
                rippedBreastPolyData = self.deletePoint(rippedBreastPolyData, edgeIds)

            smoothedBreastPolyData = self.smoothBoundary(rippedBreastPolyData, 2)
        else:
            smoothedBreastPolyData = refinedBreastPolyData

        
        # 取得平滑後的邊緣
        edgePolydata, _ = self.extractBoundaryPoints(smoothedBreastPolyData)
        
        wallMesh = self.createWallMesh(edgePolydata)

        mergedPolyData = self.mergeBreastAndBoundary(smoothedBreastPolyData, wallMesh)
        
        return smoothedBreastPolyData, edgePolydata, wallMesh

    def refineBreastPolyData(self, polyData, holeSize):
        holeFiller = vtk.vtkFillHolesFilter()
        holeFiller.SetInputData(polyData)
        holeFiller.SetHoleSize(holeSize)
        holeFiller.Update()

        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(holeFiller.GetOutputPort())
        normalFilter.ComputePointNormalsOn()
        normalFilter.SplittingOff()
        normalFilter.Update()

        return normalFilter.GetOutput()

    def smoothBoundary(self, polyData, edgeWidth):
        nonEdgePolyData = polyData
        for _ in range(edgeWidth):  # 邊緣平滑次數
            _, edgeIds = self.extractBoundaryPoints(nonEdgePolyData)
            nonEdgePolyData = self.deletePoint(nonEdgePolyData, edgeIds)

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(polyData)
        smoothFilter.SetNumberOfIterations(50)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.SetEdgeAngle(180)
        smoothFilter.SetRelaxationFactor(1)
        smoothFilter.SetSourceData(nonEdgePolyData)
        smoothFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(smoothFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        return cleanFilter.GetOutput()

    def extractBoundaryPoints(self, polyData, edgeName=""):
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(polyData)
        idFilter.SetIdsArrayName("ids")
        idFilter.PointIdsOn()
        idFilter.CellIdsOff()
        idFilter.Update()

        edgeFilter = vtk.vtkFeatureEdges()
        edgeFilter.SetInputConnection(idFilter.GetOutputPort())
        edgeFilter.BoundaryEdgesOn()
        edgeFilter.FeatureEdgesOff()
        edgeFilter.ManifoldEdgesOff()
        edgeFilter.NonManifoldEdgesOff()
        edgeFilter.Update()

        if edgeName != "":
            self.createNewModelNode(edgeFilter.GetOutput(), edgeName)

        return edgeFilter.GetOutput(), vtk_to_numpy(edgeFilter.GetOutput().GetPointData().GetArray("ids"))
        
    def deletePoint(self, polyData, delPointIds):
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(numpy_to_vtk(delPointIds))
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, polyData)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(extractSelection.GetOutput())
        geometryFilter.Update()

        return geometryFilter.GetOutput()

    def createWallMesh(self, edgePolyData):
        bounds = edgePolyData.GetBounds()
        Point_coordinates = edgePolyData.GetPoints().GetData()
        numpy_coordinates = numpy_support.vtk_to_numpy(Point_coordinates)
        print('size=', numpy_coordinates.shape, numpy_coordinates.size)
        # print(numpy_coordinates)
        originPointCount = int(numpy_coordinates.shape[0])
        
        t = 1000
        polyPoints = vtk.vtkPoints()
        polyPoints.DeepCopy(edgePolyData.GetPoints())

        points = list(range(originPointCount))
        appear = []
        for i in range(t):
            while True:
                random.shuffle(points)
                avgp = (numpy_coordinates[points[0]] + numpy_coordinates[points[1]] + numpy_coordinates[points[2]]) / 3
                h = hash(str(avgp))
                if h not in appear:
                    polyPoints.InsertPoint(originPointCount + i, avgp)
                    appear.append(h)
                    break

        originData = vtk.vtkPolyData()
        originData.SetPoints(polyPoints)

        constrain = vtk.vtkPolyData()
        constrain.SetPoints(polyPoints)
        constrain.SetPolys(vtk.vtkCellArray())

        delaunayFilter = vtk.vtkDelaunay2D()
        delaunayFilter.SetInputData(originData)
        delaunayFilter.SetSourceData(constrain)
        delaunayFilter.SetTolerance(0.01)
        delaunayFilter.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
        delaunayFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(delaunayFilter.GetOutputPort())
        cleanFilter.Update()

        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(cleanFilter.GetOutputPort())
        smoothFilter.SetNumberOfIterations(200)
        smoothFilter.SetRelaxationFactor(0.05)
        smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()

        subdivisionFilter = vtk.vtkLoopSubdivisionFilter()
        subdivisionFilter.SetNumberOfSubdivisions(2)
        subdivisionFilter.SetInputConnection(smoothFilter.GetOutputPort())
        subdivisionFilter.Update()

        return subdivisionFilter.GetOutput()

    def mergeBreastAndBoundary(self, breastPolyData, wallPolyData):
        # 先移除最外圍的點 避免與胸部data重疊
        _, edgeIds = self.extractBoundaryPoints(wallPolyData)
        rippedWallPolyData = self.deletePoint(wallPolyData, edgeIds)
        rippedWallEdge, _ = self.extractBoundaryPoints(rippedWallPolyData)
        wallStrips = vtk.vtkStripper()
        wallStrips.SetInputData(rippedWallEdge)
        wallStrips.Update()
        edge1 = wallStrips.GetOutput()

        breastEdge, _ = self.extractBoundaryPoints(breastPolyData)
        boundaryStrips = vtk.vtkStripper()
        boundaryStrips.SetInputData(breastEdge)
        boundaryStrips.Update()
        edge2 = boundaryStrips.GetOutput()

        stitcer = PolyDataStitcher()
        stitchPolyData = stitcer.stitch(edge1, edge2)
        #self.createNewModelNode(stitchPolyData, "Stitch")

        #先將胸壁與縫合面合併
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(rippedWallPolyData)
        appendFilter.AddInputData(stitchPolyData)
        appendFilter.Update()

        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(appendFilter.GetOutputPort())
        normalFilter.Update()
        
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(normalFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        holeFilter = vtk.vtkFillHolesFilter()
        holeFilter.SetInputConnection(cleanFilter.GetOutputPort())
        holeFilter.Update()

        #self.createNewModelNode(holeFilter.GetOutput(), "Stitch_Combine")

        #再次合併胸壁和胸部
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(breastPolyData)
        appendFilter.AddInputData(holeFilter.GetOutput())
        appendFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        connectFilter = vtk.vtkPolyDataConnectivityFilter()
        connectFilter.SetInputConnection(cleanFilter.GetOutputPort())
        connectFilter.SetExtractionModeToLargestRegion()
        connectFilter.Update()

        holeFilter = vtk.vtkFillHolesFilter()
        holeFilter.SetInputConnection(connectFilter.GetOutputPort())
        holeFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(holeFilter.GetOutputPort())
        cleanFilter.ConvertLinesToPointsOn()
        cleanFilter.ConvertPolysToLinesOn()
        cleanFilter.ConvertStripsToPolysOn()
        cleanFilter.Update()

        normalFilter = vtk.vtkPolyDataNormals()
        normalFilter.SetInputConnection(cleanFilter.GetOutputPort())
        normalFilter.Update()

        return normalFilter.GetOutput()
        

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)
        return modelNode

class PolyDataStitcher():
    def extract_points(self, source):
        # Travers the cells and add points while keeping their order.
        points = source.GetPoints()
        cells = source.GetLines()
        cells.InitTraversal()
        idList = vtk.vtkIdList()
        pointIds = []
        while cells.GetNextCell(idList):
            for i in range(0, idList.GetNumberOfIds()):
                pId = idList.GetId(i)
                # Only add the point id if the previously added point does not
                # have the same id. Avoid p->p duplications which occur for example
                # if a poly-line is traversed. However, other types of point
                # duplication currently are not avoided: a->b->c->a->d
                if len(pointIds) == 0 or pointIds[-1] != pId:
                    pointIds.append(pId)
        result = []
        for i in pointIds:
            result.append(points.GetPoint(i))
        return result

    def reverse_lines(self, source):
        strip = vtk.vtkStripper()
        strip.SetInputData(source)
        strip.Update()
        reversed = vtk.vtkReverseSense()
        reversed.SetInputConnection(strip.GetOutputPort())
        reversed.Update()
        return reversed.GetOutput()

    def find_closest_point(self, points, samplePoint):
        points = np.asarray(points)
        assert(len(points.shape) == 2 and points.shape[1] == 3)
        nPoints = points.shape[0]
        diff = np.array(points) - np.tile(samplePoint, [nPoints, 1])
        pId = np.argmin(np.linalg.norm(diff, axis=1))
        return pId

    def stitch(self, edge1, edge2):
        # Extract points along the edge line (in correct order).
        # The following further assumes that the polyline has the
        # same orientation (clockwise or counterclockwise).
        edge2 = self.reverse_lines(edge2)

        points1 = self.extract_points(edge1)
        points2 = self.extract_points(edge2)
        n1 = len(points1)
        n2 = len(points2)

        # Prepare result containers.
        # Variable points concatenates points1 and points2.
        # Note: all indices refer to this targert container!
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        points.SetNumberOfPoints(n1+n2)
        for i, p1 in enumerate(points1):
            points.SetPoint(i, p1)
        for i, p2 in enumerate(points2):
            points.SetPoint(i+n1, p2)

        # The following code stitches the curves edge1 with (points1) and
        # edge2 (with points2) together based on a simple growing scheme.

        # Pick a first stitch between points1[0] and its closest neighbor
        # of points2.
        i1Start = 0
        i2Start = self.find_closest_point(points2, points1[i1Start])
        i2Start += n1  # offset to reach the points2

        # Initialize
        i1 = i1Start
        i2 = i2Start
        p1 = np.asarray(points.GetPoint(i1))
        p2 = np.asarray(points.GetPoint(i2))
        mask = np.zeros(n1+n2, dtype=bool)
        count = 0
        while not np.all(mask):
            count += 1
            i1Candidate = (i1+1) % n1
            i2Candidate = (i2+1-n1) % n2+n1
            p1Candidate = np.asarray(points.GetPoint(i1Candidate))
            p2Candidate = np.asarray(points.GetPoint(i2Candidate))
            diffEdge12C = np.linalg.norm(p1-p2Candidate)
            diffEdge21C = np.linalg.norm(p2-p1Candidate)

            mask[i1] = True
            mask[i2] = True
            if diffEdge12C < diffEdge21C:
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, i1)
                triangle.GetPointIds().SetId(1, i2)
                triangle.GetPointIds().SetId(2, i2Candidate)
                cells.InsertNextCell(triangle)
                i2 = i2Candidate
                p2 = p2Candidate
            else:
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, i1)
                triangle.GetPointIds().SetId(1, i2)
                triangle.GetPointIds().SetId(2, i1Candidate)
                cells.InsertNextCell(triangle)
                i1 = i1Candidate
                p1 = p1Candidate

        # Add the last triangle.
        i1Candidate = (i1+1) % n1
        i2Candidate = (i2+1-n1) % n2+n1
        if (i1Candidate <= i1Start) or (i2Candidate <= i2Start):
            if i1Candidate <= i1Start:
                iC = i1Candidate
            else:
                iC = i2Candidate
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, i1)
            triangle.GetPointIds().SetId(1, i2)
            triangle.GetPointIds().SetId(2, iC)
            cells.InsertNextCell(triangle)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)
        poly.BuildLinks()

        return poly

class CTOBJBreastBuilderTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_CTOBJBreastBuilderLogic1()

    def test_CTOBJBreastBuilderLogic1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
        #
        # first, get some data
        #
        import SampleData
        SampleData.downloadFromURL(
            nodeNames='FA',
            fileNames='FA.nrrd',
            uris='http://slicer.kitware.com/midas3/download?items=5767')
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern="FA")
        logic = CTOBJBreastBuilderLogic()
        self.assertIsNotNone( logic.HasImageData(volumeNode) )
        self.delayDisplay('Test passed!')