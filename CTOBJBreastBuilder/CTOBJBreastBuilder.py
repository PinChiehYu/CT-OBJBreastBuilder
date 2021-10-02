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

attr_tag = "CTOBJ.ATTRIBUTE"

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

        # OBJ相關UI
        
        # Default Parameters
        self.targetColor = qt.QColor("#4573a0")

        # Parameters Area
        parametersCollapsibleButton_2 = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton_2.text = "OBJ Model"
        self.layout.addWidget(parametersCollapsibleButton_2)
        # Layout within the dummy collapsible button
        parametersFormLayout_2 = qt.QFormLayout(parametersCollapsibleButton_2)

        # input model selector
        self.OBJModelSelector = slicer.qMRMLNodeComboBox()
        self.OBJModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.OBJModelSelector.selectNodeUponCreation = True
        self.OBJModelSelector.addEnabled = False
        self.OBJModelSelector.removeEnabled = False
        self.OBJModelSelector.noneEnabled = False
        self.OBJModelSelector.showHidden = False
        self.OBJModelSelector.showChildNodeTypes = False
        self.OBJModelSelector.setMRMLScene(slicer.mrmlScene)
        self.OBJModelSelector.setToolTip("Patient's surface scanning model.")
        parametersFormLayout_2.addRow("Input OBJ Model: ", self.OBJModelSelector)

        # input texture selector
        self.OBJTextureSelector = slicer.qMRMLNodeComboBox()
        self.OBJTextureSelector.nodeTypes = ["vtkMRMLVectorVolumeNode"]
        self.OBJModelSelector.selectNodeUponCreation = True
        self.OBJTextureSelector.addEnabled = False
        self.OBJTextureSelector.removeEnabled = False
        self.OBJTextureSelector.noneEnabled = False
        self.OBJTextureSelector.showHidden = False
        self.OBJTextureSelector.showChildNodeTypes = False
        self.OBJTextureSelector.setMRMLScene(slicer.mrmlScene)
        self.OBJTextureSelector.setToolTip("Image of surface scanning.")
        parametersFormLayout_2.addRow("Texture: ", self.OBJTextureSelector)

        # inpute color selector
        self.colorButton = qt.QPushButton()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        self.colorButton.setToolTip("Color of tapes.")
        parametersFormLayout_2.addRow("Marker Color:", self.colorButton)

        # Texture Button
        self.textureButton = qt.QPushButton("Apply Texture")
        self.textureButton.toolTip = "Paste the texture onto the model."
        self.textureButton.enabled = True

        # Apply Button
        self.preprocessButton = qt.QPushButton("Remove Tape")
        self.preprocessButton.toolTip = "Remove tapes from the model."
        self.preprocessButton.enabled = True

        # Select Breasts
        self.selectBreastButton = qt.QPushButton("Finish Select Breasts")
        self.selectBreastButton.toolTip = "Click after breasts are selected."
        self.selectBreastButton.enabled = True

        objBtnsContainer = qt.QHBoxLayout()
        objBtnsContainer.addWidget(self.textureButton)
        objBtnsContainer.addWidget(self.preprocessButton)
        objBtnsContainer.addWidget(self.selectBreastButton)
        parametersFormLayout_2.addRow(objBtnsContainer)

        # Segmentation and CT 相關UI

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Segmentation and CT"
        #parametersCollapsibleButton.setFont(qt.QFont("Times", 12))
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # input segment editor
        self.segmenationSelector = slicer.qMRMLNodeComboBox()
        self.segmenationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmenationSelector.selectNodeUponCreation = True
        self.segmenationSelector.addEnabled = False
        self.segmenationSelector.removeEnabled = False
        self.segmenationSelector.noneEnabled = False
        self.segmenationSelector.showHidden = False
        self.segmenationSelector.showChildNodeTypes = False
        self.segmenationSelector.setMRMLScene(slicer.mrmlScene)
        self.segmenationSelector.setToolTip("Segmentation of skins and bones")
        parametersFormLayout.addRow("Input Segmenation( Skin and Bone ): ", self.segmenationSelector)

        # Transform Button
        self.transformButton = qt.QPushButton("Apply Transform")
        self.transformButton.toolTip = "Transform breast models to correct places."
        self.transformButton.enabled = True
        #self.transformButton.setFont(qt.QFont("Times", 12, qt.QFont.Black))
        parametersFormLayout.addRow(self.transformButton)
        parametersFormLayout.addRow(" ", None)

        #
        # input CT selector
        #
        self.CTSelector = slicer.qMRMLNodeComboBox()
        self.CTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.CTSelector.selectNodeUponCreation = True
        self.CTSelector.addEnabled = False
        self.CTSelector.removeEnabled = False
        self.CTSelector.noneEnabled = False
        self.CTSelector.showHidden = False
        self.CTSelector.showChildNodeTypes = False
        self.CTSelector.setMRMLScene(slicer.mrmlScene)
        self.CTSelector.setToolTip("Patient's CT Image.")
        parametersFormLayout.addRow("Input CT Image: ", self.CTSelector)

        #
        # Estimate Volume Button
        #
        self.createChestWallButton = qt.QPushButton("Create Chest Wall")
        self.createChestWallButton.toolTip = "Create chest wall segmentation from CT Image."
        self.createChestWallButton.enabled = False
        parametersFormLayout.addRow(self.createChestWallButton)
        parametersFormLayout.addRow(" ", None)

        #
        # Resampling Widget
        #
        self.chestWallSelector = slicer.qMRMLNodeComboBox()
        self.chestWallSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.chestWallSelector.addAttribute("vtkMRMLSegmentationNode", attr_tag, "ChestWall")
        self.chestWallSelector.selectNodeUponCreation = True
        self.chestWallSelector.editEnabled = False
        self.chestWallSelector.addEnabled = False
        self.chestWallSelector.removeEnabled = False
        self.chestWallSelector.noneEnabled = False
        self.chestWallSelector.showHidden = False
        self.chestWallSelector.showChildNodeTypes = False
        self.chestWallSelector.setMRMLScene(slicer.mrmlScene)
        self.chestWallSelector.setToolTip("Output segmentation of chest wall.")
        parametersFormLayout.addRow("Output Chestwall Segmentation : ", self.chestWallSelector)

        self.oversamplingFactorSpinBox = qt.QDoubleSpinBox()
        self.oversamplingFactorSpinBox.setRange(0.05, 2.0)
        self.oversamplingFactorSpinBox.setSingleStep(0.05)
        self.oversamplingFactorSpinBox.setValue(0.5)

        self.segmentationGeometryButton = qt.QPushButton("Change Sampling")
        self.segmentationGeometryButton.toolTip = "Press to change sampling factor."
        self.segmentationGeometryButton.enabled = False
        self.layout.addWidget(self.segmentationGeometryButton)

        samplingContainer = qt.QHBoxLayout()
        samplingContainer.addWidget(self.oversamplingFactorSpinBox)
        samplingContainer.addWidget(self.segmentationGeometryButton)
        parametersFormLayout.addRow("Over Sampling Factor : ", samplingContainer)

        # Create Closed Breast Button
        self.closedBreastButton = qt.QPushButton("Create Closed Breast")
        self.closedBreastButton.toolTip = ""
        self.closedBreastButton.enabled = False
        self.closedBreastButton.setToolTip("Create closed breast segmentations based on chest wal.")
        parametersFormLayout.addRow(self.closedBreastButton)
        parametersFormLayout.addRow(" ", None)


        #
        # Editing Segmentation
        #
        segmentationEditorCollapsibleButton = ctk.ctkCollapsibleButton()
        segmentationEditorCollapsibleButton.text = "Editing Segmentation"
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
        self.statButton.setFont(qt.QFont("Times", 15, qt.QFont.Black))
        segmentationEditorFormLayout.addWidget(self.statButton)


        # connections
        # 防呆
        self.OBJModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOBJModelSelect)
        self.OBJTextureSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOBJModelSelect)
        self.segmenationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentSelect)
        self.CTSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onCTSelect)
        
        # 按鈕行為
        self.createChestWallButton.connect('clicked(bool)', self.onCreateChestWallButton)
        self.colorButton.connect("clicked(bool)", self.onSelectColor)
        self.textureButton.connect("clicked(bool)", self.onTextureButton)
        self.preprocessButton.connect("clicked(bool)", self.onPreprocessButton)
        self.selectBreastButton.connect("clicked(bool)", self.onSelectBreast)
        self.transformButton.connect("clicked(bool)", self.onTransformButton)
        self.closedBreastButton.connect("clicked(bool)", self.onClosedBreastVolumeButton)
        self.statButton.connect('clicked(bool)', self.calculateStatistics)
        self.segmentationGeometryButton.connect('clicked(bool)', self.onSegmentationGeometryButton)

        self.refreshUIState()
    
    def refreshUIState(self):
        self.onOBJModelSelect()
        self.onSegmentSelect()
        self.onCTSelect()
        
        if self.chestWallSelector.currentNode():
            self.segmentationGeometryButton.enabled = True
            self.closedBreastButton.enabled = True
    
    def onOBJModelSelect(self):
        self.textureButton.enabled = self.OBJModelSelector.currentNode() and self.OBJTextureSelector.currentNode()
    
    def onSegmentSelect(self):
        self.transformButton.enabled = self.segmenationSelector.currentNode()

    def onCTSelect(self):
        self.createChestWallButton.enabled = self.CTSelector.currentNode()

    ###

    def onSelectColor(self):
        self.targetColor = qt.QColorDialog.getColor()
        self.colorButton.setStyleSheet("background-color: " + self.targetColor.name())
        self.colorButton.update()

    def onTextureButton(self):
        self.logic.showTextureOnModel(self.OBJModelSelector.currentNode(), self.OBJTextureSelector.currentNode())

    def onPreprocessButton(self):
        self.logic.startPreprocessing(self.OBJModelSelector.currentNode(), self.OBJTextureSelector.currentNode(), self.targetColor)

    def onSelectBreast(self):
        self.logic.truncateBreastPolyData()
        
    def finishPreprocessing(self):
        self.selectBreastButton.enabled = True
        self.logic.setupFiducialNodeOperation()

    def onTransformButton(self):
        # 待修正：如果沒有要讓使用者選擇SegmentationNode就不需要開Selector
        current = slicer.util.getNode("Segmentation")
        self.segmenationSelector.setCurrentNode(current)
        self.segmenationSelector.setMRMLScene(slicer.mrmlScene)

        self.logic.performTransform(self.OBJModelSelector.currentNode(), self.segmenationSelector.currentNode(), self.CTSelector.currentNode())
    
    def onCreateChestWallButton(self):
        self.logic.createChestWall(self.CTSelector.currentNode())
        self.segmentationGeometryButton.enabled = True
        self.closedBreastButton.enabled = True
        
        self.chestWallSelector.setCurrentNode(slicer.util.getNode("ChestWallSegNode"))

    def onClosedBreastVolumeButton(self):
        self.logic.createBreastVolume(self.CTSelector.currentNode())
    
    def onSegmentationGeometryButton(self):
        segmentationNode = self.chestWallSelector.currentNode()

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
        segmentationNode.CreateClosedSurfaceRepresentation()

        print("Finish Resolution Changing")

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
    
    def onReload(self):
        importlib.reload(PectoralSideModule)
        ScriptedLoadableModuleWidget.onReload(self)
        
    def cleanup(self):
        pass

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
        self.chectWallSegNodeName = "ChestWallSegNode"
        self.chestWallName = "ChestWall"
        self.numOfBreast = 2

        self.breastModelName = "Breast_"

        #兩側胸部model的bounding box
        #[xmin, xmax, ymin, ymax, zmin, zmax]
        self.breastBounding = [sys.maxsize, 0, sys.maxsize, 0, sys.maxsize, 0]
        self.chestWallBounding = [sys.maxsize, 0, sys.maxsize, 0, sys.maxsize, 0]

    # Actual algorithm
    def createChestWall(self, inputVolume, pectoralSmoothingIterations=4000):
        #避免已經建立好胸壁又再次執行這個步驟
        if slicer.mrmlScene.GetFirstNodeByName(self.chectWallSegNodeName) is not None:
            print(self.chectWallSegNodeName + " already exists!")
            return

        logging.info('Processing started')
        inputImage = sitkUtils.PullVolumeFromSlicer(inputVolume) #對應:sitkUtils.PushVolumeFromSlicer
        direction = inputImage.GetDirection()
        inputImage = sitk.Flip(inputImage, [direction[0] < 0, direction[4] < 0, direction[8] < 0]) # 根據旋轉矩陣確保所有病人的座標方向一致

        #把一部分不需要的資料消除
        truncatedImage = self.truncateUnecessaryBodyPart(inputImage)

        result = PectoralSideModule.EvaluatePectoralSide(truncatedImage, pectoralSmoothingIterations)

        vtk_result = self.sitkImageToVtkOrientedImage(result)
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segNode.SetName(self.chectWallSegNodeName)
        segNode.SetAttribute(attr_tag, "ChestWall")
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
        segNode.CreateDefaultDisplayNodes()
        segNode.AddSegmentFromBinaryLabelmapRepresentation(vtk_result, self.chestWallName)

        # show in 3D
        segNode.CreateClosedSurfaceRepresentation()
    
    def truncateUnecessaryBodyPart(self, image, expansion = 3):
        imageSize = image.GetSize()

        self.calculateBreastBoundingBox()

        topZBounding = max(self.breastBounding[4] - expansion, 0)
        botZBounding = min(self.breastBounding[5] + expansion + 1, imageSize[2])

        truncated = image[:, :, topZBounding : botZBounding]

        return truncated
    
    def calculateBreastBoundingBox(self):
        for i in range(self.numOfBreast):
            breastModelSegNode = slicer.util.getNode(self.breastModelName + str(i) + "_segmentation")

            image = self.segmentsToSitkImage(breastModelSegNode, True)
            bounding = PectoralSideModule.GetBinaryBoundingBox(image) #[xstart, ystart, zstart, xsize, ysize, zsize]

            for axis in range(3):
                axis_min = axis * 2
                axis_max = axis * 2 + 1
                self.breastBounding[axis_min] = bounding[axis] if self.breastBounding[axis_min] > bounding[axis] else self.breastBounding[axis_min]
                self.breastBounding[axis_max] = bounding[axis] + bounding[axis + 3] if bounding[axis] + bounding[axis + 3] > self.breastBounding[axis_max] else self.breastBounding[axis_max]
    
    def calculateChestWallBoundingBox(self):
        chestWallSegNode = slicer.util.getNode(self.chectWallSegNodeName)

        image = self.segmentsToSitkImage(chestWallSegNode, True)
        bounding = PectoralSideModule.GetBinaryBoundingBox(image) #[xstart, ystart, zstart, xsize, ysize, zsize]

        for axis in range(3):
            axis_min = axis * 2
            axis_max = axis * 2 + 1
            self.chestWallBounding[axis_min] = bounding[axis] if self.chestWallBounding[axis_min] > bounding[axis] else self.chestWallBounding[axis_min]
            self.chestWallBounding[axis_max] = bounding[axis] + bounding[axis + 3] if bounding[axis] + bounding[axis + 3] > self.chestWallBounding[axis_max] else self.chestWallBounding[axis_max]

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

        # change fiducial node name and set to blue
        fiducialNode = slicer.util.getNode("Fiducials")

        ### 根據顏色，修改載入的fiducialNode名字
        color = fiducialNode.GetDisplayNode().GetSelectedColor()
        if color == (1.0, 0.0, 0.0):
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
    
    def performTransform(self, modelNode, segmentationNode, inputVolume):

        # step1
        # 改seg顏色, 改fid node名字
        self.transformSetup(segmentationNode)

        # step2
        # 做transform
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        transformNode.SetName("Registration Transform")
        parameters = {}
    
        try:
            seg_fid = slicer.util.getNode("seg_fid")
            seg_fid.GetMarkupsDisplayNode().SetTextScale(0.0)
            model_fid = slicer.util.getNode("model_fid")
            model_fid.GetMarkupsDisplayNode().SetTextScale(0.0)
        except:
            print("Fiducal not found!")
            return

        parameters["saveTransform"] = transformNode.GetID()
        parameters["movingLandmarks"] = model_fid.GetID()
        parameters["fixedLandmarks"] = seg_fid.GetID()
        fiducialReg = slicer.modules.fiducialregistration
        slicer.cli.runSync(fiducialReg, None, parameters)

        mat = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(mat)
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)

        model_fid.SetAndObserveTransformNodeID(transformNode.GetID())

        color = seg_fid.GetDisplayNode().GetSelectedColor()
        if color==(1.0,0.0,0.0):
            modelNode.SetAndObserveTransformNodeID(transformNode.GetID())
            print("color=red")
            print("transform:", self.numOfBreast)

            for i in range(self.numOfBreast):
                breastNode = slicer.util.getNode(self.breastModelName + str(i))

                #將polydata實際做轉移，而不是只轉觀察方向
                applyTransform = vtk.vtkTransformPolyDataFilter()
                applyTransform.SetTransform(transform)
                applyTransform.SetInputData(breastNode.GetPolyData())
                applyTransform.Update()

                breastNode.SetAndObservePolyData(applyTransform.GetOutput())

        # step3
        # change data type
        self.changeType(inputVolume)

        # 不必要的model眼睛關起來
        modifyModelNode = slicer.util.getNode("Modified_Model")
        modifyModelNode.GetDisplayNode().VisibilityOff()

    def changeType(self, inputVolume):    
        ### export model to seg ###
        print("export:", self.numOfBreast)
        for i in range(self.numOfBreast):
            modelNode = slicer.util.getNode(self.breastModelName + str(i))
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", self.breastModelName + str(i) + "_segmentation")
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
            slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segNode)
            segNode.CreateBinaryLabelmapRepresentation()

    def createBreastVolume(self, inputVolume):
        chestWallSegNode = slicer.util.getNode(self.chectWallSegNodeName)
        chestWallSeg = chestWallSegNode.GetSegmentation()
        sourceSegmentId = chestWallSeg.GetSegmentIdBySegmentName(self.chestWallName)

        closedBreastSegNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        closedBreastSegNode.SetName("ClosedBreastSegNode")
        #把胸壁加到最後的seg node底下
        closedBreastSegNode.GetSegmentation().CopySegmentFromSegmentation(chestWallSeg, sourceSegmentId)
        closedBreastSegNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
        closedBreastSegNode.CreateDefaultDisplayNodes()

        #計算bounding box(目前Reload後須重新計算，完整整合後可移除)
        self.calculateBreastBoundingBox()
        self.calculateChestWallBoundingBox()
        
        #Y軸範圍(基於胸壁厚度)
        y_length = self.chestWallBounding[3] - self.chestWallBounding[2]
        y_max = self.chestWallBounding[2] + 25  # 可調整比例以符合數據

        for n in range(self.numOfBreast):
            # 複製一份胸壁segment到胸部segment
            breastModelSegNode = slicer.util.getNode(self.breastModelName + str(n) + "_segmentation")
            breastModelSeg = breastModelSegNode.GetSegmentation()
            breastModelSeg.CopySegmentFromSegmentation(chestWallSeg, sourceSegmentId)

            image = self.segmentsToSitkImage(breastModelSegNode, True)

            #把加上去的chestwall移除
            chestWallSegId = breastModelSeg.GetSegmentIdBySegmentName(self.chestWallName)
            breastModelSeg.RemoveSegment(chestWallSegId)

            # (281黃, 206綠, 268紅)
            image_shape = image.GetSize()
            print("breast:", self.breastBounding)
            print("chestwall:", self.chestWallBounding)
            print("image_shape:", image_shape)
            for z in range(self.breastBounding[4], self.breastBounding[5] + 1): #Z軸
                axialSlice = image[0:image_shape[0], 0:image_shape[1], z]

                for x in range(self.breastBounding[0], self.breastBounding[1] + 1): #X軸
                    start_growing = False
                    #從胸部前緣，最多走到胸腔的1/2處
                    for y in range(self.breastBounding[2], y_max): #Y軸
                        # 先確定碰到點
                        if not start_growing and axialSlice[x, y] == 1:
                            start_growing = True
                        # 再往後長  
                        elif start_growing and axialSlice[x, y] == 0:
                            axialSlice[x, y] = 1
                        # 長到碰到胸壁
                        elif start_growing and axialSlice[x, y] == 2:
                            break
        
                #res = sitk.GetImageFromArray(array)
                image = sitk.Paste(
                            destinationImage = image,
                            sourceImage = axialSlice,
                            sourceSize = [image_shape[0], image_shape[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, z])
            
            image = sitk.Equal(image, 1)
            vtk_image = self.sitkImageToVtkOrientedImage(image)

            if n==0:
                closedBreastSegNode.AddSegmentFromBinaryLabelmapRepresentation(vtk_image, "ClosedBreast_" + str(n), [1.0, 1.0, 0.0])
            else:
                closedBreastSegNode.AddSegmentFromBinaryLabelmapRepresentation(vtk_image, "ClosedBreast_" + str(n), [0.0, 0.0, 1.0])

        # show in 3d
        closedBreastSegNode.CreateClosedSurfaceRepresentation()
    
    def segmentsToSitkImage(self, segmentationNode, extent = False):
        mode = slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY if extent else slicer.vtkSegmentation.EXTENT_UNION_OF_EFFECTIVE_SEGMENTS 

        ### 為segmentation增加一個相對應的labelmapvolume
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode, mode)

        # 換data type to scalar volume
        outputvolumenode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.modules.volumes.logic().CreateScalarVolumeFromVolume(slicer.mrmlScene, outputvolumenode, labelmapVolumeNode)

        image = sitkUtils.PullVolumeFromSlicer(outputvolumenode)
        image = sitk.Cast(image, sitk.sitkInt16)

        direction = image.GetDirection()
        image = sitk.Flip(image, [direction[0] < 0, direction[4] < 0, direction[8] < 0])

        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        slicer.mrmlScene.RemoveNode(outputvolumenode)

        return image

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

        self.widget.finishPreprocessing()
        
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
        self.numOfBreast = numFids
        print("numOfBreast:",self.numOfBreast)
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

        for i in range(self.numOfBreast):
            # 尋找最接近選取點的mesh
            connectFilter = vtk.vtkPolyDataConnectivityFilter()
            connectFilter.SetInputData(self.modifidedModelNode.GetPolyData())
            connectFilter.SetExtractionModeToClosestPointRegion()
            connectFilter.SetClosestPoint(breastPos[i])
            connectFilter.Update()

            rawBreastPolyData = connectFilter.GetOutput()

            smoothedBreastPolyData = wallGenerator.generateWall(rawBreastPolyData, True)
            self.createNewModelNode(smoothedBreastPolyData, "Breast_{}".format(i))
        
        slicer.mrmlScene.RemoveNode(self.modifidedModelNode)
    
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
        #edgePolydata, _ = self.extractBoundaryPoints(smoothedBreastPolyData)
        #wallMesh = self.createWallMesh(edgePolydata)
        #mergedPolyData = self.mergeBreastAndBoundary(smoothedBreastPolyData, wallMesh)
        
        return smoothedBreastPolyData #, edgePolydata, wallMesh

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

    def createNewModelNode(self, polyData, nodeName):
        modelNode = slicer.mrmlScene.AddNode(slicer.vtkMRMLModelNode())
        modelNode.SetName(nodeName)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(polyData)
        return modelNode

"""
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
"""

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
        #self.assertIsNotNone( logic.HasImageData(volumeNode) )
        self.delayDisplay('Test passed!')
