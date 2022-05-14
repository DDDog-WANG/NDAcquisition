# NDAcquisition

* This is one of my research now in Saito Lab. The outline of this research is like figure below. The data is multiple cells, so first step is to split each cell and save as one cell image, next step is to classify with two classes that the cell is healthy or ill. Finally visualize extracted features that is express some important biology meaning, and explan them by biology knowedge like chromosome heterochromatin. 

  ![修士研究](./DataSample/修士研究.png)

## Data

* The data is not provided in public, the data sample is as following.

  * This is a piece of Z-stack telescope data that is made by Collaborator 光山先生. 
  * Two different label that the cell is sick or healthy(use medicine to cure the sick part)


![0120](./DataSample/0120.png) ![0140](./DataSample/0140.png)

![0220](./DataSample/0220.png)

![0240](./DataSample/0240.png)



## Instance Segmentation

* using maskrcnn

![segmentationsSample](./DataSample/segmentationsSample.png)

* Then get the new one cell datasets.

![onecell0140](./DataSample/onecell0140.png)

![onecell0240](./DataSample/onecell0240.png)

## Classification

* Using resnet18
* Using transformer

## Visualization and Explanation

* using the GradCAM and GuidedBackprop

  ![camgb01](./DataSample/camgb01.png)

  ![camgb02](./DataSample/camgb02.png)

