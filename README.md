\# RGB-HSV K-Means Image Segmentation



An image segmentation project that compares K-means clustering in RGB and HSV color spaces using greedy centroid initialization.



\## Overview



This project implements K-means based image segmentation in two different color spaces:



\- RGB

\- HSV



A custom greedy centroid selection method is used to initialize the K-means centroids. The segmentation results are then compared visually to examine the differences between the RGB and HSV representations.



The project also calculates a pixel-wise absolute difference between the RGB segmentation result and the additional HSV segmentation result.



\## Methodology



\### 1. Image Loading and Preprocessing



The input image is loaded and converted from BGR to RGB.



The image is resized using a configurable resize factor before segmentation.



\### 2. Greedy Centroid Initialization



Instead of selecting all K-means centroids randomly, the project uses a greedy initialization strategy.



The first centroid is selected randomly.



Each following centroid is selected as the pixel having the maximum distance from the nearest already-selected centroid.



This initialization is used for both RGB and HSV feature spaces.



\### 3. RGB K-Means Segmentation



The RGB image is reshaped into a collection of pixel vectors.



K-means clustering is then performed iteratively:



1\. Assign each pixel to its closest centroid.

2\. Calculate the new centroid of each cluster.

3\. Repeat until the centroids converge or the maximum number of iterations is reached.



The clustered pixels are then reconstructed into the segmented RGB image.



\### 4. HSV K-Means Segmentation



The image is converted from RGB to HSV and the same K-means procedure is applied in HSV space.



The HSV segmentation result is converted back to RGB for visualization.



\### 5. Segmentation Comparison



The project compares:



\- RGB segmentation

\- HSV segmentation

\- Additional HSV segmentation



A visual comparison is provided to examine differences between the resulting segmentations.



\### 6. Difference Analysis



The absolute pixel-wise difference between the RGB segmentation and the additional HSV segmentation is calculated using OpenCV.



This result provides a visual representation of the areas where the two segmentation approaches produce different pixel values.



\## Parameters



The main configurable parameters include:



```python

k = 4

max\_iter = 8

resize\_factor = 0.3

```



Where:



\- `k` is the number of clusters.

\- `max\_iter` is the maximum number of K-means iterations.

\- `resize\_factor` controls the input image scaling.



\## Results



\### Segmentation Comparison



The following image compares the RGB and HSV segmentation outputs.



!\[Segmentation Comparison](results/segmentation\_comparison.png)



\### RGB vs HSV Difference



The following image shows the absolute difference between the RGB segmentation and the additional HSV segmentation.



!\[RGB vs HSV Difference](results/rgb\_hsv\_difference.png)



\## Requirements



Install the required dependencies with:



```bash

pip install -r requirements.txt

```



\## Usage



Run the script with:



```bash

python rgb\_hsv\_kmeans\_segmentation.py

```



The program will process the input image and generate the segmentation comparison and difference results.



\## Project Structure



```text

RGB-HSV-KMeans-Image-Segmentation/

│

├── rgb\_hsv\_kmeans\_segmentation.py

├── README.md

├── requirements.txt

├── .gitignore

│

└── results/

&#x20;   ├── segmentation\_comparison.png

&#x20;   └── rgb\_hsv\_difference.png

```



\## Technologies



\- Python

\- OpenCV

\- NumPy

\- Matplotlib

\- K-Means Clustering

\- Image Processing

\- Computer Vision

\- RGB Color Space

\- HSV Color Space

