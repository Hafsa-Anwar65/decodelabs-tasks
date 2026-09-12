# Project 2 
### DecodeLabs Robotics & Automation Internship

## 🎯 Objective

Develop a computer vision system that inspects gears on a conveyor belt and automatically classifies each one as **PASS** or **FAIL** based on visible defects such as broken teeth or cracks using OpenCV.

---

## 🧠 Project Overview

The inspection process consists of three main stages:

### Image Preprocessing

The input image is converted to grayscale, filtered to reduce noise, and thresholded to separate the gear from the background.

### Defect Detection

Contours and convex hull analysis are used to examine the gear shape. Structural irregularities are identified by measuring contour defects.

### Inspection Decision

The detected defects are compared against a predefined threshold. Parts within the acceptable limit are marked **PASS**, while damaged parts are labeled **FAIL**.

---

## 📊 Output

After processing, the system:

* Detects visible gear defects.
* Labels each image as **PASS** or **FAIL**.
* Highlights detected defects on the image.
* Saves the annotated results in the **output** folder.


## ⚙️ Customization

You can adjust:

* Threshold values
* Noise filtering method
* Defect detection sensitivity
* Test images for inspection

---

## 🏭 Real-World Application

The same workflow can be integrated into automated manufacturing systems where cameras inspect products in real time, helping improve quality control and reduce manual inspection.

---

## 📌 Submission Files

* Python source files
* Generated inspection results
* Sample PASS and FAIL images
* Short explanation of the selected defect threshold
