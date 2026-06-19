# A Parallel Consensus Hybrid Framework for Robust LSB Image Steganalysis

This repository contains the complete open-source implementation of a high-precision, dual-domain parallel hybrid framework designed for the structural detection of Spatial Least Significant Bit (LSB) steganography inside digital images.

Developed at **Yuan Ze University, Taiwan**.

---

## Architectural Overview
Unlike traditional steganalysis systems that rely entirely on computationally heavy single-track networks or slow, sequential multi-stage pipelines, this architecture processes media assets through **concurrent analytical channels** simultaneously:

1. **Statistical Domain (Random Forest):** Isolates the raw image LSB bit-plane to evaluate structural bit uniformity across three specific dimension maps: LSB Density Ratio, Horizontal Bit Transitions, and Vertical Bit Transitions.
2. **Deep Learning Spatial Domain (CNN):** Routes inputs through a local $3 \times 3$ Gaussian High-Pass Filter module. This suppresses distracting semantic image shapes/boundaries (background objects) to expose fragile, distributed steganographic spatial noise.

The final prediction is mapped via a **Weighted Soft-Voting Consensus Matrix Layer** assigning a $70\%$ trust weight to the Deep CNN array and a $30\%$ stabilizer weight to the Random Forest baseline.

                 +-----------------------------------+
                 |         INPUT IMAGE (I)           |
                 +-----------------+-----------------+
                                   |
               +-------------------+-------------------+
               | (Simultaneous Split)                  | (Simultaneous Split)
               v                                       v
  +-------------------------+             +-------------------------+
  |  STATISTICAL PIPELINE   |             | DEEP LEARNING PIPELINE  |
  +------------+------------+             +------------+------------+
  | LSB Bit-Plane Extraction|             |  3x3 Gaussian Filtering  |
  | Feature Domain Mapping  |             |  Residual Noise Map     |
  | Random Forest Classifier|             | 2D Convolutional Net    |
  +------------+------------+             +------------+------------+
               |                                       |
               v (Score: P_RF)                         v (Score: P_CNN)
               +-------------------+-------------------+
                                   |
                                   v
                 +-----------------------------------+
                 |     SOFT-VOTING CONSENSUS LAYER   |
                 |  P_final = (0.7*P_CNN)+(0.3*P_RF) |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-----------------------------------+
                 |         FINAL DECISION GATE       |
                 |     [STEGO]     OR     [CLEAN]    |
                 +-----------------------------------+

---

## 🛠️ Key Technical Contributions (Beyond Baseline Projects)
This project was inspired by foundational single-track architectures like `pbrucla/steganography-cnns`. However, our framework introduces several crucial novel modifications to survive real-world deployment tests:
* **Hybrid Parallel Convergence:** Instead of relying strictly on deep network parameters (which are highly susceptible to dataset texture bias), we fuse a discrete statistical validation logic that anchors performance on unconstrained media samples.
* **The Scale Invariance Solution (Patch-Cropping Engine):** Traditional testing libraries pass images into models via standard linear interpolation downscaling (resizing). This blends adjacent pixel elements together, entirely wiping out fragile LSB data. We bypass this blindspot by introducing a localized **32x32 pixel target patch-cropping mechanism** that leaves raw binary streams unaltered during real-world inference testing.
* **Empirical Validation:** Attains an overall classification accuracy profile of **$94.6\%$** with a verified False Positive Rate (FPR) of just **$4.1\%$** on CIFAR-10 derived control groups.

---

##  Installation & System Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/darliehenry033/Steganalysis-Project-.git](https://github.com/darliehenry033/Steganalysis-Project-.git)
   cd Steganalysis-Project-
