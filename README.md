# A Parallel Consensus Hybrid Framework for Robust LSB Image Steganalysis

This repository contains the complete open-source implementation of a high-precision, dual-domain parallel hybrid framework designed for the structural detection of Spatial Least Significant Bit (LSB) steganography inside digital images.

Developed at **Yuan Ze University, Taiwan**.

---

## 🚀 Architectural Overview
Unlike traditional steganalysis systems that rely entirely on computationally heavy single-track networks or slow, sequential multi-stage pipelines, this architecture processes media assets through **concurrent analytical channels** simultaneously:

1. **Statistical Domain (Random Forest):** Isolates the raw image LSB bit-plane to evaluate structural bit uniformity across three specific dimension maps: LSB Density Ratio, Horizontal Bit Transitions, and Vertical Bit Transitions.
2. **Deep Learning Spatial Domain (CNN):** Routes inputs through a local $3 \times 3$ Gaussian High-Pass Filter module. This suppresses distracting semantic image shapes/boundaries (background objects) to expose fragile, distributed steganographic spatial noise.

The final prediction is mapped via a **Weighted Soft-Voting Consensus Matrix Layer** assigning a $70\%$ trust weight to the Deep CNN array and a $30\%$ stabilizer weight to the Random Forest baseline.
