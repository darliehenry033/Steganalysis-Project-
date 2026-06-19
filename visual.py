import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from google.colab import files
import cv2

print("Please upload an image to visualize its steganographic features:")
uploaded_vis = files.upload()

if uploaded_vis:
    file_name = list(uploaded_vis.keys())[0]
    raw_img = Image.open(file_name)

    # 1. Prepare raw patch (preserving pure LSB data)
    width, height = raw_img.size
    left = (width - 32) // 2
    top = (height - 32) // 2
    cropped_img = raw_img.crop((left, top, left + 32, top + 32))
    img_array = np.array(cropped_img)

    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # 2. Extract Mathematical Visualizations
    # Calculate Residual Map (What the CNN looks at)
    blur = cv2.GaussianBlur(img_array, (3, 3), 0)
    residual = img_array.astype(np.float32) - blur.astype(np.float32)
    # Normalize residual channel for visual display
    residual_show = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Calculate LSB Bit Planar Map (What the Random Forest looks at)
    lsb_map = img_array & 1
    # Multiply by 255 so 1s show as pure white and 0s as pure black
    lsb_map_show = (lsb_map[:, :, 0] * 255).astype(np.uint8)

    # Calculate exact feature metrics
    img_lsb_ratio = lsb_map.mean()
    img_h_trans = np.mean(lsb_map[:, 1:] != lsb_map[:, :-1])
    img_v_trans = np.mean(lsb_map[1:, :] != lsb_map[:-1, :])

    # 3. Plotting the Model Insight Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Steganalysis Diagnostic View: {file_name}", fontsize=16, fontweight='bold', y=0.95)

    # Top Left: Original Extracted Patch
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("1. Extracted 32x32 Target Patch\n(Original Vector Area)", fontsize=12)
    axes[0, 0].axis('off')

    # Top Right: Residual Texture Analysis
    axes[0, 1].imshow(residual_show)
    axes[0, 1].set_title("2. CNN Input: High-Pass Residual Map\n(Isolates High-Frequency Pixel Noise)", fontsize=12)
    axes[0, 1].axis('off')

    # Bottom Left: LSB Bit Distribution
    axes[1, 0].imshow(lsb_map_show, cmap='gray')
    axes[1, 0].set_title("3. Random Forest View: Raw LSB Bit Mask\n(Looking for unnatural noise symmetry)", fontsize=12)
    axes[1, 0].axis('off')

    # Bottom Right: Transition Profile Bar Graphs
    metrics = ['LSB Ratio', 'Horiz. Trans', 'Vert. Trans']
    current_values = [img_lsb_ratio, img_h_trans, img_v_trans]
    stego_baselines = [0.50, 0.50, 0.50]  # True random stego naturally pulls close to 0.50
    clean_baselines = [0.48, 0.41, 0.41]  # Natural camera/smooth textures sit much lower

    x = np.arange(len(metrics))
    width_bar = 0.25

    axes[1, 1].bar(x - width_bar, clean_baselines, width_bar, label='Typical Clean Profile', color='#2ecc71', alpha=0.6)
    axes[1, 1].bar(x, current_values, width_bar, label='Your Uploaded File', color='#e74c3c')
    axes[1, 1].bar(x + width_bar, stego_baselines, width_bar, label='Expected Stego Baseline', color='#34495e', alpha=0.4)

    axes[1, 1].set_title("4. Statistical Bit Signatures", fontsize=12)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylim(0, 0.65)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1, 1].legend(loc='lower left')

    plt.tight_layout()
    plt.show()

    # 4. Trigger Prediction Output with Context
    print("\n" + "="*60)
    print("DIAGNOSTIC INSIGHT:")
    if img_h_trans > 0.46 or img_v_trans > 0.46:
        print("-> Notice step 4: Your image's transition rates are elevated close to 50.00%.")
        print("   This flat statistical uniformity is a signature indicator of encrypted/hidden payloads.")
    else:
        print("-> Notice step 4: Your image's transition rates track closely with natural clean textures,")
        print("   suggesting variations are likely regular compression artifacts rather than hidden bits.")
    print("="*60)
