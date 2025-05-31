import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the images
img1 = mpimg.imread('Screenshot-2025-01-28-at-9.58.16-AM.jpg')  # 4-layer CNN
img2 = mpimg.imread('Screenshot-2025-02-01-at-3.24.21-PM.jpg')  # 6-layer CNN
img3 = mpimg.imread('Screenshot-2025-01-28-at-9.58.44-AM.jpg')  # 2-layer CNN

# Create a stacked plot
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Plot each image with corresponding labels
axes[0].imshow(img3)  # 2-Layer CNN (moved to top)
axes[0].axis('off')
axes[0].set_title('2-Layer CNN Validation Accuracy')

axes[1].imshow(img1)  # 4-Layer CNN (moved to middle)
axes[1].axis('off')
axes[1].set_title('4-Layer CNN Validation Accuracy')

axes[2].imshow(img2)  # 6-Layer CNN (moved to bottom)
axes[2].axis('off')
axes[2].set_title('6-Layer CNN Validation Accuracy')

plt.subplots_adjust(hspace=0.4)
plt.show()
