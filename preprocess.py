import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlap(a1, a2, b1, b2):
    # Length of overlap between two 1D ranges.
    return max(0, min(a2, b2) - max(a1, b1))

def box_center(box):
    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def merge_two_boxes(box1, box2):
    
    # Merge two bounding boxes.
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)

    return [left, top, right - left, bottom - top]


def should_merge(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    cx1, cy1 = box_center(box1)
    cx2, cy2 = box_center(box2)

    gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    vertical_overlap = overlap(y1, y1 + h1, y2, y2 + h2)
    horizontal_overlap = overlap(x1, x1 + w1, x2, x2 + w2)

    # Rule 1 : Touching components
    if gap_x <= 2 and gap_y <= 2:
        return True

    # Rule 2 : Broken handwriting
    if (gap_x <= 6 and vertical_overlap > min(h1, h2) * 0.55):
        return True

    # Rule 3 : "="
    if (gap_y <= 8 and horizontal_overlap > min(w1, w2) * 0.75):
        return True
  
    # Rule 4 : Divide (÷)
    if (abs(cx1 - cx2) < max(w1, w2) * 1.2 and abs(cy1 - cy2) < 45):
        total_height = (max(y1 + h1, y2 + h2) - min(y1, y2))
        total_width = (max(x1 + w1, x2 + w2) - min(x1, x2))

        if total_height > max(h1, h2):
            return True

    # Rule 5 : Very close small pieces
    if (gap_x < 10 and gap_y < 10 and max(w1, h1) < 40 and max(w2, h2) < 40):
        return True

    return False

# Merge until stable
def merge_boxes(boxes):

    boxes = [list(b) for b in boxes]
    changed = True
    while changed:

        changed = False
        used = [False] * len(boxes)
        new_boxes = []
        for i in range(len(boxes)):

            if used[i]:
                continue

            current = boxes[i]
            used[i] = True
            merged = True
            while merged:

                merged = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    if should_merge(current, boxes[j]):
                        current = merge_two_boxes(current, boxes[j])
                        used[j] = True
                        merged = True
                        changed = True

            new_boxes.append(current)
        boxes = new_boxes
    return boxes


# Remove tiny noise

def remove_small_components(stats, min_area=25):

    boxes = []
    for i in range(1, len(stats)):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        boxes.append([x, y, w, h])

    return boxes

# Sort boxes left-to-right
def sort_boxes(boxes):
    return sorted(boxes, key=lambda b: (b[0], b[1]))

# Crop Character
def crop_character(thresh, box):

    x, y, w, h = box
    roi = thresh[y:y + h, x:x + w]
    roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    roi = cv2.resize(roi, (224, 224))
    return {"image": roi, "box": (x, y, w, h)}


# Main Preprocessing Function
def preprocess_equation(file):

    # Read Image
    file.seek(0)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to read image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # Morphological Closing
    # Connect nearby strokes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Morphological Opening
    # Remove isolated noise
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh,connectivity=8)

    # Remove Noise
    boxes = remove_small_components(stats, min_area=25)

    # Merge Components
    boxes = merge_boxes(boxes)

    # Remove Invalid Boxes
    filtered = []
    for box in boxes:

        x, y, w, h = box
        if w < 5:
            continue
        if h < 5:
            continue
        if w > thresh.shape[1]:
            continue
        if h > thresh.shape[0]:
            continue
        filtered.append(box)

    boxes = filtered

    # Sort Left -> Right
    boxes = sort_boxes(boxes)

    # Merge Again (sometimes first merge creates new neighbors)

    boxes = merge_boxes(boxes)
    boxes = sort_boxes(boxes)

    characters = []

    H, W = thresh.shape
    for box in boxes:

        x, y, w, h = box
        pad = 3

        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(W - x, w + pad * 2)
        h = min(H - y, h + pad * 2)

        character = crop_character(thresh, (x, y, w, h))
        characters.append(character)

    # Optional Debug Image
    debug = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

    # Draw Bounding Boxes on Threshold Image
    boxed_thresh = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(boxed_thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return boxed_thresh, characters


# TEST

# image_path = r"D:\tech\programming language\equation solver\dataset\Handwritten_equations_images\1.png"

# with open(image_path, "rb") as image_file:
#     output, characters = preprocess_equation(image_file)

# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
# plt.title("Detected Characters")
# plt.axis("off")
# plt.show()


# rows = (len(characters) + 3) // 4
# fig, axes = plt.subplots(rows, 4, figsize=(8, rows * 2))
# axes = np.array(axes).reshape(-1)

# for ax in axes:
#     ax.axis("off")

# for i, char in enumerate(characters):
#     axes[i].imshow(char["image"], cmap="gray")
#     axes[i].set_title(str(i + 1))

# plt.tight_layout()
# plt.show()