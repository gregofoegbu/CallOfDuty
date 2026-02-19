import os
import shutil

LABELS_DIR = "../../data2/labels"                # folder containing .txt files
IMAGES_SOURCE_DIR = "../../ExtractedFrames/VOD-2025-11-16-Frames"         # folder where your extracted .jpg frames are
IMAGES_TARGET_DIR = "../../data2/images"         # folder where you want to copy/move matched .jpg files

os.makedirs(IMAGES_TARGET_DIR, exist_ok=True)

label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]

missing = []

for label_file in label_files:
    base = os.path.splitext(label_file)[0]    # "002164"
    img_name = base + ".jpg"                  # "002164.jpg"

    src_img_path = os.path.join(IMAGES_SOURCE_DIR, img_name)
    dst_img_path = os.path.join(IMAGES_TARGET_DIR, img_name)

    if os.path.exists(src_img_path):
        print(f"Moving: {img_name}")
        shutil.copy2(src_img_path, dst_img_path)   # or shutil.move(...)
    else:
        print(f"❌ Missing image for label: {label_file}")
        missing.append(img_name)

print("\nDone!")
print(f"Missing images count: {len(missing)}")
if missing:
    print("Missing files:")
    for m in missing:
        print(" -", m)
