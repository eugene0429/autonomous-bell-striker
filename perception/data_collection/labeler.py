import cv2
import os
import argparse
import sys

# Global variables
drawing = False
ix, iy = -1, -1
img_vis = None
img_clean = None
bboxes = []  # List of current image bboxes: (class_id, x_center, y_center, width, height)
current_class_id = 0
classes = []

def yolo_to_pixel(bbox, img_w, img_h):
    cls_id, x_c, y_c, w, h = bbox
    xmin = int((x_c - w/2) * img_w)
    ymin = int((y_c - h/2) * img_h)
    xmax = int((x_c + w/2) * img_w)
    ymax = int((y_c + h/2) * img_h)
    return xmin, ymin, xmax, ymax

def draw_bboxes(img):
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    for bbox in bboxes:
        cls_id, cx, cy, bw, bh = bbox
        xmin, ymin, xmax, ymax = yolo_to_pixel(bbox, w, h)
        label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        
        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display current instruction
    instructions = [
        "L-Click & Drag: Draw BBox",
        "A / D: Prev / Next Image (Auto-saves)",
        "Z: Undo last BBox",
        "C: Clear all BBoxes",
        "0-9: Change Class",
        "Q / ESC: Quit"
    ]
    for i, inst in enumerate(instructions):
        cv2.putText(img_copy, inst, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Display current class
    current_class_name = classes[current_class_id] if current_class_id < len(classes) else str(current_class_id)
    cv2.putText(img_copy, f"Current Class [{current_class_id}]: {current_class_name}", 
                (10, 30 + len(instructions) * 25 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return img_copy

def draw_rect(event, x, y, flags, param):
    global drawing, ix, iy, img_vis, img_clean, bboxes, current_class_id
    
    if img_clean is None:
        return

    h, w = img_clean.shape[:2]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_vis = draw_bboxes(img_clean)
            cv2.rectangle(img_vis, (ix, iy), (x, y), (0, 255, 255), 2)
            cv2.imshow("YOLO Labeler", img_vis)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
        xmin, xmax = min(ix, x), max(ix, x)
        ymin, ymax = min(iy, y), max(iy, y)
        
        # Valid BBox check
        if xmax - xmin > 5 and ymax - ymin > 5:
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            
            bboxes.append((current_class_id, x_center, y_center, box_w, box_h))
        
        img_vis = draw_bboxes(img_clean)
        cv2.imshow("YOLO Labeler", img_vis)

def save_labels(txt_path):
    if len(bboxes) == 0:
        if os.path.exists(txt_path):
            os.remove(txt_path) # Remove empty label files
        return
        
    with open(txt_path, 'w') as f:
        for bbox in bboxes:
            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

def load_labels(txt_path):
    global bboxes
    bboxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

def run_labeler(img_dir, out_dir=None, user_classes=["bell1", "bell2"]):
    global img_vis, img_clean, bboxes, current_class_id, classes
    
    classes = user_classes
    if out_dir is None:
        out_dir = img_dir
        
    if not os.path.exists(img_dir):
        print(f"Error: Image directory '{img_dir}' not found.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"No images found in {img_dir}")
        sys.exit(1)
        
    cv2.namedWindow("YOLO Labeler", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLO Labeler", draw_rect)
    
    idx = 0
    while True:
        if idx < 0:
            idx = 0
            print("Already at the first image.")
        elif idx >= len(images):
            idx = len(images) - 1
            print("Reached the end of images.")
            
        img_name = images[idx]
        img_path = os.path.join(img_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(out_dir, txt_name)
        
        img_clean = cv2.imread(img_path)
        if img_clean is None:
            print(f"Could not read {img_path}")
            idx += 1
            if idx >= len(images):
                break
            continue
            
        load_labels(txt_path)
        
        img_vis = draw_bboxes(img_clean)
        cv2.imshow("YOLO Labeler", img_vis)
        print(f"[{idx+1}/{len(images)}] Loaded: {img_name} (Labels: {len(bboxes)})")
        
        while True:
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('d'):  # Next
                save_labels(txt_path)
                idx += 1
                break
            elif key == ord('a'):  # Prev
                save_labels(txt_path)
                idx -= 1
                break
            elif key == ord('c'):  # Clear
                bboxes = []
                img_vis = draw_bboxes(img_clean)
                cv2.imshow("YOLO Labeler", img_vis)
            elif key == ord('q') or key == 27:  # Quit (q or ESC)
                save_labels(txt_path)
                print("Exiting and saving.")
                cv2.destroyAllWindows()
                return
            elif ord('0') <= key <= ord('9'):
                new_class = key - ord('0')
                if new_class < len(classes):
                    current_class_id = new_class
                    img_vis = draw_bboxes(img_clean)
                    cv2.imshow("YOLO Labeler", img_vis)
                else:
                    print(f"Class '{new_class}' not valid. Maximum class id is {len(classes)-1}.")
            # Remove last bbox (z)
            elif key == ord('z'):
                if len(bboxes) > 0:
                    bboxes.pop()
                    img_vis = draw_bboxes(img_clean)
                    cv2.imshow("YOLO Labeler", img_vis)
                    print("Undid last bounding box.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Easy YOLO BBox Labeling Tool for Teams")
    parser.add_argument("-i", "--img_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("-o", "--out_dir", type=str, default=None, help="Directory to save labels (Default: same as img_dir)")
    parser.add_argument("-c", "--classes", type=str, nargs='+', default=["bell1", "bell2"], help="List of class names (e.g. -c car pedestrian bus)")
    args = parser.parse_args()
    
    run_labeler(args.img_dir, args.out_dir, args.classes)
