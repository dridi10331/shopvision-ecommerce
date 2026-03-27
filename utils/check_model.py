from ultralytics import YOLO
import time
import numpy as np

# Load model
model = YOLO('models/best.pt')

# Get model info
print("="*50)
print("MODEL INFORMATION")
print("="*50)
print(f"Number of classes: {len(model.names)}")
print(f"Class names: {list(model.names.values())}")
print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# Test inference speed on dummy image
print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)

# Create dummy 640x640 image
dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(5):
    _ = model(dummy_img, verbose=False)

# Measure inference time
times = []
for _ in range(30):
    start = time.time()
    results = model(dummy_img, verbose=False)
    end = time.time()
    times.append((end - start) * 1000)  # Convert to ms

avg_time = np.mean(times)
fps = 1000 / avg_time

print(f"Average inference time: {avg_time:.1f}ms")
print(f"FPS: {fps:.1f}")
print(f"Min time: {min(times):.1f}ms")
print(f"Max time: {max(times):.1f}ms")

# Try to get validation metrics if available
try:
    results = model.val()
    if results:
        print("\n" + "="*50)
        print("VALIDATION METRICS")
        print("="*50)
        if hasattr(results, 'box'):
            print(f"mAP@0.5: {results.box.map50:.3f}")
            print(f"mAP@0.5-0.95: {results.box.map:.3f}")
            print(f"Precision: {results.box.mp:.3f}")
            print(f"Recall: {results.box.mr:.3f}")
except Exception as e:
    print(f"\nValidation metrics not available: {e}")
