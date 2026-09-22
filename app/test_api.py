import os
import sys

# Ensure root directory is in sys.path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from fastapi.testclient import TestClient
from app.main import app

def run_tests():
    print("Initializing FastAPI TestClient (this will trigger models loading in lifespan)...")
    
    # Using 'with' triggers startup and shutdown lifespan events
    with TestClient(app) as client:
        print("\n--- Testing Health Check Endpoint ---")
        health_resp = client.get("/health")
        print(f"Health Response Status: {health_resp.status_code}")
        print(f"Health Response JSON: {health_resp.json()}")
        assert health_resp.status_code == 200, "Health check failed!"
        
        print("\n--- Testing Predict Endpoint with Sample Image ---")
        sample_img_path = os.path.join(workspace_dir, "images/train/mango10_sia_u.jpg")
        if not os.path.exists(sample_img_path):
            print(f"Error: Sample image not found at {sample_img_path}")
            return
            
        print(f"Uploading: {sample_img_path}")
        with open(sample_img_path, "rb") as f:
            files = {"file": ("PXL_20260415_033647496.jpg", f, "image/jpeg")}
            # Default intrinsics
            data = {
                "fx": 2123.0,
                "fy": 2123.0,
                "cx": 1500.0,
                "cy": 2000.0
            }
            pred_resp = client.post("/predict", files=files, data=data)
            
        print(f"Predict Response Status: {pred_resp.status_code}")
        if pred_resp.status_code != 200:
            print(f"Error details: {pred_resp.text}")
            assert False, "Prediction request failed!"
            
        res_json = pred_resp.json()
        print("Prediction successful! Response structure check:")
        print(f"Success key: {res_json.get('success')}")
        preds = res_json.get("predictions")
        print(f"Orientation predicted: {preds.get('orientation')}")
        print(f"Ripeness predicted: {preds.get('is_ripe')}")
        print(f"Estimated Mango physical properties: {preds.get('mango')}")
        print(f"Estimated Seed (Endocarp) properties: {preds.get('seed')}")
        
        # Verify visualization assets
        viz = res_json.get("visualizations")
        print("\nVisualization overlays present:")
        for name, data_uri in viz.items():
            print(f"  - {name}: {data_uri[:50]}... ({len(data_uri)} chars)")
            assert data_uri.startswith("data:image/png;base64,"), f"{name} is not a valid base64 URI"
            
        print("\nAll integration tests passed successfully!")

if __name__ == "__main__":
    run_tests()
