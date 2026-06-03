import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# STEP 1: Configuration & Bounding Box
# ==========================================
LAT_BOUNDS = (39.50, 40.00)
LON_BOUNDS = (-121.80, -121.20)
GRID_SIZE = 64 

DATA_DIR = os.path.join("..", "test_and_tar")
WEIGHTS_PATH = "camp_fire_weights.pth" 

# ==========================================
# STEP 2: Import Your Model
# ==========================================
# Integrating the classes from your 'maine.py' file
from maine import EarthformerQML, Config, build_config_from_data

def pixel_to_latlon(y, x, lat_bounds, lon_bounds, grid_size=64):
    """Converts a 64x64 pixel index back to real-world Latitude and Longitude."""
    lat_range = lat_bounds[1] - lat_bounds[0]
    lon_range = lon_bounds[1] - lon_bounds[0]
    
    lat = lat_bounds[1] - (y / grid_size) * lat_range
    lon = lon_bounds[0] + (x / grid_size) * lon_range
    return lat, lon

def run_camp_fire_test():
    print("🔥 INITIALIZING CAMP FIRE PREDICTION TEST 🔥\n")

    # 1. Load the Data first to build the Config
    print("1. Loading Preprocessed ERA5 & FIRMS Data...")
    test_input = np.load(os.path.join(DATA_DIR, "test_input.npy"))
    test_target = np.load(os.path.join(DATA_DIR, "test_target.npy"))
    
    # Use your helper to auto-generate the config based on data dimensions
    # This ensures patch sizes and d_model are correct
    cfg = build_config_from_data(test_input, test_target)
    
    # 2. Initialize the Model with Config
    print("2. Initializing EarthformerQML with Dynamic Config...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EarthformerQML(cfg) # Integrated initialization
    
    if os.path.exists(WEIGHTS_PATH):
        # Using weights_only=True for security/best practice in newer PyTorch
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
        print("   Weights loaded successfully!")
    else:
        print(f"   WARNING: Could not find '{WEIGHTS_PATH}'. Running with untrained weights!")
    
    model.to(device)
    model.eval() 

    # 3. Prepare Single Sequence for Inference
    target_idx = -9
    # Shape expected by model: (B, T, H, W, C)
    X_tensor = torch.from_numpy(test_input[target_idx]).unsqueeze(0).float().to(device)
    
    # 4. Run Inference
    print("\n3. Running Weather Data through AI...")
    with torch.no_grad():
        # The model returns a dictionary of results
        output_dict = model(X_tensor)
        
        # We focus on the 'fire_probability' head
        # Result shape: [B, T_out, 1, H, W]
        prediction_logits = output_dict["fire_probability"]
        
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(prediction_logits)
        
        # Grab the first batch, first time step, and remove single-dim channels
        risk_map = probs[0, 0, 0].cpu().numpy()

    # 5. Analyze Results
    print("\n4. Analyzing Spatial Predictions...")
    
    max_risk_val = np.max(risk_map)
    y_idx, x_idx = np.unravel_index(np.argmax(risk_map), risk_map.shape)
    
    pred_lat, pred_lon = pixel_to_latlon(y_idx, x_idx, LAT_BOUNDS, LON_BOUNDS, GRID_SIZE)
    
    print(f"\n==================================================")
    print(f" RESULTS: CAMP FIRE PREDICTION")
    print(f"==================================================")
    print(f" Highest Ignition Risk: {(max_risk_val * 100):.2f}%")
    print(f" Predicted Coordinates: Latitude {pred_lat:.4f}, Longitude {pred_lon:.4f}")
    print(f" Actual Camp Fire start: ~Latitude 39.81, Longitude -121.43")
    print(f"==================================================\n")

    # 6. Generate Visual Heatmap
    print("5. Generating Visual Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Use a more detailed color map for fire 'magma' or 'inferno'
    im = plt.imshow(risk_map, cmap='magma', origin='upper')
    plt.colorbar(im, label="Ignition Probability")
    
    plt.scatter(x_idx, y_idx, s=200, facecolors='none', edgecolors='cyan', linewidth=2, label="AI Prediction")
    
    plt.title(f"Earthformer Fire Risk Prediction\nMax Risk: {max_risk_val*100:.1f}% at [{pred_lat:.2f}, {pred_lon:.2f}]")
    plt.legend()
    
    save_name = "camp_fire_predictions_map.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"   Heatmap saved as '{save_name}'!")

if __name__ == "__main__":
    run_camp_fire_test()