import os
import torch

def export_and_verify_onnx(model, input_dim, folder_path="data/model", file_name="vae_model.onnx"):
    # 1. Set the model to evaluation mode for a stable export
    model.eval()
    
    # 2. Ensure the directory structure exists safely
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, file_name)
    
    # 3. Create the dummy input matching your feature dimensions
    dummy_input = torch.randn(1, input_dim) 
    
    # 4. Run the export process
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'mu', 'logvar'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 5. Verify and print the result
    if os.path.exists(save_path):
        print(f"Success: ONNX model successfully saved to {save_path}")
        return save_path
    else:
        print(f"Error: Failed to save the model to {save_path}")
        return None
