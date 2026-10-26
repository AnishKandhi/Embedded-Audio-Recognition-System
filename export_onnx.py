import torch
import os
from model import ColorAudioCNN

def main():
    print("Step 1: Loading model parameters...")
    model = ColorAudioCNN(num_classes=3)
    model.load_state_dict(torch.load("color_model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    print("Step 2: Exporting as a single self-contained model file...")
    dummy_input = torch.randn(1, 1, 64, 63)
    output_filename = "standalone_color_model.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_filename, 
        export_params=True,        
        opset_version=15,          
        do_constant_folding=True,  
        input_names=['input'],     
        output_names=['output'],   
        
        # THIS IS THE EXACT FIX: Completely blocks the generation of 
        # external data references and forces everything into the file shell.
        use_external_data_format=False
    )
    
    if os.path.exists(output_filename):
        print(f"\n SUCCESS! Standalone model file created: {output_filename}")
        print(f" File Size: {os.path.getsize(output_filename) / 1024:.2f} KB")

if __name__ == "__main__":
    main()