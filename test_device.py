#!/usr/bin/env python3
"""
Test script to verify device picking functionality
"""

import torch
from utils import get_device

def test_device_picking():
    print("Testing device picking...")
    
    device, autocast_context = get_device()
    print(f"Selected device: {device}")
    print(f"Autocast context type: {type(autocast_context)}")
    
    # Test tensor creation on device
    test_tensor = torch.randn(2, 3, device=device)
    print(f"Test tensor device: {test_tensor.device}")
    print(f"Test tensor shape: {test_tensor.shape}")
    
    # Test autocast context
    with autocast_context:
        result = test_tensor * 2.0
        print(f"Result tensor device: {result.device}")
        print(f"Result tensor dtype: {result.dtype}")
    
    print("Device picking test completed successfully!")

if __name__ == "__main__":
    test_device_picking() 