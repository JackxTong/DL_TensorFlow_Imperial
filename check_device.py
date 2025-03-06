import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    num_devices  = torch.cuda.device_count()
    for i in range(num_devices):
        device = torch.cuda.device(i)
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        total_mem = torch.cuda.get_device_properties(device).total_memory
        print(f"Device {i}: {total_mem/1024**3:.2f}GB")