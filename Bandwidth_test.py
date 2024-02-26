import torch
import time

def BandwidthTest():
    # Get Device param
    DeviceNum = torch.cuda.device_count()
    device_ids = [i for i in range(DeviceNum)]

    # Init data on Device 0
    data_size_mb = 1024
    data = torch.randn(data_size_mb* 1024 * 1024 // 4, device=device_ids[0])
    targets = [torch.device(f"cuda:{device_id}") for device_id in device_ids]

    # Bandwidth
    results = []

    for target in targets:
        result = torch.zeros_like(data)
        start_time = time.time()
        result.copy_(data, non_blocking=True)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        results.append(result)
        print(f"Elapsed time for copying to {target}: {elapsed_time} seconds")



def main():
    BandwidthTest()


if __name__ == "__main__":
    main()