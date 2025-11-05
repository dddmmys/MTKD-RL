import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hashlib
from dataset.cifar100 import get_cifar100_dataloaders

def validate_deterministic_loading():
    data = "./data/cifar100"
    train_loader, val_loader = get_cifar100_dataloaders(
        data_folder=data, 
        batch_size=64, 
        num_workers=8, 
        shuffle_train=False, 
        use_augmentation=False, 
        drop_last=True
    )
    
    # 收集前几个批次的哈希值
    batch_hashes = []
    for i, (images, labels) in enumerate(train_loader):
        if i >= 5:  # 检查5个批次就够了
            break
            
        # 计算批次哈希
        combined_bytes = images.numpy().tobytes() + labels.numpy().tobytes()
        batch_hash = hashlib.md5(combined_bytes).hexdigest()
        batch_hashes.append(batch_hash)
        print(f"Batch {i}: {batch_hash}")
    
    return batch_hashes

# 多次运行验证
print("第一次运行:")
hashes1 = validate_deterministic_loading()

print("\n第二次运行:")
hashes2 = validate_deterministic_loading()

print("\n哈希值是否一致:", hashes1 == hashes2)

"""
第一次运行:
Files already downloaded and verified
Files already downloaded and verified
Batch 0: 99dd97d833a7414933946acf2bacbc6d
Batch 1: 67975fc431db037c04e9ba3acac810ab
Batch 2: af5b61cf6409df46218b89391b13f5b7
Batch 3: c7b7af408dfcd581ef5c89f507f9fe73
Batch 4: 235349b204e49d3cb01e018672e7cb44

第二次运行:
Files already downloaded and verified
Files already downloaded and verified
Batch 0: 99dd97d833a7414933946acf2bacbc6d
Batch 1: 67975fc431db037c04e9ba3acac810ab
Batch 2: af5b61cf6409df46218b89391b13f5b7
Batch 3: c7b7af408dfcd581ef5c89f507f9fe73
Batch 4: 235349b204e49d3cb01e018672e7cb44
"""