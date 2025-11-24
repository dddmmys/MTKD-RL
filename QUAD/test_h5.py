import h5py

# file_path = "QUAD/exp/vq/RegNetY_400MF_layer5_cb2/num_batches_10-layer_5-embedding_embeddings.h5"
file_path = "QUAD/exp/vq/RegNetX_400MF_layer5_cb8/splits1/train_ci.h5"

def basic_h5_inspection(file_path):
    """
    快速查看HDF5文件的基本结构
    """
    with h5py.File(file_path, 'r') as f:
        print("文件中的数据集（或组）有：")
        i = 0
        for key in f.keys():
            print(f"  - {key}")
            # 判断是组还是数据集，并显示其形状
            if isinstance(f[key], h5py.Dataset):
                print(f"    形状: {f[key].shape}, 数据类型: {f[key].dtype}")
            else:
                print(f"    [这是一个组，包含以下对象：]")
                for sub_key in f[key].keys():
                    print(f"      - {sub_key}")
            i += 1
            if i >=10: break

basic_h5_inspection(file_path)