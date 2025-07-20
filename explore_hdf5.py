import h5py

# 데이터셋 경로
hdf5_path = "/home/user/Desktop/project/FAIL-Detect/data/robomimic/datasets/can/low_dim_v15.hdf5"
demo_id = "demo_0"

with h5py.File(hdf5_path, "r") as f:
    print(f"🔍 Reading demo: {demo_id}")
    
    # actions shape
    actions = f[f"data/{demo_id}/actions"]
    print(f"✅ actions shape: {actions.shape}")  # e.g., (T, action_dim)

    # rewards shape (optional)
    rewards = f[f"data/{demo_id}/rewards"]
    print(f"✅ rewards shape: {rewards.shape}")

    # states shape (optional)
    states = f[f"data/{demo_id}/states"]
    print(f"✅ states shape: {states.shape}")

    # obs keys and shapes
    obs_group = f[f"data/{demo_id}/obs"]
    print(f"\n✅ obs keys and shapes:")
    obs_dim_total = 0
    for key in obs_group.keys():
        shape = obs_group[key].shape
        dim = shape[1] if len(shape) == 2 else shape
        print(f"  🔹 {key}: shape={shape}")
        if isinstance(dim, int):
            obs_dim_total += dim
        elif isinstance(dim, tuple) and len(dim) == 2:
            obs_dim_total += dim[1]

    print(f"\n📏 total obs_dim: {obs_dim_total}")
    print(f"📏 action_dim: {actions.shape[1]}")
    print(f"📏 trajectory_dim (obs + action): {obs_dim_total + actions.shape[1]}")
