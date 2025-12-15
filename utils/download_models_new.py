from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="namdp-ptit/ViRanker",
    local_dir="./models/namdp-ptit/ViRanker",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="bkai-foundation-models/vietnamese-bi-encoder",
    local_dir="./models/bkai-foundation-models/vietnamese-bi-encoder",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="./models/Qwen/Qwen3-4B",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    local_dir="./models/Qwen/Qwen3-VL-4B-Instruct",
    local_dir_use_symlinks=False
)