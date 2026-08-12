#!/bin/bash

# --- Configuration ---
root_dir="$(realpath -s $(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/..)"

declare -A HOST_CONFIGS
HOST_CONFIGS["nezha,REMOTE_HOST"]="192.168.0.163"
HOST_CONFIGS["nezha,REMOTE_USER"]="user"
HOST_CONFIGS["nezha,REMOTE_PASSWORD"]="1"
HOST_CONFIGS["nezha,REMOTE_PATH"]="/home/user/deployments/sysid_ws"

HOST_CONFIGS["orin,REMOTE_HOST"]="192.168.0.162"
HOST_CONFIGS["orin,REMOTE_USER"]="ubuntu"
HOST_CONFIGS["orin,REMOTE_PASSWORD"]="ubuntu"
HOST_CONFIGS["orin,REMOTE_PATH"]="/home/ubuntu/deployments/sysid_ws"

# --- Specify the synch files
SYNC_FILES=(
  # "install"                   # need to rebuild on orion
#   "scripts"
#   "models"
#   "src/engineai_deploy" 
#   "src/skin/skin_real"
#   "src/skin/skin_feature_generator"
#   "src/skin/skin_utilities"
#   "src/robots/pm01_description"
#   "src/robots/pm01_ros_bridge"
#   "src/utilities"
    "identification"
)

# --- Argument Parsing ---
DRY_RUN=""
TARGET_HOST="orin"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dry-run) DRY_RUN="--dry-run"; shift ;;
        *) TARGET_HOST="$1"; shift ;;
    esac
done

# --- Validation & Setup ---
if [[ -z "${HOST_CONFIGS["$TARGET_HOST,REMOTE_HOST"]}" ]]; then
    echo "Error: Unknown target host '$TARGET_HOST'"; exit 1
fi

REMOTE_HOST="${HOST_CONFIGS["$TARGET_HOST,REMOTE_HOST"]}"
REMOTE_USER="${HOST_CONFIGS["$TARGET_HOST,REMOTE_USER"]}"
REMOTE_PASSWORD="${HOST_CONFIGS["$TARGET_HOST,REMOTE_PASSWORD"]}"
REMOTE_PATH="${HOST_CONFIGS["$TARGET_HOST,REMOTE_PATH"]}"

if [[ -n "$DRY_RUN" ]]; then echo -e "\033[0;33m[DRY RUN MODE]\033[0m"; fi

# --- Sync Function ---
sync_item() {
    local relative_path=$1
    local local_full_path="$root_dir/$relative_path"
    
    if [ ! -e "$local_full_path" ]; then
        echo "Warning: '$local_full_path' not found, skipping..."
        return
    fi

    # Automatically create the parent directory on the remote side 
    # (e.g., if syncing src/pkg_1, this ensures 'src' exists)
    if [[ -z "$DRY_RUN" ]]; then
        remote_parent=$(dirname "$REMOTE_PATH/$relative_path")
        sshpass -p "$REMOTE_PASSWORD" ssh -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $remote_parent"
    fi

    echo "Syncing: $relative_path"
    
    sshpass -p "$REMOTE_PASSWORD" rsync -avz $DRY_RUN --delete \
        "${local_full_path%/}/" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$relative_path/"
}

# --- Execution ---
for item in "${SYNC_FILES[@]}"; do
    sync_item "$item"
done

echo "Source synchronization completed successfully!"
echo "The Remote Path is: $REMOTE_PATH"