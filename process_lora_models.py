import os
import json
import time
import queue
import torch
import threading
import hashlib
import safetensors.torch
from datetime import datetime

# Filenames for incremental progress and results
PROCESSED_JSON = "processed.json"
GROUPS_JSON = "groups.json"
REFINED_GROUPS_JSON = "refined_groups.json"
DUPLICATES_JSON = "duplicates.json"

# Thread settings
MAX_WORKERS = 4

def load_json(filename):
    if os.path.isfile(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def compute_shape_signature(state_dict):
    shapes = []
    for _, param in state_dict.items():
        shapes.append(tuple(param.shape))
    shape_str = str(shapes).encode("utf-8")
    return hashlib.sha256(shape_str).hexdigest()

def compute_weight_hash(tensor):
    data_bytes = tensor.numpy().tobytes()
    return hashlib.sha256(data_bytes).hexdigest()

def producer_thread(directory, file_queue, processed_data, processed_lock, total_files):
    print(f"[Producer] Scanning directory: {directory}")
    file_list = [f for f in os.listdir(directory) if f.endswith(".safetensors")]
    total_files[0] = len(file_list)
    print(f"[Producer] Found {len(file_list)} .safetensors files.")
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        with processed_lock:
            if file_path in processed_data:
                print(f"[Producer] Skipping already processed file: {file_path}")
                continue
        print(f"[Producer] Loading file: {file_path}")
        start_load = time.time()
        try:
            state_dict = safetensors.torch.load_file(file_path)
            load_time = time.time() - start_load
            file_queue.put((file_path, state_dict, load_time))
            print(f"[Producer] Queued file: {file_path} (Load time: {load_time:.2f}s)")
        except Exception as e:
            print(f"[Producer] Error loading file: {file_path}, {str(e)}")

    # Add sentinel for each consumer so they know we're done producing.
    for _ in range(MAX_WORKERS):
        file_queue.put(None)

def consumer_thread_shape_grouping(
    file_queue,
    groups,
    groups_lock,
    processed_data,
    processed_lock,
    processed_count,
    total_files,
    flush_interval=2
):
    while True:
        item = file_queue.get()
        if item is None:
            print("[Consumer] Received sentinel. Exiting.")
            file_queue.task_done()
            return

        file_path, state_dict, load_time = item
        print(f"[Consumer] Processing file: {file_path}")
        shape_sig = compute_shape_signature(state_dict)
        with groups_lock:
            if shape_sig not in groups:
                groups[shape_sig] = set()
            groups[shape_sig].add(file_path)
        with processed_lock:
            processed_data[file_path] = {
                "load_time_sec": load_time,
                "size_kb": os.path.getsize(file_path) / 1024.0,
            }
            processed_count[0] += 1
            print(f"[Consumer] Completed file: {file_path} ({processed_count[0]}/{total_files[0]})")

            if processed_count[0] % flush_interval == 0:
                print("[Consumer] Flushing processed data and groups to disk.")
                save_json(PROCESSED_JSON, processed_data)
                temp_groups = {k: list(v) for k, v in groups.items()}
                save_json(GROUPS_JSON, temp_groups)

        file_queue.task_done()

def refine_group(group_files, refined_groups, duplicates, refined_groups_lock, duplicates_lock):
    print(f"[Refine] Refining group of {len(group_files)} file(s): {group_files}")
    if len(group_files) <= 1:
        print("[Refine] Group has 1 or 0 files, skipping refinement.")
        return

    current_level = {None: set(group_files)}
    while True:
        new_level = {}
        layer_processed = False

        for subgroup_key, subgroup_files in current_level.items():
            if len(subgroup_files) <= 1:
                if subgroup_key not in new_level:
                    new_level[subgroup_key] = set()
                new_level[subgroup_key].update(subgroup_files)
                continue

            print(f"[Refine] Subgroup of size {len(subgroup_files)} needs further layer-wise refinement.")
            layer_hash_map = {}
            try:
                for f in subgroup_files:
                    state_dict = safetensors.torch.load_file(f)
                    for key, param in state_dict.items():
                        if param.shape != torch.Size([]):
                            file_hash = compute_weight_hash(param)
                            if file_hash not in layer_hash_map:
                                layer_hash_map[file_hash] = []
                            layer_hash_map[file_hash].append(f)
                            break
                layer_processed = True
            except Exception as e:
                print(f"[Refine] Error refining a file in subgroup: {str(e)}")

            for hash_val, files_set in layer_hash_map.items():
                if len(files_set) > 1:
                    if hash_val not in new_level:
                        new_level[hash_val] = set()
                    new_level[hash_val].update(files_set)
                else:
                    if hash_val not in new_level:
                        new_level[hash_val] = set()
                    new_level[hash_val].update(files_set)

        current_level = new_level
        if not layer_processed:
            print("[Refine] No further layer splits possible. Stopping refinement here.")
            break

    final_groups = list(current_level.values())
    for fg in final_groups:
        if len(fg) <= 1:
            continue
        print(f"[Refine] Checking subgroup with {len(fg)} files for true duplicates.")
        reference_file = list(fg)[0]
        try:
            ref_dict = safetensors.torch.load_file(reference_file)
            for other_file in list(fg)[1:]:
                other_dict = safetensors.torch.load_file(other_file)
                if len(ref_dict.keys()) != len(other_dict.keys()):
                    continue
                all_match = True
                for (k1, v1), (k2, v2) in zip(ref_dict.items(), other_dict.items()):
                    if k1 != k2 or not torch.equal(v1, v2):
                        all_match = False
                        break
                if all_match:
                    print(f"[Refine] Found duplicates: {reference_file} and {other_file}")
                    with duplicates_lock:
                        if reference_file not in duplicates:
                            duplicates[reference_file] = set()
                        duplicates[reference_file].add(other_file)
        except Exception as e:
            print(f"[Refine] Error verifying duplicates in subgroup: {str(e)}")

    with refined_groups_lock:
        print("[Refine] Building refined_groups map from final subgroups.")
        for idx, fg in enumerate(final_groups):
            if len(fg) == 1:
                key_str = "unique_" + list(fg)[0]
                refined_groups[key_str] = list(fg)
            else:
                cluster_hash = hashlib.sha256(
                    str(sorted(list(fg))).encode("utf-8")
                ).hexdigest()
                refined_groups[cluster_hash] = list(fg)

def conflict_resolution(groups, refined_groups, duplicates, refined_groups_lock, duplicates_lock):
    print("[ConflictResolution] Starting conflict resolution for groups.")
    for shape_sig, file_paths in groups.items():
        if len(file_paths) > 1:
            print(f"[ConflictResolution] Refining group with shape signature {shape_sig}, size {len(file_paths)}.")
            group_files = list(file_paths)
            refine_group(group_files, refined_groups, duplicates, refined_groups_lock, duplicates_lock)
            save_json(REFINED_GROUPS_JSON, refined_groups)
    print("[ConflictResolution] Completed.")

def run_adaptive_lora_analysis(lora_directory):
    print("[System] Starting Adaptive LoRA Analysis.")
    processed_data = load_json(PROCESSED_JSON)
    raw_groups = load_json(GROUPS_JSON)
    if raw_groups:
        groups = {}
        for k, v in raw_groups.items():
            groups[k] = set(v)
    else:
        groups = {}

    refined_groups = load_json(REFINED_GROUPS_JSON)
    duplicates = load_json(DUPLICATES_JSON)
    if not isinstance(refined_groups, dict):
        refined_groups = {}
    if not isinstance(duplicates, dict):
        duplicates = {}

    processed_count = [len(processed_data)]
    total_files = [0]

    file_queue = queue.Queue()
    groups_lock = threading.Lock()
    processed_lock = threading.Lock()
    refined_groups_lock = threading.Lock()
    duplicates_lock = threading.Lock()

    print("[System] Spawning producer thread.")
    prod_thread = threading.Thread(
        target=producer_thread,
        args=(lora_directory, file_queue, processed_data, processed_lock, total_files),
        daemon=True
    )
    prod_thread.start()

    print(f"[System] Spawning {MAX_WORKERS} consumer thread(s).")
    consumers = []
    for i in range(MAX_WORKERS):
        t = threading.Thread(
            target=consumer_thread_shape_grouping,
            args=(
                file_queue,
                groups,
                groups_lock,
                processed_data,
                processed_lock,
                processed_count,
                total_files
            ),
            daemon=True
        )
        t.start()
        consumers.append(t)

    prod_thread.join()
    print("[System] Producer thread finished. Waiting for queue to drain.")
    file_queue.join()
    print("[System] Consumers have processed all files.")

    print("[System] Saving final processed data and group info.")
    save_json(PROCESSED_JSON, processed_data)
    temp_groups = {k: list(v) for k, v in groups.items()}
    save_json(GROUPS_JSON, temp_groups)

    print("[System] Initiating conflict resolution phase.")
    conflict_resolution(groups, refined_groups, duplicates, refined_groups_lock, duplicates_lock)

    print("[System] Saving final refined groups and duplicates data.")
    save_json(REFINED_GROUPS_JSON, refined_groups)
    save_json(DUPLICATES_JSON, duplicates)

    print("=== Adaptive LoRA Analysis Completed ===")
    print(f"Total files detected: {total_files[0]}")
    print(f"Total files processed: {processed_count[0]}")
    print("See processed.json, groups.json, refined_groups.json, and duplicates.json for details.")

def analyze_lora_models(directory):
    run_adaptive_lora_analysis(directory)
    print("Analysis complete.")

if __name__ == "__main__":
    lora_directory = "D:/ComfyUI_resources/loras"
    analyze_lora_models(lora_directory)
