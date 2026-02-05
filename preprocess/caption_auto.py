from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import json
import glob
from filelock import FileLock


def get_image_files(directory):
    """Recursively finds all common image files in a directory."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)

def main(args):
    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")


    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    print("Using prompt: \n", prompt_text)

    # data
    all_entries = os.listdir(args.data_root)
    dirs = [d for d in all_entries if os.path.isdir(os.path.join(args.data_root, d))]
    dirs.sort()
    # Process state initialization (if not exists)
    if not os.path.exists(args.process_state_path):
        state = {}
        for d in dirs:
            path = os.path.join('/datasets/imagenet/train', d)
            state[d] = "null"
        with open(args.process_state_path, "w") as f:
            json.dump(state, f, indent=4)

    for dir_name in tqdm(dirs, desc="Processing directories"):
        dir_path = os.path.join(args.data_root, dir_name)
        file_path_list = get_image_files(dir_path)

        # check process state : 'null', 'processing', 'done'
        with FileLock(args.process_state_path + ".lock"):
            with open(args.process_state_path, "r") as f:
                state = json.load(f)
                if state[dir_name] == "null":
                    state[dir_name] = "processing"
                    with open(args.process_state_path, "w") as fw:
                        json.dump(state, fw, indent=4)
                elif state[dir_name] == "processing":
                    continue
                elif state[dir_name] == "done":
                    continue
                else:
                    raise ValueError(f"Unknown state {state[dir_name]} for directory {dir_name}")

        results = []
        for img_path in tqdm(file_path_list, desc=f"Processing images in {dir_name}"):
            # img = Image.open(img_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # Preparation for inference
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            caption = output_text[0].strip()
            rel_path = os.path.relpath(img_path, args.data_root)
            entry = {
                    "path": rel_path,
                    "class_id": dir_name,
                    "caption": caption
                }
            results.append(entry)

        with FileLock(args.process_state_path + ".lock"):
            with open(args.process_state_path, "r") as f:
                state = json.load(f)
                assert state[dir_name] == "processing" , f"Directory {dir_name} should be in 'processing' state."
                # save
                with FileLock(args.output_path + ".lock"):
                    with open(args.output_path, "a", encoding="utf-8") as f:
                        for entry in results:
                            f.write(json.dumps(entry) + "\n")

                # mark as done
                state[dir_name] = "done"
                with open(args.process_state_path, "w") as fw:
                    json.dump(state, fw, indent=4)
                

        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Caption ImageNet dataset using Qwen3-VL-8B-Instruct")
    
    # Data arguments
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing class folders (e.g., /path/to/imagenet/train)")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Directory to save caption JSONL files")
    # Indexing arguments for parallel processing
    parser.add_argument("--process_state_path", type=str, default="imagenet_process_state.json",
                        help="Path to the JSON file tracking processing state")
    # Model arguments
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", 
                        help="Hugging Face model ID")
    parser.add_argument("--prompt_file", type=str, default="prompt.txt",  help="Prompt text file")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    
    main(args)
    
