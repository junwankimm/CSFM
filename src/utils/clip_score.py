import clip
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize
import torch
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import torchvision.transforms.functional as F

class ClipScore:
    def __init__(self,device='cuda', prefix='A photo depicts', weight=1.0):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()
        self.transform = Compose([
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '
        self.w = weight
    
    def extract_all_images(self, images):
        images_input = self.transform(images)
        if self.device == 'cuda':
            images_input = images_input.to(torch.float16)
        image_feature = self.model.encode_image(images_input)
        return image_feature
    
    def extract_all_texts(self, texts, need_prefix):
        if need_prefix:
            c_data = clip.tokenize(self.prefix + texts, truncate=True).to(self.device)
        else:
            c_data = clip.tokenize(texts, truncate=True).to(self.device)
        text_feature = self.model.encode_text(c_data)
        return text_feature
    
    def get_clip_score(self, img, text, need_prefix=False):
        img_f = self.extract_all_images(img)
        text_f = self.extract_all_texts(text,need_prefix)
        images = img_f / torch.sqrt(torch.sum(img_f**2, axis=1, keepdims=True))
        candidates = text_f / torch.sqrt(torch.sum(text_f**2, axis=1, keepdims=True))
        clip_per = self.w * torch.clip(torch.sum(images * candidates, axis=1), 0, None)
        return clip_per
    
    def get_text_clip_score(self, text_1, text_2, need_prefix=False):
        text_1_f = self.extract_all_texts(text_1,need_prefix)
        text_2_f = self.extract_all_texts(text_2,need_prefix)
        candidates_1 = text_1_f / torch.sqrt(torch.sum(text_1_f**2, axis=1, keepdims=True))
        candidates_2 = text_2_f / torch.sqrt(torch.sum(text_2_f**2, axis=1, keepdims=True))
        per = self.w * torch.clip(torch.sum(candidates_1 * candidates_2, axis=1), 0, None)
        results = 'ClipS : ' + str(format(per.item(),'.4f'))
        print(results)
        return per.sum()
    
    def get_img_clip_score(self, img_1, img_2, weight = 1):
        img_f_1 = self.extract_all_images(img_1)
        img_f_2 = self.extract_all_images(img_2)
        images_1 = img_f_1 / torch.sqrt(torch.sum(img_f_1**2, axis=1, keepdims=True))
        images_2 = img_f_2 / torch.sqrt(torch.sum(img_f_2**2, axis=1, keepdims=True))
        per = weight * torch.clip(torch.sum(images_1 * images_2, axis=1), 0, None)
        return per.sum()

    
    def calculate_clip_score(self, images_unprocessed, captions):
        # Assume (N, 3, 224, 224), Resize((224, 224)), [-1, 1]
        images_unprocessed = F.resize(images_unprocessed, (224, 224), antialias=True)
        images_unprocessed = images_unprocessed.to(self.device) 
        # [-1, 1] -> [0, 1] 
        # image_unprocessed = 0.5 * (image_unprocessed + 1.)
        # image_unprocessed.clamp_(0., 1.)
        
        return self.get_clip_score(images_unprocessed, captions)






def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ClipScore_model = ClipScore(device=device)
    
    print(f"Loading data from {args.jsonl_path}...")
    data_pairs = []
    with open(args.jsonl_path, 'r') as f:
        for line in f:
            data_pairs.append(json.loads(line))
    
    print(f"Loaded {len(data_pairs)} image-caption pairs.")
    
    try:
        resizer = Resize((224, 224), antialias=True)
    except TypeError:
        resizer = Resize((224, 224))
    
    all_clip_scores = []
    
    for i in tqdm(range(0, len(data_pairs), args.batch_size), desc="Calculating CLIP Scores"):
        batch_data = data_pairs[i : i + args.batch_size]
        
        image_batch = []
        caption_batch = []
        
        for item in batch_data:
            try:
                caption = item['recaption_short']
                image_path = os.path.join(args.image_base_dir, item['path'])
                
                img = Image.open(image_path).convert("RGB")
                img_tensor = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
                # img_tensor = (img_tensor * 2.0) - 1.0 # (3, H, W) 텐서
                img_tensor = resizer(img_tensor)
                
                image_batch.append(img_tensor)
                caption_batch.append(caption)
                
            except Exception as e:
                print(f"Warning: Skipping {item.get('path', 'N/A')} due to error: {e}")
        
        if not image_batch:
            continue
            
        image_tensors = torch.stack(image_batch)
        
        with torch.no_grad():
            clip_scores_batch = ClipScore_model.calculate_clip_score(image_tensors, caption_batch)
            all_clip_scores.extend(clip_scores_batch.cpu().numpy())

    scores_array = np.array(all_clip_scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    count = len(scores_array)

    print("\n--- Results ---")
    print(f"Total valid pairs processed: {count}")
    print(f"Mean CLIP Score: {mean_score:.4f}")
    print(f"Std Dev CLIP Score: {std_score:.4f}")

    with open(args.output_file, 'w') as f:
        f.write(f"Total valid pairs processed: {count}\n")
        f.write(f"Mean CLIP Score: {mean_score:.4f}\n")
        f.write(f"Std Dev CLIP Score: {std_score:.4f}\n")

    print(f"\nStatistics saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP scores for image-caption pairs.")
    
    parser.add_argument("--jsonl-path", type=str, default="/scratch/jk9321/IN21K_CC12M/imagenet1k_val_captions_qwen3_8B_instruct_fixed.jsonl",
                        help="Path to the .jsonl file containing image paths and captions.")
    
    parser.add_argument("--image-base-dir", type=str, default="/imagenet/val",
                        help="Base directory for the relative image paths in the .jsonl file.")
    
    parser.add_argument("--output-file", type=str, default="/home/jk9321/scratch/jk9321/IN21K_CC12M/clip_score_stats_new.txt", 
                        help="Path to save the output text file with mean/std stats.")
    
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size for processing.")
    
    args = parser.parse_args()
    # breakpoint()
    main(args)