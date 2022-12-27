from datetime import datetime
import os
from pathlib import Path



def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir

def save_text_samples(all_texts_list, save_path):
    full_text = '\n'.join(all_texts_list)
    with open(save_path, 'w', encoding='utf-8') as fo:
        fo.write(full_text)