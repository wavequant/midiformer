import os
import csv
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import config
from midi_utils import is_valid_midi, midi_to_events, get_key_and_transpose_offset

def load_splits_from_csv(root_dir, csv_name):
    midi_paths = {'train': [], 'validation': [], 'test': []}
    csv_path = os.path.join(root_dir, csv_name)
    print(f"Информация: Зареждам дейтасет от {csv_path}...")
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            if split in midi_paths:
                full_midi_path = os.path.join(root_dir, row['midi_filename'])
                if os.path.exists(full_midi_path):
                    midi_paths[split].append(full_midi_path)
    print(f"Информация: Намерени: {len(midi_paths['train'])} train, {len(midi_paths['validation'])} validation, и {len(midi_paths['test'])} test файлове.")
    return midi_paths

def build_vocabulary(all_events):
    print("\n--- Генериране на речник ---")
    event_counts = Counter(all_events)
    most_common_events = [event for event, count in event_counts.most_common(config.VOCAB_SIZE - 2)]
    
    event_to_int = {'<pad>': 0, '<unk>': 1}
    int_to_event = {0: '<pad>', 1: '<unk>'}
    
    for i, event in enumerate(most_common_events, start=2):
        event_to_int[event] = i
        int_to_event[i] = event
        
    print(f"Информация: Размер на речника: {len(event_to_int)}")
    return event_to_int, int_to_event

def main():
    os.makedirs("data", exist_ok=True)
    
    midi_paths_by_split = load_splits_from_csv(config.MAESTRO_ROOT_DIR, config.MAESTRO_CSV_NAME)
    
    all_train_events = []
    
    print("Информация: генерирам речник...")
    with ProcessPoolExecutor() as executor:
        train_midi_paths = midi_paths_by_split['train']
        results = list(tqdm(executor.map(midi_to_events, train_midi_paths), total=len(train_midi_paths), desc="Парсване на МИДИ-та"))

    valid_train_paths = []
    for path, event_list in zip(train_midi_paths, results):
        if event_list:
            valid_train_paths.append(path)
            all_train_events.extend(event_list)

    event_to_int, int_to_event = build_vocabulary(all_train_events)

    print("\n--- Токенизация ---")
    processed_data = {
        'event_to_int': event_to_int,
        'int_to_event': int_to_event
    }

    for split in ['train', 'validation', 'test']:
        print(f"Обработвам {split} събсет-а...")
        all_split_tokens = []
        
        if split == 'train':
            tokens = [event_to_int.get(e, event_to_int['<unk>']) for e in all_train_events]
            all_split_tokens.extend(tokens)
        else:
            with ProcessPoolExecutor() as executor:
                midi_paths = midi_paths_by_split[split]
                results = list(tqdm(executor.map(midi_to_events, midi_paths), total=len(midi_paths), desc=f"Парсване на {split} файлове"))
            
            for event_list in results:
                if event_list:
                    tokens = [event_to_int.get(e, event_to_int['<unk>']) for e in event_list]
                    all_split_tokens.extend(tokens)
        
        processed_data[f'{split}_tokens'] = all_split_tokens
        print(f"Информация: Дължина, в токени, на {split} събсет-а: {len(all_split_tokens):,}")

    print(f"Информация: Записвам файл с токенизираните данни: {config.TOKEN_DATA_PATH}")
    with open(config.TOKEN_DATA_PATH, "wb") as f:
        pickle.dump(processed_data, f)
        
    print("\nПодготовката на данните завърши.")

if __name__ == '__main__':
    main()