import os
import csv
import pickle
import argparse
import heapq
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from model import MusicTransformer 
from midi_utils import events_to_midi, midi_to_events

def generate_with_sampling(model, start_sequence_events, num_tokens, int_to_event, event_to_int, device, top_k, temperature):
    model.eval()
    generated_tokens = [event_to_int.get(s, event_to_int['<unk>']) for s in start_sequence_events]
    
    pbar = tqdm(range(num_tokens), desc="Генериране с top-k семплиране")
    for i, _ in enumerate(pbar):
        current_sequence = torch.tensor([generated_tokens[-config.SEQUENCE_LENGTH:]], dtype=torch.long, device=device)
        #if i%10==0:
            #print('\n')
            #print(current_sequence[:20])
            #print(current_sequence.shape)
            
        with torch.no_grad():
            output = model(current_sequence)
            logits = output[0, -1, :]
            logits /= temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)
            
    return [int_to_event.get(token, '<unk>') for token in generated_tokens]

def generate_with_beam_search(model, start_sequence_events, num_tokens, int_to_event, event_to_int, device, beam_size, block_ngrams):
    model.eval()
    
    start_tokens = [event_to_int.get(s, event_to_int['<unk>']) for s in start_sequence_events]

    active_beams = [(0.0, start_tokens)]
    
    pbar = tqdm(range(num_tokens), desc=f"Генериране с beam search, k={beam_size})")
    for _ in pbar:
        potential_next_beams = []
        for log_prob, sequence in active_beams:
            current_sequence_tensor = torch.tensor([sequence[-config.SEQUENCE_LENGTH:]], dtype=torch.long, device=device)
            
            with torch.no_grad():
                output = model(current_sequence_tensor)
                next_token_log_probs = F.log_softmax(output[0, -1, :], dim=-1)

            if block_ngrams > 0 and len(sequence) >= block_ngrams:
                existing_ngrams = set()
                for i in range(len(sequence) - block_ngrams + 1):
                    existing_ngrams.add(tuple(sequence[i:i+block_ngrams]))
                
                for i in range(len(next_token_log_probs)):
                    token_to_add = i
                    ngram_candidate = tuple(sequence[-(block_ngrams-1):] + [token_to_add])
                    if ngram_candidate in existing_ngrams:
                        next_token_log_probs[i] = -float('inf')
            
            top_next_log_probs, top_next_tokens = torch.topk(next_token_log_probs, beam_size)
            
            for i in range(beam_size):
                token = top_next_tokens[i].item()
                prob = top_next_log_probs[i].item()
                
                if prob == -float('inf'): continue
                
                new_sequence = sequence + [token]
                new_log_prob = log_prob + prob
                heapq.heappush(potential_next_beams, (-new_log_prob, new_sequence))

        active_beams = []
        seen_sequences = set()
        while potential_next_beams and len(active_beams) < beam_size:
            score, seq = heapq.heappop(potential_next_beams)
            if tuple(seq) not in seen_sequences:
                active_beams.append((-score, seq))
                seen_sequences.add(tuple(seq))
        
    best_log_prob, best_sequence = max(active_beams, key=lambda x: x[0])
    return [int_to_event.get(token, '<unk>') for token in best_sequence]


def get_seed_from_test_file(seed_index, seed_length, event_to_int, int_to_event):
    print(f"\nИнформация: Сийдваме генерацията с тестов файл с пореден номер {seed_index}.")
    csv_path = os.path.join(config.MAESTRO_ROOT_DIR, config.MAESTRO_CSV_NAME)
    test_files = [os.path.join(config.MAESTRO_ROOT_DIR, r['midi_filename']) for r in csv.DictReader(open(csv_path)) if r['split'] == 'test']
    if seed_index >= len(test_files):
        print(f"Грешка: индекс {seed_index} извън броя на семпъли.")
        return None
    seed_file_path = test_files[seed_index]
    print(f"Информация: Използваме сийд файл: {os.path.basename(seed_file_path)}")
    seed_events = midi_to_events(seed_file_path)
    if not seed_events: return None
    seed_tokens = [event_to_int.get(e, event_to_int['<unk>']) for e in seed_events]
    return [int_to_event[t] for t in seed_tokens[:seed_length]]


def main():
    
    parser = argparse.ArgumentParser(description="Генерира музика с обучен Transformer модел.")
    
    parser.add_argument("model_paths", nargs='+', type=str, 
                        help="Път(ища) до файловете с тренирания модел (.pt).")
                        
    parser.add_argument("--output_dir", type=str, default="generated_music", 
                        help="Директория, в която да се запазят генерираните MIDI файлове.")
                        
    parser.add_argument("--num_tokens", type=int, default=1024, 
                        help="Брой нови токени, които да бъдат генерирани след началната последователност.")
                        
    parser.add_argument("--beam_size", type=int, default=0, 
                        help="Ширина на лъча за 'beam search'. Ако е > 0, се активира beam search. В противен случай се използва top-k семплиране.")
                        
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Контролира случайността при семплиране (използва се, ако beam_size=0). По-висока стойност = по-случайно.")
                        
    parser.add_argument("--top_k", type=int, default=20, 
                        help="Филтрира до 'k' на брой най-вероятни токени (използва се, ако beam_size=0).")
                        
    parser.add_argument("--seed", type=str, default="note_on_60 velocity_20", 
                        help="Начална последователност от текстови събития. Използва се, ако --test_seed_index не е зададен.")
                        
    parser.add_argument("--test_seed_index", type=int, default=None, 
                        help="Използвай началото на файл от тестовия сет като начална последователност, зададен чрез неговия индекс.")
                        
    parser.add_argument("--seed_length", type=int, default=100, 
                        help="Брой токени, които да се вземат от началния файл от тестовия сет.")
                        
    parser.add_argument("--block_ngrams", type=int, default=3, 
                        help="Блокира повтарящи се n-грами с този размер. Използва се само с beam search. Задайте 0, за да деактивирате.")

    args = parser.parse_args()
    
    device = torch.device(config.DEVICE)
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(config.TOKEN_DATA_PATH, 'rb') as f: data = pickle.load(f)
    int_to_event, event_to_int = data['int_to_event'], data['event_to_int']
    start_sequence_events = get_seed_from_test_file(args.test_seed_index, args.seed_length, event_to_int, int_to_event) if args.test_seed_index is not None else args.seed.split()
    if start_sequence_events is None: return

    for model_path in args.model_paths:
        print("\n" + "="*50)
        print(f"--- Ползвам модел: {os.path.basename(model_path)} ---")
        
        try:
            model = MusicTransformer(vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBEDDING_DIM, num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS, ff_dim=config.FF_DIM, dropout=config.DROPOUT).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            print(f"Грешка: не е намерен {model_path}. Прескачам.")
            continue
        
        if args.beam_size > 0:
            generated_events = generate_with_beam_search(
                model, start_sequence_events, args.num_tokens, 
                int_to_event, event_to_int, device, 
                args.beam_size, args.block_ngrams
            )
            gen_method_str = f"beam_{args.beam_size}_ngram_{args.block_ngrams}"
        else:
            generated_events = generate_with_sampling(model, start_sequence_events, args.num_tokens, int_to_event, event_to_int, device, args.top_k, args.temperature)
            gen_method_str = f"topk_{args.top_k}_temp_{args.temperature}"

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        seed_str = f"seed_{args.test_seed_index}" if args.test_seed_index is not None else "seed_text"
        output_filename = f"{model_name}_{seed_str}_{gen_method_str}.mid"
        output_path = os.path.join(args.output_dir, output_filename)
        
        events_to_midi(generated_events, output_path)

    print("\n" + "="*50 + "\n--- Готово ---")

if __name__ == '__main__':
    main()