import os
import pretty_midi
import music21
import numpy as np
import mido

#VELOCITY_BINS = np.linspace(0, 127, 32, dtype=int)
#TIME_SHIFT_BINS = np.linspace(0.01, 2, 200, dtype=float)

VELOCITY_BINS = np.linspace(0, 127, 6, dtype=int)
TIME_SHIFT_BINS = np.linspace(0.01, 0.5, 15, dtype=float)

def quantize_velocity(velocity):
    return int(np.argmin(np.abs(VELOCITY_BINS - velocity)))

def quantize_time_shift(time_shift):
    if time_shift <= 0:
        return 0
    return int(np.argmin(np.abs(TIME_SHIFT_BINS - time_shift))) + 1

def is_valid_midi(midi_path: str):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        time_signatures = pm.time_signature_changes
        if not time_signatures or not (time_signatures[0].numerator == 4 and time_signatures[0].denominator == 4):
            return False

        total_notes = sum(len(instrument.notes) for instrument in pm.instruments)
        if pm.get_end_time() < 15 or total_notes < 50:
            return False

    except Exception:
        return False
        
    return True

def get_key_and_transpose_offset(midi_path: str):
    try:
        score = music21.converter.parse(midi_path)
        
        key = score.analyze('key')

        if key.mode == 'major':
            target_key = music21.pitch.Pitch('C')
        elif key.mode == 'minor':
            target_key = music21.pitch.Pitch('A')
        else:
            target_key = music21.pitch.Pitch('C')

        interval = music21.interval.Interval(key.tonic, target_key)
        return interval.semitones
    except Exception:
        return 0

def midi_to_events(midi_path: str):
    if not is_valid_midi(midi_path):
        return []

    transpose_offset = get_key_and_transpose_offset(midi_path)
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_note_events = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            transposed_pitch = note.pitch + transpose_offset
            if 0 <= transposed_pitch <= 127:
                all_note_events.append(('note_on', note.start, transposed_pitch, note.velocity))
                all_note_events.append(('note_off', note.end, transposed_pitch, 0))

    all_note_events.sort(key=lambda x: x[1])

    event_strings = []
    last_time = 0.0

    for event_type, event_time, pitch, velocity in all_note_events:
        time_shift = event_time - last_time
        if time_shift > 0:
            time_bin = quantize_time_shift(time_shift)
            if time_bin > 0:
                event_strings.append(f"time_shift_{time_bin}")
        
        quantized_pitch = int(pitch)
        if event_type == 'note_on':
            event_strings.append(f"note_on_{quantized_pitch}")
            event_strings.append(f"velocity_{quantize_velocity(velocity)}")
        else:
            event_strings.append(f"note_off_{quantized_pitch}")
            
        last_time = event_time
        
    return event_strings

def events_to_midi(events: list[str], output_path: str, ticks_per_beat: int = 480):
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    delta_time_in_seconds = 0.0
    current_velocity = 64 

    for event in events:
        parts = event.split('_')
        event_type = parts[0]
        
        if event_type == "time": 
            time_val_index = int(parts[-1])
            if time_val_index > 0:
                delta_time_in_seconds += TIME_SHIFT_BINS[time_val_index - 1]
        
        elif event_type == "velocity":
            velocity_bin = int(parts[-1])
            current_velocity = int(VELOCITY_BINS[velocity_bin])
            
        elif event_type == "note":
            pitch = int(parts[-1])

            ticks = int(mido.second2tick(delta_time_in_seconds, ticks_per_beat, 500000))
            
            velocity = current_velocity if parts[1] == 'on' else 0
            
            track.append(mido.Message(
                'note_on', 
                note=pitch, 
                velocity=velocity, 
                time=ticks
            ))
            delta_time_in_seconds = 0.0
            
    midi.save(output_path)
    print(f"Информация: МИДИ файлове запазени в {output_path}")