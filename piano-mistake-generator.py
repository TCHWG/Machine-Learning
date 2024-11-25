import pretty_midi
import numpy as np
import pandas as pd
import copy
import random
from typing import List, Tuple, Optional, Dict
from enum import Enum
import os

class ErrorType(Enum):
    FORWARD_BACKWARD_INSERTION = "forward_backward_insertion"
    MISTOUCH = "mistouch"
    NOTE_SUBSTITUTION = "note_substitution"
    DRAGGING_NOTE = "dragging_note"
    NO_ERROR = "no_error"

class PianoMistakeGenerator:
    def __init__(self, midi_file: str):
        try:
            self.midi = pretty_midi.PrettyMIDI(midi_file)
            self.original_midi = copy.deepcopy(self.midi)  
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")
        self.original_notes = self._extract_notes()
        self.timing_noise_range = (-0.05, 0.05)  
        self.velocity_noise_range = (-10, 10)    
        self.outlier_probability = 0.05          
        
    def _add_noise_to_note(self, note: pretty_midi.Note, noise_level: float = 1.0):
        """Add controlled noise to a single note"""
        # Timing noise
        timing_noise = random.gauss(0, 0.02 * noise_level)
        note.start += timing_noise
        note.end += timing_noise
        
        # Velocity noise
        note.velocity = int(np.clip(
            note.velocity + random.gauss(0, 5 * noise_level),
            20, 127  # MIDI velocity range
        ))
        
        # Duration noise
        duration_noise = random.gauss(0, 0.01 * noise_level)
        note.end = max(note.start + 0.1, note.end + duration_noise) 
        
        return note

    def _generate_outlier(self, note: pretty_midi.Note) -> pretty_midi.Note:
        """Generate an outlier version of a note"""
        outlier = copy.deepcopy(note)
        outlier_type = random.choice(['timing', 'velocity', 'pitch', 'duration'])
        
        if outlier_type == 'timing':
            # Large timing shift
            shift = random.uniform(0.5, 1.0) * random.choice([-1, 1])
            outlier.start += shift
            outlier.end += shift
        
        elif outlier_type == 'velocity':
            # Extreme velocity
            outlier.velocity = random.choice([
                random.randint(0, 20),    # Very soft
                random.randint(120, 127)  # Very loud
            ])
        
        elif outlier_type == 'pitch':
            # Extreme pitch shift
            outlier.pitch += random.choice([
                random.randint(-24, -18),  # Very low
                random.randint(18, 24)     # Very high
            ])
        
        else:  # duration
            if random.random() < 0.5:
                # Very short
                outlier.end = outlier.start + random.uniform(0.05, 0.1)
            else:
                # Very long
                outlier.end = outlier.start + random.uniform(2.0, 3.0)
        
        return outlier
        
    def _extract_notes(self) -> List[Dict]:
        """Extract notes with their temporal relationships"""
        notes = []
        for instrument in self.midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                })
        return sorted(notes, key=lambda x: x['start'])

    def _find_neighbor_pitch(self, pitch: int, forward: bool = True) -> int:
        """Find neighboring pitch in the scale"""
        scale_steps = [0, 2, 4, 5, 7, 9, 11]  # Major scale steps
        pitch_class = pitch % 12
        octave = pitch // 12
        
        current_step_idx = min(range(len(scale_steps)), 
                             key=lambda i: abs(scale_steps[i] - pitch_class))
        
        if forward:
            next_step_idx = (current_step_idx + 1) % len(scale_steps)
            if next_step_idx == 0:
                octave += 1
        else:
            next_step_idx = (current_step_idx - 1) % len(scale_steps)
            if next_step_idx == len(scale_steps) - 1:
                octave -= 1
                
        return octave * 12 + scale_steps[next_step_idx]

    def generate_forward_backward_insertion(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Generate forward or backward neighbor insertions"""
        modified = copy.deepcopy(midi)
        notes = modified.instruments[0].notes
        
        for i in range(len(notes)):
            if random.random() < 0.15:  # 15% chance for insertion
                neighbor = copy.deepcopy(notes[i])
                forward = random.choice([True, False])
                neighbor.pitch = self._find_neighbor_pitch(notes[i].pitch, forward)
                
                overlap = random.uniform(0.05, 0.15)  # 50-150ms overlap
                if forward:
                    neighbor.start = notes[i].start - overlap
                else:
                    neighbor.end = notes[i].end + overlap
                    
                notes.insert(i, neighbor)
        
        return modified

    def generate_mistouch(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Generate mistouch errors - notes within 2 semitones"""
        modified = copy.deepcopy(midi)
        notes = modified.instruments[0].notes
        
        for i in range(len(notes)):
            if random.random() < 0.12:  # 12% chance for mistouch
                mistouch = copy.deepcopy(notes[i])
                # Pitch difference is 1 or 2 semitones
                pitch_diff = random.choice([-2, -1, 1, 2])
                mistouch.pitch = notes[i].pitch + pitch_diff
                
                # Modify timing and velocity
                mistouch.start += random.uniform(-0.03, 0.03)
                mistouch.end += random.uniform(-0.03, 0.03)
                mistouch.velocity = int(mistouch.velocity * random.uniform(0.7, 0.9))
                
                notes.insert(i, mistouch)
        
        return modified

    def generate_note_substitution(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Generate note substitution errors, including chord substitutions"""
        modified = copy.deepcopy(midi)
        notes = modified.instruments[0].notes
        
        # Group notes into chords
        time_threshold = 0.05  # 50ms threshold for chord detection
        i = 0
        while i < len(notes):
            chord_notes = [notes[i]]
            j = i + 1
            while j < len(notes) and abs(notes[j].start - notes[i].start) < time_threshold:
                chord_notes.append(notes[j])
                j += 1
            
            # Decide whether to substitute this note/chord
            if random.random() < 0.1:  # 10% chance for substitution
                if len(chord_notes) > 1:  # Chord substitution
                    # Replace with parallel major/minor or shift the entire chord
                    shift = random.choice([-12, -7, -3, 3, 7, 12])
                    for note in chord_notes:
                        note.pitch += shift
                else:  # Single note substitution
                    chord_notes[0].pitch += random.choice([-12, -7, -3, 3, 7, 12])
            
            i = j
        
        return modified

    def generate_dragging_note(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Generate dragging note errors - shifting notes by time"""
        modified = copy.deepcopy(midi)
        notes = modified.instruments[0].notes
        
        # Choose a random point to start dragging
        if len(notes) > 1:
            drag_start_idx = random.randint(0, len(notes) - 1)
            drag_time = random.uniform(0.1, 0.5)  # 100-500ms drag
            
            # Apply progressive dragging effect
            for i in range(drag_start_idx, len(notes)):
                notes[i].start += drag_time
                notes[i].end += drag_time
                drag_time += random.uniform(0, 0.05)
        
        return modified

    def extract_features(self, midi: pretty_midi.PrettyMIDI) -> Dict:
        """Extract enhanced features for mistake detection"""
        features = {}
        notes = midi.instruments[0].notes
        
        # Basic temporal features
        note_timings = [note.start for note in notes]
        if len(note_timings) > 1:
            ioi_diffs = np.diff(note_timings)
            features['avg_ioi'] = np.mean(ioi_diffs)
            features['std_ioi'] = np.std(ioi_diffs)
            features['max_ioi'] = np.max(ioi_diffs)
            features['ioi_range'] = np.max(ioi_diffs) - np.min(ioi_diffs)
        else:
            features.update({'avg_ioi': 0, 'std_ioi': 0, 'max_ioi': 0, 'ioi_range': 0})
            
        # Enhanced timing features
        if len(notes) > 1:
            # Detect potential dragging
            timing_diffs = np.diff([note.start for note in notes])
            features['timing_irregularity'] = np.std(timing_diffs) / np.mean(timing_diffs)
            features['max_timing_jump'] = np.max(timing_diffs)
            
            # Detect overlaps
            overlaps = []
            for i in range(len(notes)-1):
                if notes[i].end > notes[i+1].start:
                    overlaps.append(notes[i].end - notes[i+1].start)
            features['overlap_count'] = len(overlaps)
            features['max_overlap'] = max(overlaps) if overlaps else 0
            
        # Enhanced pitch features
        pitches = [note.pitch for note in notes]
        if len(pitches) > 1:
            pitch_diffs = np.abs(np.diff(pitches))
            features['pitch_irregularity'] = np.std(pitch_diffs)
            features['small_intervals'] = np.sum(pitch_diffs <= 2) / len(pitch_diffs)  # Potential mistouches
            features['large_jumps'] = np.sum(pitch_diffs >= 12) / len(pitch_diffs)
            
            # Detect potential substitutions
            scale_degrees = [p % 12 for p in pitches]
            features['non_scale_notes'] = sum(1 for d in scale_degrees if d not in [0,2,4,5,7,9,11]) / len(pitches)
        
        original_notes = self.original_midi.instruments[0].notes
        features['note_count_diff'] = len(notes) - len(original_notes)
        
        # Analyze note density
        total_duration = notes[-1].end - notes[0].start if notes else 0
        features['note_density'] = len(notes) / total_duration if total_duration > 0 else 0
        
        return features

    def generate_dataset(self, num_samples: int = 100) -> Tuple[List[pretty_midi.PrettyMIDI], List[str], List[Dict]]:
        """Generate dataset with various mistakes and features"""
        dataset = []
        labels = []
        features_list = []
        
        error_generators = {
            ErrorType.FORWARD_BACKWARD_INSERTION: self.generate_forward_backward_insertion,
            ErrorType.MISTOUCH: self.generate_mistouch,
            ErrorType.NOTE_SUBSTITUTION: self.generate_note_substitution,
            ErrorType.DRAGGING_NOTE: self.generate_dragging_note
        }
        
        samples_per_error = num_samples // (len(ErrorType))  
        
        for error_type in ErrorType:
            if error_type != ErrorType.NO_ERROR:
                for _ in range(samples_per_error):
                    modified = error_generators[error_type](self.midi)
                    
                    # Add noise to all notes
                    noise_level = random.uniform(0.5, 1.5)  # Vary noise level
                    for note in modified.instruments[0].notes:
                        self._add_noise_to_note(note, noise_level)
                    
                    if random.random() < self.outlier_probability:
                        note_idx = random.randint(0, len(modified.instruments[0].notes) - 1)
                        modified.instruments[0].notes[note_idx] = self._generate_outlier(
                            modified.instruments[0].notes[note_idx]
                        )
                    
                    features = self.extract_features(modified)
                    features['has_error'] = 1
                    features['error_type'] = error_type.value
                    
                    dataset.append(modified)
                    labels.append(error_type.value)
                    features_list.append(features)
        
        no_error_samples = num_samples - samples_per_error * (len(ErrorType) - 1) 
        for _ in range(no_error_samples):
            clean = copy.deepcopy(self.midi)
            noise_level = random.uniform(0.3, 0.7)  
            for note in clean.instruments[0].notes:
                self._add_noise_to_note(note, noise_level)
            
            if random.random() < self.outlier_probability * 0.5:  
                note_idx = random.randint(0, len(clean.instruments[0].notes) - 1)
                clean.instruments[0].notes[note_idx] = self._generate_outlier(
                    clean.instruments[0].notes[note_idx]
                )
            
            features = self.extract_features(clean)
            features['has_error'] = 0
            features['error_type'] = ErrorType.NO_ERROR.value
            
            dataset.append(clean)
            labels.append(ErrorType.NO_ERROR.value)
            features_list.append(features)
        
        return dataset, labels, features_list

    def save_dataset(self, dataset: List[pretty_midi.PrettyMIDI], labels: List[str], 
                    features_list: List[Dict], output_dir: str):
        """Save generated dataset to MIDI files and CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save MIDI files
        for i, (midi, label) in enumerate(zip(dataset, labels)):
            filename = os.path.join(output_dir, f"sample_{i:04d}_{label}.mid")
            midi.write(filename)
        
        df = pd.DataFrame(features_list)
        df['midi_file'] = [f"sample_{i:04d}_{label}.mid" for i, label in enumerate(labels)]
        cols = ['midi_file'] + [col for col in df.columns if col != 'midi_file']
        df = df[cols]
        df.to_csv(os.path.join(output_dir, "dataset_features.csv"), index=False)

if __name__ == "__main__":
    import argparse
    import pathlib
    
    parser = argparse.ArgumentParser(description='Enhanced Piano Mistake Generator')
    parser.add_argument("input_midi_file", type=pathlib.Path, help="Path to input MIDI file")
    parser.add_argument("output_folder", type=pathlib.Path, help="Root output folder")
    parser.add_argument('--var', nargs='?', default=100, const=100, type=int, 
                    help='Number of variations to generate')
    args = parser.parse_args()
    
    if not args.input_midi_file.exists():
        raise FileNotFoundError(f"Input MIDI file {args.input_midi_file} does not exist")
    
    generator = PianoMistakeGenerator(str(args.input_midi_file))
    dataset, labels, features = generator.generate_dataset(num_samples=args.var)
    generator.save_dataset(dataset, labels, features, str(args.output_folder))