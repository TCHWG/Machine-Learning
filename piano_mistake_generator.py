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
    def __init__(self, midi_file: str, window_size: float = 2.0, window_overlap: float = 0.5):
        """
        Initialize the windowed mistake generator
        
        Args:
            midi_file (str): Path to the input MIDI file
            window_size (float): Size of the window in seconds
            window_overlap (float): Overlap between windows as a fraction of window size
        """
        try:
            self.midi = pretty_midi.PrettyMIDI(midi_file)
            self.original_midi = copy.deepcopy(self.midi)  
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")
        
        self.window_size = window_size
        self.window_overlap = window_overlap

    def _extract_windowed_notes(self, start_time: float, end_time: float, midi_object = None) -> List[Dict]:
        """
        Extract notes within a specific time window
        
        Args:
            start_time (float): Start of the window
            end_time (float): End of the window
        
        Returns:
            List of notes within the window
        """
        windowed_notes = []
        midi_object = self.midi if midi_object is None else midi_object
        for instrument in midi_object.instruments:
            for note in instrument.notes:
                # Check if note is within or overlaps the window
                if (note.start < end_time and note.end > start_time):
                    # Adjust note start and end to window boundaries
                    adjusted_note = {
                        'pitch': note.pitch,
                        'start': max(note.start, start_time),
                        'end': min(note.end, end_time),
                        'velocity': note.velocity,
                        'duration': min(note.end, end_time) - max(note.start, start_time)
                    }
                    windowed_notes.append(adjusted_note)
        
        return sorted(windowed_notes, key=lambda x: x['start'])

    def _compute_window_comparison_features(self, original_notes: List[Dict], modified_notes: List[Dict]) -> Dict:
        """
        Compute comparative features between original and modified window
        """
        features = {}
        
        # Pitch comparison
        original_pitches = [note['pitch'] for note in original_notes]
        modified_pitches = [note['pitch'] for note in modified_notes]
        
        features['pitch_change_count'] = sum(
            1 for p_orig, p_mod in zip(original_pitches, modified_pitches) 
            if p_orig != p_mod
        )
        
        # Timing comparison
        original_starts = [note['start'] for note in original_notes]
        modified_starts = [note['start'] for note in modified_notes]
        
        features['timing_shift_magnitude'] = np.mean([
            abs(orig - mod) for orig, mod in zip(original_starts, modified_starts)
        ])
        
        # Velocity comparison
        original_velocities = [note['velocity'] for note in original_notes]
        modified_velocities = [note['velocity'] for note in modified_notes]
        
        features['velocity_variation'] = np.std(
            [abs(orig - mod) for orig, mod in zip(original_velocities, modified_velocities)]
        )
        
        # Structural changes
        features['note_count_change'] = len(modified_notes) - len(original_notes)
        features['note_density_change'] = (
            len(modified_notes) / (modified_notes[-1]['end'] - modified_notes[0]['start']) -
            len(original_notes) / (original_notes[-1]['end'] - original_notes[0]['start'])
        )
        
        return features
      
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

    def generate_windowed_dataset(self, num_samples_per_window: int = 5) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Generate a windowed dataset with mistake variations
        """
        # Determine window parameters
        total_duration = self.midi.get_end_time()
        
        # Compute window start points with overlap
        window_step = self.window_size * (1 - self.window_overlap)
        window_starts = np.arange(0, total_duration - self.window_size, window_step)
        
        # Dataset storage
        dataset_windows = []
        dataset_features = []
        dataset_labels = []
        
        # Error generators
        error_generators = {
            ErrorType.FORWARD_BACKWARD_INSERTION: self.generate_forward_backward_insertion,
            ErrorType.MISTOUCH: self.generate_mistouch,
            ErrorType.NOTE_SUBSTITUTION: self.generate_note_substitution,
            ErrorType.DRAGGING_NOTE: self.generate_dragging_note,
            ErrorType.NO_ERROR: self.generate_no_error  
        }
        
        # Iterate through windows
        for start_time in window_starts:
            end_time = start_time + self.window_size
            
            # Extract original window notes
            original_window_notes = self._extract_windowed_notes(start_time, end_time)
            
            # Skip if no notes in window
            if not original_window_notes:
                continue
            
            # Generate multiple variations for each window
            for _ in range(num_samples_per_window):
                # Randomly choose error type
                error_type = random.choice(list(error_generators.keys()))
                
                # Create modified window
                modified_midi = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(0)  # Piano
                
                # Convert original window notes to MIDI notes
                midi_window_notes = [
                    pretty_midi.Note(
                        pitch=note['pitch'], 
                        start=note['start'], 
                        end=note['end'], 
                        velocity=note['velocity']
                    ) for note in original_window_notes
                ]
                instrument.notes = midi_window_notes
                modified_midi.instruments.append(instrument)
                
                # Apply mistake generation
                modified_midi = error_generators[error_type](modified_midi)
                
                # Extract modified window notes
                modified_window_notes = [
                    {
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity,
                        'duration': note.end - note.start
                    } for note in modified_midi.instruments[0].notes 
                    if start_time <= note.start < end_time
                ]
                
                # Compute comparative features
                comparison_features = self._compute_window_comparison_features(
                    original_window_notes, 
                    modified_window_notes
                )
                
                # Add metadata
                comparison_features['window_start'] = start_time
                comparison_features['window_end'] = end_time
                comparison_features['error_type'] = error_type.value
                
                # Store results
                dataset_windows.append({
                    'original_window': original_window_notes,
                    'modified_window': modified_window_notes
                })
                dataset_features.append(comparison_features)
                dataset_labels.append(error_type.value)
        
        return dataset_windows, dataset_features, dataset_labels
    
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
        """Generate mistouch errors"""
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
        """Generate note substitution errors"""
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
        """Generate dragging note errors"""
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
      
      
    def generate_no_error(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Generate no note errors"""
        clean = copy.deepcopy(self.midi)
        notes = clean.instruments[0].notes 
        
        for note in notes:
            note.start += random.uniform(-0.02, 0.02)
            note.end += random.uniform(-0.02, 0.02)
            note.velocity += random.randint(-5, 5)
        
        return clean
      
      

    def save_windowed_dataset(self, 
                               dataset_windows: List[Dict], 
                               dataset_features: List[Dict], 
                               dataset_labels: List[str], 
                               output_dir: str):
        """
        Save the windowed dataset to CSV and potentially MIDI files
        
        Args:
            dataset_windows (List[Dict]): Windows dataset
            dataset_features (List[Dict]): Extracted features
            dataset_labels (List[str]): Error type labels
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame of features
        df = pd.DataFrame(dataset_features)
        df['label'] = dataset_labels
        df['midi_file'] = [f"window_{i:04d}_{label}.mid" for i, label in enumerate(dataset_labels)]
        column_order = [
        'midi_file', 
        'window_start', 
        'window_end', 
        'pitch_change_count', 
        'timing_shift_magnitude', 
        'velocity_variation', 
        'note_count_change', 
        'note_density_change', 
        'error_type', 
        'label'
        ]
        for col in column_order:
            if col not in df.columns:
                df[col] = None
                
        df = df[column_order]
        # Save features to CSV
        df.to_csv(os.path.join(output_dir, 'dataset_features.csv'), index=False)
        
        for i, (window_data, label) in enumerate(zip(dataset_windows, dataset_labels)):
            original_midi = pretty_midi.PrettyMIDI()
            modified_midi = pretty_midi.PrettyMIDI()
            
            original_instrument = pretty_midi.Instrument(0)
            modified_instrument = pretty_midi.Instrument(0)
            
            # Convert original window notes to MIDI notes
            original_instrument.notes = [
                pretty_midi.Note(
                    pitch=note['pitch'], 
                    start=note['start'], 
                    end=note['end'], 
                    velocity=note['velocity']
                ) for note in window_data['original_window']
            ]
            
            # Convert modified window notes to MIDI notes
            modified_instrument.notes = [
                pretty_midi.Note(
                    pitch=note['pitch'], 
                    start=note['start'], 
                    end=note['end'], 
                    velocity=note['velocity']
                ) for note in window_data['modified_window']
            ]
            
            original_midi.instruments.append(original_instrument)
            modified_midi.instruments.append(modified_instrument)
            
            original_midi.write(os.path.join(output_dir, f'original_window_{i}_{label}.mid'))
            modified_midi.write(os.path.join(output_dir, f'modified_window_{i}_{label}.mid'))

if __name__ == "__main__":
    import argparse
    import pathlib
    
    parser = argparse.ArgumentParser(description='Windowed Piano Mistake Generator')
    parser.add_argument("input_midi_file", type=pathlib.Path, help="Path to input MIDI file")
    parser.add_argument("output_folder", type=pathlib.Path, help="Root output folder")
    parser.add_argument('--window_size', type=float, default=2.0, help='Window size in seconds')
    parser.add_argument('--window_overlap', type=float, default=0.5, help='Window overlap fraction')
    parser.add_argument('--variations', type=int, default=5, help='Variations per window')
    
    args = parser.parse_args()
    
    if not args.input_midi_file.exists():
        raise FileNotFoundError(f"Input MIDI file {args.input_midi_file} does not exist")
    
    generator = PianoMistakeGenerator(
        str(args.input_midi_file), 
        window_size=args.window_size, 
        window_overlap=args.window_overlap
    )
    
    windows, features, labels = generator.generate_windowed_dataset(
        num_samples_per_window=args.variations
    )
    
    generator.save_windowed_dataset(windows, features, labels, str(args.output_folder))
