import pretty_midi
import numpy as np
import pandas as pd
import copy
import os
import pickle
import tensorflow as tf
from typing import List, Dict

class MIDIErrorPredictor:
    def __init__(self, model_path: str, scaler_path: str, window_size: float = 2.0, window_overlap: float = 0.5):
        """
        Initialize the MIDI error predictor
        
        Args:
            model_path (str): Path to the saved Keras model
            scaler_path (str): Path to the saved scaler
            window_size (float): Size of the window in seconds
            window_overlap (float): Overlap between windows
        """
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)
        
        self.window_size = window_size
        self.window_overlap = window_overlap
        
        # Load label mapping
        self.label_mapping = {
            0: 'no_error',
            1: 'forward_backward_insertion',
            2: 'mistouch',
            3: 'note_substitution',
            4: 'dragging_note'
        }
    
    def _extract_windowed_notes(self, midi: pretty_midi.PrettyMIDI, start_time: float, end_time: float) -> List[Dict]:
        """
        Extract notes within a specific time window
        
        Args:
            midi (pretty_midi.PrettyMIDI): Input MIDI object
            start_time (float): Start of the window
            end_time (float): End of the window
        
        Returns:
            List of notes within the window
        """
        windowed_notes = []
        for instrument in midi.instruments:
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
    
    def _compute_window_features(self, window_notes: List[Dict]) -> Dict:
        """
        Compute features for a specific window
        
        Args:
            window_notes (List[Dict]): Notes within the window
        
        Returns:
            Dict of computed features
        """
        if not window_notes:
            return None
        
        # Pitch features
        pitches = [note['pitch'] for note in window_notes]
        
        # Timing features
        starts = [note['start'] for note in window_notes]
        
        # Velocity features
        velocities = [note['velocity'] for note in window_notes]
        
        # Compute features similar to the generator
        features = {
            'pitch_change_count': len(set(pitches)),
            'timing_shift_magnitude': np.std(starts),
            'velocity_variation': np.std(velocities),
            'note_count_change': len(window_notes),
            'note_density_change': len(window_notes) / (window_notes[-1]['end'] - window_notes[0]['start']),
            'pitch_change_ratio': len(set(pitches)) / len(window_notes),
            'timing_instability': np.std(starts) * len(window_notes),
            'velocity_volatility': np.std(velocities) * len(window_notes),
            'notes_per_second': len(window_notes) / (window_notes[-1]['end'] - window_notes[0]['start'])
        }
        
        return features
    
    def predict_midi_errors(self, midi_file: str) -> List[Dict]:
        """
        Predict errors for each window in the MIDI file
        
        Args:
            midi_file (str): Path to the MIDI file
        
        Returns:
            List of dictionaries with window predictions
        """
        # Load MIDI
        try:
            midi = pretty_midi.PrettyMIDI(midi_file)
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")
        
        # Determine window parameters
        total_duration = midi.get_end_time()
        
        # Compute window start points with overlap
        window_step = self.window_size * (1 - self.window_overlap)
        window_starts = np.arange(0, total_duration - self.window_size, window_step)
        
        # Predictions storage
        window_predictions = []
        
        for start_time in window_starts:
            end_time = start_time + self.window_size
            
            # Extract window notes
            window_notes = self._extract_windowed_notes(midi, start_time, end_time)
            
            # Skip if no notes in window
            if not window_notes:
                continue
            
            # Compute window features
            window_features = self._compute_window_features(window_notes)
            
            if window_features is None:
                continue
            
            # Prepare features for prediction
            X = pd.DataFrame([window_features])
            
            # Select the same features as training
            feature_columns = [
                'pitch_change_count', 
                'timing_shift_magnitude', 
                'velocity_variation', 
                'note_count_change', 
                'note_density_change',
                'pitch_change_ratio', 
                'timing_instability', 
                'velocity_volatility', 
                'notes_per_second'
            ]
            
            # Normalize features
            X_scaled = self.scaler.transform(X[feature_columns])
            
            # Predict
            prediction = self.model.predict(X_scaled)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Store prediction
            window_predictions.append({
                'window_start': start_time,
                'window_end': end_time,
                'label': self.label_mapping[predicted_class],
                'confidence': float(confidence)
            })
        
        return window_predictions

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MIDI Error Predictor')
    parser.add_argument('midi_file', type=str, help='Path to input MIDI file')
    parser.add_argument('--model', type=str, default='midi_error_detection_model_v4.h5', 
                        help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', 
                        help='Path to scaler')
    parser.add_argument('--output', type=str, default='midi_error_predictions.csv', 
                        help='Output CSV file for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MIDIErrorPredictor(args.model, args.scaler)
    
    # Predict errors
    predictions = predictor.predict_midi_errors(args.midi_file)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(predictions)
    df.to_csv(args.output, index=False)
    
    # Print predictions
    for pred in predictions:
        print(f"Window: {pred['window_start']} - {pred['window_end']} s | "
              f"Label: {pred['label']} | "
              f"Confidence: {pred['confidence']:.2f}")

if __name__ == "__main__":
    main()
