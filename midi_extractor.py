import pretty_midi
import numpy as np
import scipy.stats as stats
import pathlib
import csv


class MIDIFeatureExtractor:
    def __init__(self, midi_path):
        """
        Initialize the feature extractor with a MIDI file

        Args:
            midi_path (str): Path to the MIDI file
        """
        self.midi_data = pretty_midi.PrettyMIDI(midi_path)

    def extract_pitch_features(self):
        """
        Extract pitch-related features
        
        This method analyzes the pitch characteristics across all instruments in the MIDI file
        Pitch features provide insights into the melodic and tonal properties of the music

        Returns:
            dict: Pitch-related features
            
        Example:
            >>> extractor = MIDIFeatureExtractor('song.mid')
            >>> pitch_features = extractor.extract_pitch_features()
            >>> print(pitch_features['pitch_mean'])
            Outputs the average pitch of the MIDI file
        """
        all_notes = []
        for instrument in self.midi_data.instruments:
            all_notes.extend(instrument.notes)

        pitches = [note.pitch for note in all_notes]
        return {
            'pitch_mean': np.mean(pitches),
            'pitch_std': np.std(pitches),
            'pitch_range': max(pitches) - min(pitches),
            'pitch_median': np.median(pitches),
            'pitch_skewness': stats.skew(pitches),
            'unique_pitch_count': len(set(pitches)),
        }

    def extract_rhythm_features(self):
        """
        Extract rhythm and timing features
        
        This method analyzes the temporal characteristics of notes, providing insights 
        into the rhythmic structure of the musical piece

        Returns:
            dict: Rhythm-related features
            
        Example:
            >>> extractor = MIDIFeatureExtractor('rhythm_track.mid')
            >>> rhythm_features = extractor.extract_rhythm_features()
            >>> print(rhythm_features['note_density'])
            Outputs the number of notes per second in the MIDI file
        """
        all_note_starts = []
        for instrument in self.midi_data.instruments:
            all_note_starts.extend([note.start for note in instrument.notes])

        all_note_starts.sort()
        inter_onset_intervals = np.diff(all_note_starts)

        return {
            'avg_note_duration': np.mean([note.end - note.start for instrument in self.midi_data.instruments for note in instrument.notes]),
            'note_density': len(all_note_starts) / self.midi_data.get_end_time(),
            'ioi_mean': np.mean(inter_onset_intervals),
            'ioi_std': np.std(inter_onset_intervals),
            'rhythm_entropy': stats.entropy(inter_onset_intervals) if len(inter_onset_intervals) > 0 else 0,
        }

    def extract_harmonic_features(self):
        """
        Extract harmonic and tonal features
        
        This method analyzes the pitch class distribution to capture harmonic characteristics, 
        providing insights into the tonal structure of the musical piece

        Returns:
            dict: Harmonic-related features
            
        Example:
            >>> extractor = MIDIFeatureExtractor('harmony_track.mid')
            >>> harmonic_features = extractor.extract_harmonic_features()
            >>> print(harmonic_features['dominant_pitch_class'])
            Outputs the most common pitch class (0-11 representing C-B) 
        """
        pitch_classes = [note.pitch % 12 for instrument in self.midi_data.instruments for note in instrument.notes]
        pitch_class_hist, _ = np.histogram(pitch_classes, bins=12)
        normalized_hist = pitch_class_hist / np.sum(pitch_class_hist) if np.sum(pitch_class_hist) > 0 else np.zeros(12)

        return {
            'pitch_class_entropy': stats.entropy(normalized_hist) if np.sum(normalized_hist) > 0 else 0,
            'dominant_pitch_class': np.argmax(pitch_class_hist),
        }

    def extract_all_features(self):
        """
        Combine all feature extraction methods

        Returns:
            dict: Comprehensive MIDI features
            
        Example:
            >>> extractor = MIDIFeatureExtractor('complete_track.mid')
            >>> full_features = extractor.extract_all_features()
            >>> print(full_features)
            Outputs a comprehensive dictionary of musical features
        """
        return {
            **self.extract_pitch_features(),
            **self.extract_rhythm_features(),
            **self.extract_harmonic_features(),
        }


def preprocess_features(features_dict):
    """
    Normalize and prepare features for machine learning

    Args:
        features_dict (dict): Raw feature dictionary

    Returns:
        numpy.ndarray: Normalized feature vector
    """
    feature_vector = np.array(list(features_dict.values()))
    return (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)


def extract_features_from_folder(input_folder, output_csv=None):
    """
    Extract features from multiple MIDI files in a labeled directory structure

    Args:
        input_folder (pathlib.Path): Path to the root folder containing MIDI files in subdirectories
        output_csv (str): Path to save extracted features in CSV format. If None, no CSV is saved

    Returns:
        List[dict]: List of feature dictionaries with labels
    """
    input_folder = pathlib.Path(input_folder)
    features_list = []

    for label_folder in input_folder.iterdir():
        if not label_folder.is_dir():
            continue

        label = label_folder.name
        for midi_file in label_folder.glob("*.mid"):
            extractor = MIDIFeatureExtractor(str(midi_file))
            features = extractor.extract_all_features()
            features['label'] = label
            features_list.append(features)

    # Save to CSV if specified
    if output_csv:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=features_list[0].keys())
            writer.writeheader()
            writer.writerows(features_list)

    return features_list


def extract_features_single_file(midi_file):
    """
    Extract features from a single MIDI file

    Args:
        midi_file (str): Path to the MIDI file

    Returns:
        dict: Extracted features
    """
    extractor = MIDIFeatureExtractor(midi_file)
    return extractor.extract_all_features()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Feature Extractor")
    parser.add_argument("--input", required=True, help="Path to the input MIDI file or folder")
    parser.add_argument("--output_csv", help="Path to save features in CSV format")
    parser.add_argument("--single", action="store_true", help="Use single MIDI file instead of folder")

    args = parser.parse_args()

    if args.single:
        features = extract_features_single_file(args.input)
        print("Extracted Features:", features)
    else:
        features = extract_features_from_folder(args.input, args.output_csv)
        print(f"Extracted {len(features)} feature sets from folder.")
