import random
from typing import List, Tuple, Dict, Optional
from enum import Enum
import os
import pathlib
import copy
import numpy as np
import pretty_midi


class Difficulty(Enum):
    BEGINNER = "beginner"
    MEDIUM = "medium"
    ADVANCED = "advanced"


class ErrorType(Enum):
    WRONG_NOTE = "Wrong Note"
    MISSING_NOTE = "Missing Note"
    EXTRA_NOTE = "Extra Note"
    NO_ERROR = "No Error"


class PianoMistakeGenerator:
    def __init__(self, midi_file: str):
        """
        Mistake generator with a MIDI file.

        Args:
            midi_file (str): Path to the input MIDI file
        """
        try:
            self.midi = self._combine_instruments(midi_file)
            self.original_midi = copy.deepcopy(self.midi)
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")

    def _combine_instruments(self, midi_file: str) -> pretty_midi.PrettyMIDI:
        """
        Combines all instruments from a given MIDI file into a single instrument

        This method processes a MIDI file that may contain multiple instruments and combines their notes 
        into one instrument. This is useful when the goal is to manipulate the notes without 
        considering the individual instruments and aligning the NoteSequence visualization

        Args:
            midi_file (str): Path to the MIDI file

        Returns:
            pretty_midi.PrettyMIDI: Combined MIDI data
        """
        midi = pretty_midi.PrettyMIDI(midi_file)
        combined_instrument = pretty_midi.Instrument(program=0)

        for instrument in midi.instruments:
            for note in instrument.notes:
                combined_instrument.notes.append(note)

        combined_midi = pretty_midi.PrettyMIDI()
        combined_midi.instruments.append(combined_instrument)
        return combined_midi

    def _extract_notes(self) -> List[Dict]:
        """Extract notes with temporal feature"""
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

    def _add_recording_delay(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Simulate recording delay with realistic noise distribution"""
        modified = copy.deepcopy(midi)
        start_delay = np.random.normal(loc=0.05, scale=0.2, size=1)[0]
        start_delay = max(0.01, min(start_delay, 1.0))
        first_key_time = min(
            note.start for instrument in modified.instruments for note in instrument.notes)
        offset = start_delay - first_key_time

        for instrument in modified.instruments:
            for note in instrument.notes:
                note.start += offset
                note.end += offset
        end_delay = np.random.normal(loc=0.05, scale=0.2, size=1)[0]
        end_delay = max(0.01, min(end_delay, 1.0))
        modified._PrettyMIDI__end_time = modified.get_end_time() + end_delay
        return modified

    def _generate_mistake(self, midi: pretty_midi.PrettyMIDI, error_type: ErrorType, difficulty: Difficulty) -> pretty_midi.PrettyMIDI:
        """Generate mistakes"""
        modified = copy.deepcopy(midi)
        notes = modified.instruments[0].notes

        mistake_probabilities = {
            Difficulty.BEGINNER: 0.2,
            Difficulty.MEDIUM: 0.1,
            Difficulty.ADVANCED: 0.05
        }
        prob = mistake_probabilities[difficulty]

        if error_type == ErrorType.WRONG_NOTE:
            for note in notes:
                if random.random() < prob:
                    forward = random.choice([True, False])
                    note.pitch = self._find_neighbor_pitch(note.pitch, forward)

        elif error_type == ErrorType.MISSING_NOTE:
            modified.instruments[0].notes = [
                note for note in notes if random.random() >= prob
            ]

        elif error_type == ErrorType.EXTRA_NOTE:
            for note in notes:
                if random.random() < prob:
                    extra_note = copy.deepcopy(note)
                    extra_note.pitch += random.choice([-2, -1, 1, 2])
                    extra_note.start += random.uniform(-0.1, 0.1)
                    modified.instruments[0].notes.append(extra_note)

            modified.instruments[0].notes.sort(key=lambda n: n.start)

        return modified

    def generate_dataset(
        self,
        num_variations: int = 1000,
        difficulties: List[Difficulty] = None
    ) -> Tuple[List[pretty_midi.PrettyMIDI], List[str]]:
        """
        Generate a comprehensive dataset with nuanced mistake distribution.

        Args:
            num_variations (int): Total number of variations to generate
            difficulties (List[Difficulty]): Skill levels to include

        Returns:
            Tuple of generated MIDI files and their corresponding labels
        """
        if difficulties is None:
            difficulties = list(Difficulty)

        dataset = []
        labels = []

        distribution = {
            ErrorType.NO_ERROR: 0.25,
            ErrorType.WRONG_NOTE: 0.25,
            ErrorType.MISSING_NOTE: 0.25,
            ErrorType.EXTRA_NOTE: 0.25
        }

        # Generate samples for non-outlier error types
        for error_type, error_prob in distribution.items():
            samples_for_type = int(num_variations * error_prob)

            if error_type == ErrorType.NO_ERROR:
                # Generate clean samples
                for _ in range(samples_for_type):
                    clean_midi = self._add_recording_delay(
                        copy.deepcopy(self.midi))
                    dataset.append(clean_midi)
                    labels.append(ErrorType.NO_ERROR.value)
            else:
                # Generate single error and combined error samples
                single_samples = int(samples_for_type * 0.35)
                combined_samples = int(samples_for_type * 0.65)

                for is_combined in [False, True]:
                    target_samples = combined_samples if is_combined else single_samples

                    for _ in range(target_samples):
                        difficulty = random.choice(difficulties)

                        # Generate primary mistake
                        modified_midi = self._generate_mistake(
                            copy.deepcopy(self.midi),
                            error_type,
                            difficulty
                        )

                        # For combined errors, add another mistake type
                        if is_combined:
                            other_error_types = [
                                et for et in ErrorType
                                if et not in [ErrorType.NO_ERROR, error_type]
                            ]
                            secondary_error = random.choice(other_error_types)
                            modified_midi = self._generate_mistake(
                                modified_midi,
                                secondary_error,
                                difficulty
                            )
                        modified_midi = self._add_recording_delay(
                            modified_midi)
                        dataset.append(modified_midi)
                        labels.append(f"{error_type.value}{'_Combined' if is_combined else ''}")

        # Add outlier data with 0.01 of the total dataset size
        outlier_samples_needed = int(num_variations * 0.01)
        for _ in range(outlier_samples_needed):
            modified_midi = copy.deepcopy(self.midi)
            for instrument in modified_midi.instruments:
                for note in instrument.notes:
                    note.pitch += random.randint(-12, 12)
                    note.start += random.uniform(-0.5, 0.5)
                    note.end += random.uniform(-0.5, 0.5)

            noise_label = random.choice([ErrorType.NO_ERROR.value, ErrorType.EXTRA_NOTE.value,
                                        ErrorType.WRONG_NOTE.value, ErrorType.MISSING_NOTE.value])
            dataset.append(modified_midi)
            labels.append(noise_label + "_outlier")

        return dataset, labels

    def save_dataset(
        self,
        dataset: List[pretty_midi.PrettyMIDI],
        labels: List[str],
        output_dir: pathlib.Path
    ):
        """
        Save generated dataset to MIDI files.

        Args:
            dataset (List[pretty_midi.PrettyMIDI]): Generated MIDI files
            labels (List[str]): Corresponding labels
            output_dir (pathlib.Path): Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (midi, label) in enumerate(zip(dataset, labels)):
            print(label)
            label_dir = output_dir / label.split('_')[0]
            label_dir.mkdir(parents=True, exist_ok=True)
            
            filename = label_dir / f"sample_{idx:04d}.mid"
            midi.write(str(filename))
            print(f"Saved: {filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced Piano Mistake Generator')
    parser.add_argument("input_midi_file", type=pathlib.Path,
                        help="Path to input MIDI file")
    parser.add_argument("output_folder", type=pathlib.Path,
                        help="Root output folder")
    parser.add_argument('--variations', type=int,
                        default=1000, help='Number of variations')

    args = parser.parse_args()

    if not args.input_midi_file.exists():
        raise FileNotFoundError(
            f"Input MIDI file {args.input_midi_file} does not exist")

    generator = PianoMistakeGenerator(str(args.input_midi_file))

    difficulties = [Difficulty.BEGINNER,
                    Difficulty.MEDIUM, Difficulty.ADVANCED]

    dataset, labels = generator.generate_dataset(
        num_variations=args.variations,
        difficulties=difficulties
    )

    generator.save_dataset(dataset, labels, args.output_folder)


if __name__ == "__main__":
    main()
