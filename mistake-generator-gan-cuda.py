import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import librosa
import soundfile as sf
from tqdm.auto import tqdm 

class SingleSongMistakeGAN:
    def __init__(self, original_performance, latent_dim=100, batch_size=32):
        """
        Initialize GAN for a single song performance with
        """
        # Determine the device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.original_performance = original_performance.to(self.device)
        
        self.latent_dim = latent_dim  
        self.batch_size = batch_size
        self.pitch_range = original_performance.shape[0]
        self.time_steps = original_performance.shape[1]
        
        self.generator = self._create_generator(self.latent_dim).to(self.device)
        self.discriminator = self._create_discriminator().to(self.device)
        
        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Mistake strategies specific to the song
        self.mistake_strategies = [
            self._wrong_note_mistake,
            self._timing_mistake,
            self._rhythm_mistake
        ]
    
    def _create_generator(self, latent_dim):
        """Create a generator network specific to the song's characteristics"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.pitch_range * self.time_steps),
            nn.Tanh()
        )
    
    def _create_discriminator(self):
        """Create a discriminator to detect how close the generated performance is to the original"""
        return nn.Sequential(
            nn.Linear(self.pitch_range * self.time_steps, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _wrong_note_mistake(self, performance):
        """Generate wrong note mistakes based on the original performance"""
        performance = performance.clone()
        # Identify most common notes in the original performance
        mistake_prob = 0.1
        noise = torch.rand_like(performance) < mistake_prob
        
        # Replace with nearby notes or slightly different pitches
        wrong_notes = performance.clone()
        for i in range(performance.shape[0]):
            for j in range(performance.shape[1]):
                if noise[i, j]:
                    shift = np.random.choice([-1, 1]) * np.random.randint(1, 3)
                    wrong_notes[i, j] = torch.clamp(
                        performance[i, j] + shift, 
                        min=0, 
                        max=1
                    )
        return wrong_notes
    
    def _timing_mistake(self, performance):
        """Create timing-related mistakes"""
        performance = performance.clone()
        shifted = performance.clone()
        shift_amount = np.random.randint(1, 5)
        shifted = torch.roll(shifted, shifts=shift_amount, dims=1)
        
        # Add randomness to preserve musical context
        noise = torch.rand_like(shifted) * 0.1
        return shifted * (1 + noise)
    
    def _rhythm_mistake(self, performance):
        """Disrupt rhythm while maintaining some original structure"""
        performance = performance.clone()
        disrupted = performance.clone()
        mask = torch.rand_like(disrupted) < 0.2
        disrupted[mask] *= 0.5  
        return disrupted
    
    def train(self, epochs=1000):
        """
        Train the GAN to generate plausible performance mistakes
        
        Focuses on creating variations that are close to but not identical 
        to the original performance
        """
        torch.autograd.set_detect_anomaly(True)
        dataset = self._prepare_dataset()

        epoch_progress = tqdm(range(epochs), desc="Training Epochs", position=0)
        
        for epoch in epoch_progress:
            batch_progress = tqdm(dataset, desc=f"Epoch {epoch+1}", position=1, leave=False)
            
            epoch_d_losses = []
            epoch_g_losses = []
            
            for batch_real in batch_progress:
                # Move batch to device and ensure it requires gradient
                batch_real = batch_real.to(self.device).requires_grad_(True)
                
                # Prepare real and fake samples
                batch_size = batch_real.size(0)
                
                # Train Discriminator
                self.discriminator.zero_grad()
                
                # Real samples
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output = self.discriminator(batch_real.view(batch_size, -1))
                
                # Fake samples
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_performances = self.generator(z).view(batch_size, self.pitch_range, self.time_steps)
                
                # Apply random mistake to some fake performances
                for i in range(fake_performances.size(0)):
                    if torch.rand(1).item() < 0.5:
                        mistake_func = np.random.choice(self.mistake_strategies)
                        fake_performances[i] = mistake_func(fake_performances[i])
                
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                fake_output = self.discriminator(fake_performances.view(batch_size, -1).detach())
                
                # Discriminator loss
                d_loss = self.criterion(real_output, real_labels) + \
                         self.criterion(fake_output, fake_labels)
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.generator.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_performances = self.generator(z).view(batch_size, self.pitch_range, self.time_steps)
                
                gen_output = self.discriminator(fake_performances.view(batch_size, -1))
                g_loss = self.criterion(gen_output, torch.ones(batch_size, 1).to(self.device))
                
                g_loss.backward()
                self.g_optimizer.step()

                # Store losses for averaging
                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())
                
                # Update batch progress bar
                batch_progress.set_postfix({
                    'D Loss': np.mean(epoch_d_losses),
                    'G Loss': np.mean(epoch_g_losses)
                })
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                'Avg D Loss': np.mean(epoch_d_losses),
                'Avg G Loss': np.mean(epoch_g_losses)
            })
        
        epoch_progress.close()
        
        return
                
    def _prepare_dataset(self):
        """
        Prepare dataset with batches
        
        Returns:
        - DataLoader-like iterable of batches
        """
        # Reshape original performance to match batch requirements
        performance_flat = self.original_performance.view(1, -1)
        
        # Create batches by repeating and adding noise
        batches = []
        for _ in range(0, 1000, self.batch_size):
            batch = performance_flat.repeat(self.batch_size, 1)
            batch += torch.randn_like(batch) * 0.1  
            batches.append(batch.to(self.device))
        
        return batches
    
    def generate_mistake_variations(self, num_variations=5):
        """
        Generate multiple mistake variations of the original performance
        
        Returns:
        - List of performances with different types of mistakes
        """
        variations = []
        for _ in range(num_variations):
            # Generate base variation
            z = torch.randn(1, self.latent_dim).to(self.device)
            generated_performance = self.generator(z).view(
                self.pitch_range, self.time_steps
            )
            
            # Apply random mistake strategy
            mistake_func = np.random.choice(self.mistake_strategies)
            mistaken_performance = mistake_func(generated_performance)
            
            variations.append(mistaken_performance.cpu())  
        
        return variations

    def save_piano_roll_as_audio(self, piano_roll, output_path, sr=22050, hop_length=512):
        """
        Convert piano roll back to audio for listening
        
        Parameters:
        - piano_roll: Tensor representation of piano roll
        - output_path: Path to save the output WAV file
        """
        if piano_roll.is_cuda:
            piano_roll = piano_roll.cpu()
        
        # Convert tensor to numpy
        piano_roll_np = piano_roll.numpy()
        spec = librosa.db_to_power(piano_roll_np)
        y_reconstructed = librosa.griffinlim(spec, hop_length=hop_length)
        
        sf.write(output_path, y_reconstructed, sr)

def audio_to_piano_roll(audio_path, sr=22050, hop_length=512, n_bins=88):
    """
    Convert audio file to piano roll representation
    
    Parameters:
    - audio_path: Path to the WAV file
    - sr: Sample rate for loading
    - hop_length: Hop length for STFT
    - n_bins: Number of pitch classes (standard piano range)
    
    Returns:
    - Piano roll tensor (pitch_classes x time_steps)
    """
    # Load audio file
    y, sample_rate = librosa.load(audio_path, sr=sr)
    spec = np.abs(librosa.stft(y, hop_length=hop_length))
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sample_rate, 
        n_mels=n_bins, 
        hop_length=hop_length
    )
    
    # Normalize
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    piano_roll = torch.tensor(mel_spec, dtype=torch.float32)
    
    return piano_roll

def main():
    from tqdm.notebook import tqdm  
    from IPython.display import display, HTML
    display(HTML('<h2>🎹 Piano Mistake Generation 🎵</h2>'))

    # Check and print CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    audio_path = '/content/input/twinkle-twinkle-little-star.wav' 

    # Convert audio to piano roll with progress bar
    print("Converting audio to piano roll...")
    original_performance = audio_to_piano_roll(audio_path)

    # Initialize GAN for the specific song
    print("Initializing Mistake GAN...")
    song_mistake_gan = SingleSongMistakeGAN(original_performance, batch_size=32)
    
    # Train the GAN with progress tracking
    print("Training GAN to generate musical mistakes...")
    song_mistake_gan.train(epochs=500)
    
    # Generate mistake variations with progress bar
    print("Generating musical mistake variations...")
    mistake_variations = song_mistake_gan.generate_mistake_variations(num_variations=3)
    
    # Create output folder
    output_folder = "GAN_output"
    os.makedirs(output_folder, exist_ok=True)

    # Save variations with progress bar
    print("Saving mistake variations as audio...")
    for i, variation in enumerate(tqdm(mistake_variations, desc="Saving Variations")):
        output_path = os.path.join(output_folder, f'mistake_variation_{i+1}.wav')
        song_mistake_gan.save_piano_roll_as_audio(variation, output_path)
        print(f"💾 Saved mistake variation {i+1} to {output_path}")
    
    # Final notification
    display(HTML('<h3>✨ Musical Mistake Generation Complete! ✨</h3>'))

if __name__ == "__main__":
    main()
