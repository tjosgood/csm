import io
import os
import re
import time

import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
import soundfile as sf
from generator import load_csm_1b, Segment
from dataclasses import dataclass
import torchaudio.transforms
import torch.nn.functional as F
from scipy.signal import medfilt
import numpy as np

#os.environ["NO_TORCH_COMPILE"] = "1"

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            " People may not have found another way or that is their way and that is their method, and so it actually is interesting hearing it from your perspectve"
           # "has created characters like black adder, Jonny English, and of course, Mr bean. he’s returning to our screens in johnny English strikes again"
        ),
        "audio": "dj.wav"
    },
    "conversational_b": {
        "text": (
            "Hi, I'm Ant, your AI assistant here in anthology, here to help you explore the platform, answer your questions, and share insights about the music industry."
          #  """essentially prize money for all these incredible innovators who are really doing stuff that’s actually going to, genuinely, like, help protect and repair the planet like there’s an amazing woman called charlotte who makes these sustainable stoves that stop, you know, like toxic"""
       # "put together and they, you know, their all in their thirties and forties, they’ve all been around the block, there’s no children involved and they all think that they lost the love of their life and its about weather they can find their way back"
        ),
        "audio": "ant.wav"
    }
}

_script = """
[S1] Welcome to Exploring Innovation in Music! Today: AI. Not the scary kind, but AI that empowers.
[S1] We're diving into Anthology, a new platform built by industry insiders.
[S1] And to show you how it works, we’ve got a unique guest: Ant, Anthology’s AI Assistant!
[S1] Ant, welcome! Ready to introduce yourself and Anthology?
[S2] Thanks so much! As an AI, I'm designed to assist, and I'm really pleased to be here and chat about how we're aiming to make things a bit easier and more effective for everyone in the independent sector.
[S1] Great to have you. Now, Ant, there's AI concern, right? Especially with music creation. Anthology’s different?
[S2] Absolutely.
[S1] You had a powerful way of putting this when Anthology launched?
[S2] We did! It’s fair folks are wary. But AI, applied right, is a game-changer. My focus? Not replacing creativity, but supporting labels, artists, rights holders. Think of me as your dedicated industry guide within Anthology.
[S1] A guide for what, exactly?
[S2] Managing catalogues, global rights, Content ID, Neighbouring Rights – unlocking your music’s full potential. I use industry knowledge to simplify tasks, cut admin, so you can focus on the music.
[S1] So, Anthology is the platform, you're the AI helper inside. Let's get practical. An artist just wants their record heard. How do you help?
[S2] Right! For artists, it’s discovery. Anthology gets tracks to global platforms.
[S1] And your AI magic?
[S2] I guide them through Pitching in Anthology – telling Spotify, Apple Music about their release. I help them craft that perfect pitch.
[S1] Smart. What else?
[S2] Understanding data! Insights, Trends, TikTok – I show where music’s hot, explain metrics like 'Creations,' so they know their audience. Plus, Slinky pages, metadata for voice search? I’m right there.
[S1] So, data into action. Cool. What about a label with a huge back catalogue?
[S2] Catalogue managers! Love 'em. Anthology is their solid base. My AI assistance is key with that much data.
[S1] How so?
[S2] It's a central hub. I help navigate it, understand financial reports, revenue streams. The AI crunches those numbers for Royalty Analytics – spotting trends, valuable assets.
[S1] Clarity for big operations. And the DIY 'tenant' user, doing it all themselves?
[S2] The indie powerhouse! Anthology is their end-to-end toolkit.
[S1] And your role?
[S2] Their expert guide! Step-by-step release creation – audio, artwork, metadata, release dates. Plus, interpreting all that Trends and Royalty data for smart business decisions. Problem solver too, if things get tricky.
[S1] You mentioned being a "tireless industry guide." What’s that look like behind the scenes? Beyond the interface?
[S2] Great question! I'm always on, always working. Scanning your catalogue data, the wider digital landscape.
[S1] So, you spot things?
[S2] Instantly! A big playlist add? Sudden stream surge? I can ping you directly – Teams, WhatsApp. You won’t miss a beat.
[S1] Instant alerts! Love it.
[S2] And admin! Missing royalty files? Exchange rate issues? Low physical stock? I flag it, point you to solutions.
[S1] And for strategy?
[S2] Definitely. I can help with artist audits, even brainstorm marketing ideas based on solid performance data. Intelligent assistance, big picture to tiny details.
[S1] That’s seriously proactive! Now, Anthology pulls together so many music biz bits: deliveries, streaming, physical, D2C, rights… That’s a mountain! How do you help consolidate that fragmentation?
[S2] You nailed it! The biz is fragmented. Anthology is the central hub that tames the chaos.
[S1] One-stop shop, basically?
[S2] Exactly! Streaming trends, physical stock, D2C sales – all in one place. My role? Help you make sense of this unified data. Digital Deliveries, complex Rights Management – Neighbouring Rights, Content ID, mechanicals, the works.
[S1] All under one roof. Sounds essential.
[S2] It is. No more juggling a dozen systems. Teams work smarter, make decisions with the full picture.
[S1] Big time-saver, fewer errors. You mentioned task tracking too, right?
[S2] Yep! An integrated Ticketing System in Anthology.
[S1] How does that help smooth things out?
[S2] Raise a ticket for any issue, question, query. I guide users on the best way. It centralises comms and tasks, right where your assets live. Super efficient for teams of any size.
[S1] And you’re always learning, evolving with the industry?
[S2] Constantly! The music world never stops, and neither do I. My knowledge isn't static. I adapt to new data, platform updates, user interactions.
[S1] So, always current, always relevant?
[S2] That's the goal. Helping users leverage tech to amplify their creative work. Remember, this supports the business side, freeing up artists and labels for their true passion: the music.
[S1] Ant, this has been genuinely eye-opening. It’s crystal clear that Anthology, with you steering the AI, is a genuine partner for the independent music sector. Thanks so much for joining us!
[S2] My pleasure! Always happy to show how tech can be a positive force, not a scary one.
[S1] And thank you to our listeners for tuning in. You can find out more about Anthology and Ant over at https://anthology.to/. Join us next time on The Future of Music!
"""


script = """
[S1] Welcome to Exploring Innovation in Music! Today: AI. Not the scary kind, but AI that empowers.
[S1] We're diving into Anthology, a new platform built by industry insiders.
[S1] And to show you how it works, we’ve got a unique guest: Ant, Anthology’s AI Assistant!
[S1] Ant, welcome! Ready to introduce yourself and Anthology?
[S2] Thanks so much! As an AI, I'm designed to assist, and I'm really pleased to be here and chat about how we're aiming to make things a bit easier and more effective for everyone in the independent sector.
[S1] Great to have you. Now, Ant, there's AI concern, right? Especially with music creation. Anthology’s different?
"""

def peak_normalize_tensor(waveform, peak=0.90):
    """Scale waveform to given peak (default 0.99 to avoid clipping)."""
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform * (peak / max_val)
    return waveform


def remove_clicks_multichannel(audio_np, threshold=0.5, kernel_size=3):
    # audio_np shape: [channels, time]
    cleaned = np.copy(audio_np)
    for ch in range(cleaned.shape[0]):
        diff = np.abs(np.diff(cleaned[ch]))
        click_indices = np.where(diff > threshold)[0] + 1
        for idx in click_indices:
            start = max(0, idx - kernel_size)
            end = min(cleaned.shape[1], idx + kernel_size + 1)
            cleaned[ch, idx] = np.median(cleaned[ch, start:end])
    return cleaned

def save_tensor_as_mp3(audio_tensor: torch.Tensor, sample_rate: int, output_path: str):
    if audio_tensor.ndim != 2:
        raise ValueError("Expected audio tensor with shape [channels, time]")

    audio_np = audio_tensor.cpu().numpy().astype(np.float32)
    audio_np = remove_clicks_multichannel(audio_np)

    # Transpose to [time, channels] for soundfile
    audio_np = audio_np.T

    # Save as WAV to a buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    buffer.seek(0)

    # Convert to MP3 using pydub
    audio_segment = AudioSegment.from_file(buffer, format="wav")
    audio_segment.export(output_path, format="mp3")

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor


def concatenate_with_silence(segments, sample_rate, silence_duration_sec=0.02):
    # Determine shape from first segment
    channels = segments[0].shape[0]
    silence_samples = int(sample_rate * silence_duration_sec)

    # Silence shape: [channels, silence_samples]
    silence = torch.zeros((channels, silence_samples), dtype=segments[0].dtype)

    output = []
    for i, seg in enumerate(segments):
        output.append(seg)
        if i < len(segments) - 1:
            output.append(silence)

    return torch.cat(output, dim=-1)  # Concatenate along time axis

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=peak_normalize_tensor(audio_tensor))

def parse_script(script_text):
    segments = []
    pattern = r"\[(S\d+)\]\s*(.*?)(?=\[\s*S\d+\s*\]|$)"
    matches = re.finditer(pattern, script_text, re.DOTALL)
    for match in matches:
        speaker_tag = match.group(1).strip()
        text_segment = match.group(2).strip()
        if text_segment:
            segments.append((speaker_tag, text_segment))
    return segments


def main():
    # Select the best available device, skipping MPS due to float64 limitations

    device = "cuda"
    print(f"Using device: {device}")

    # Load model
    generator = load_csm_1b(device)

    # Prepare prompts
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        generator.sample_rate
    )

    # Generate conversation


    full_conversation = [ {"text": x[1], "speaker_id": int(x[0].strip("[").strip("]").strip("S"))-1} for x in parse_script(script)]

    def clean(s):
        s=   s.replace(": ",", ").replace("; ",", ")
        s=  s.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")

        return s

    conversation=[]
    for c in full_conversation:
        words = c["text"].split()
        subm = ""
        for w in words:
            subm += (w+" ")
            if len(subm.split()) > 100:
                conversation.append(c | {"text": clean(subm) })
                subm = ""
        if subm:
            conversation.append(c | {"text":clean(subm) })

    # Generate each utterance
    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    first_gen = True

    for  n,utterance in tqdm.tqdm( list(enumerate( conversation))):
        print(f"Generating: {utterance['speaker_id']}: ({n+1}/{len(conversation)}) {utterance['text']}")

        my_past = [x for x in generated_segments if x.speaker ==utterance['speaker_id'] ]
        pr =  [x for x in prompt_segments if x.speaker ==utterance['speaker_id'] ]

        #context = prompt_segments+ generated_segments[-1:]
        context = pr+ my_past[-1:]

        audio_tensor = generator.generate(
            text=utterance['text'],
            speaker=utterance['speaker_id'],
            context=context,
            max_audio_length_ms=20_000,
            temperature=0.7,
            seed=954634 if first_gen else None
        )
        first_gen =False

        audio_tensor = peak_normalize_tensor(audio_tensor)
        generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))

        s = time.time()

        all_audio = concatenate_with_silence([seg.audio.unsqueeze(0).cpu() for seg in generated_segments],generator.sample_rate,0.04)
        save_tensor_as_mp3(all_audio,generator.sample_rate,  "full_conversation.mp3")

        print(f"save in {time.time()-s}")
    print("Successfully generated full_conversation.wav")

if __name__ == "__main__":
    main()
