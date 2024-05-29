use std::collections::HashMap;
use std::error::Error;

lazy_static! {
    static ref NOTE_MAP: HashMap<char, i32> = {
        let mut m = HashMap::new();
        m.insert('C', 0);
        m.insert('D', 2);
        m.insert('E', 4);
        m.insert('F', 5);
        m.insert('G', 7);
        m.insert('A', 9);
        m.insert('B', 11);
        m.insert('c', 0);
        m.insert('d', 2);
        m.insert('e', 4);
        m.insert('f', 5);
        m.insert('g', 7);
        m.insert('a', 9);
        m.insert('b', 11);
        m
    };
}

lazy_static! {
    static ref ACC_MAP: HashMap<char, i32> = {
        let mut m = HashMap::new();
        m.insert('#', 1);
        m.insert('♯', 1);
        m.insert('b', -1);
        m.insert('♭', -1);
        m.insert('!', -1);
        m.insert('𝄪', 2);
        m.insert('𝄫', -2);
        m.insert('♮', 0);
        m
    };
}

pub fn hz_to_note(frequency: f64) -> String {
    midi_to_note(hz_to_midi(frequency))
}

pub fn hz_to_note_vec(frequencies: &[f64]) -> Vec<String> {
    frequencies.iter().map(|frequency| hz_to_note(*frequency)).collect()
}

pub fn hz_to_midi(frequency: f64) -> f64 {
    12.0 * (frequency.log2() - 440.0_f64.log2()) + 69.0
}

pub fn hz_to_midi_vec(frequencies: &[f64]) -> Vec<f64> {
    frequencies
        .iter()
        .map(|frequency| hz_to_midi(*frequency))
        .collect()
}

pub fn midi_to_hz(midi: f64) -> f64 {
    440.0 * (2.0_f64.powf((midi - 69.0) / 12.0))
}

pub fn midi_to_hz_vec(midis: &[f64]) -> Vec<f64> {
    midis.iter().map(|midi| midi_to_hz(*midi)).collect()
}

pub fn midi_to_note(midi: f64) -> String {
    const MIDI_MAP: [&str; 12] = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

    let midi = midi.round();
    let note = MIDI_MAP[midi.rem_euclid(12.0) as usize].to_string();
    let octave = ((midi / 12.0) as i32 - if midi < 0.0 { 2 } else { 1 }).to_string();

    note + &octave
}

pub fn midi_to_note_vec(midis: &[f64]) -> Vec<String> {
    midis.iter().map(|midi| midi_to_note(*midi)).collect()
}

pub fn note_to_hz(note: &str) -> Result<f64, Box<dyn Error>> {
    Ok(midi_to_hz(note_to_midi(note)?))
}

pub fn note_to_hz_vec(notes: &[&str]) -> Result<Vec<f64>, Box<dyn Error>> {
    notes.iter().map(|note| note_to_hz(note)).collect()
}

pub fn note_to_midi(note: &str) -> Result<f64, Box<dyn Error>> {
    let mut note_iter = note.chars();

    let letter = match note_iter.next() {
        Some(ch) => match NOTE_MAP.get(&ch) {
            Some(value) => *value,
            None => return Err("Invalid note format".into()),
        }
        None => return Err("Note is empty".into()),
    };

    let (accent, note_len) = match note_iter.next() {
        Some(ch) => match ACC_MAP.get(&ch) {
            Some(value) => (*value, 2),
            None => (0, 1),
        }
        None => return Ok(letter as f64 + 12.0),
    };

    let octave: i32 = note.chars().skip(note_len).collect::<String>().parse()?;

    Ok((letter + accent) as f64 + ((octave + 1) * 12) as f64)
}

pub fn note_to_midi_vec(notes: &[&str]) -> Result<Vec<f64>, Box<dyn Error>> {
    notes.iter().map(|note| note_to_midi(note)).collect()
}

pub fn hz_to_mel_htk(frequency: f64) -> f64 {
    2595.0 * (1.0 + frequency / 700.0).log10()
}

pub fn hz_to_mel_htk_vec(frequencies: &[f64]) -> Vec<f64> {
    frequencies
        .iter()
        .map(|frequency| hz_to_mel_htk(*frequency))
        .collect()
}

pub fn hz_to_mel_slaney(frequency: f64) -> f64 {
    if frequency < 1000.0 {
        frequency * 3.0 / 200.0
    } else {
        15.0 + 27.0 * (frequency / 1000.0).ln() / 6.4_f64.ln()
    }
}

pub fn hz_to_mel_slaney_vec(frequencies: &[f64]) -> Vec<f64> {
    frequencies
        .iter()
        .map(|frequency| hz_to_mel_slaney(*frequency))
        .collect()
}

pub fn mel_htk_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

pub fn mel_htk_to_hz_vec(mels: &[f64]) -> Vec<f64> {
    mels.iter().map(|mel| mel_htk_to_hz(*mel)).collect()
}

pub fn mel_slaney_to_hz(mel: f64) -> f64 {
    1000.0 * 6.4_f64.powf((mel - 15.0) / 27.0)
}

pub fn mel_slaney_to_hz_vec(mels: &[f64]) -> Vec<f64> {
    mels.iter().map(|mel| mel_slaney_to_hz(*mel)).collect()
}

pub fn hz_to_octs(frequency: f64) -> f64 {
    (frequency / (440.0 / 16.0)).log2()
}

pub fn hz_to_octs_vec(frequencies: &[f64]) -> Vec<f64> {
    frequencies.iter().map(|frequency| hz_to_octs(*frequency)).collect()
}

pub fn octs_to_hz(oct: f64) -> f64 {
    440.0 / 16.0 * 2.0_f64.powf(oct)
}

pub fn octs_to_hz_vec(octs: &[f64]) -> Vec<f64> {
    octs.iter().map(|oct| octs_to_hz(*oct)).collect()
}

pub fn a4_to_tuning(frequency: f64, bins_per_octave: Option<u16>) -> f64 {
    let bins = bins_per_octave.unwrap_or(12);

    bins as f64 * (frequency / 440.0).log2()
}

pub fn a4_to_tuning_vec(frequencies: &[f64], bins_per_octave: Option<u16>) -> Vec<f64> {
    frequencies.iter().map(|frequency| a4_to_tuning(*frequency, bins_per_octave)).collect()
}

pub fn tuning_to_a4(tuning: f64, bins_per_octave: Option<u16>) -> f64 {
    let bins = bins_per_octave.unwrap_or(12);

    440.0 * 2.0_f64.powf(tuning / bins as f64)
}

pub fn tuning_to_a4_vec(tunings: &[f64], bins_per_octave: Option<u16>) -> Vec<f64> {
    tunings.iter().map(|tuning| tuning_to_a4(*tuning, bins_per_octave)).collect()
}

mod tests {
    use rodio::Sink;

    use super::*;

    #[test]
    fn midi_to_note_test() {
        assert_eq!(&midi_to_note(0.0), "C-1");
        assert_eq!(&midi_to_note(37.0), "C♯2");
        assert_eq!(&midi_to_note(-2.0), "A♯-2");
        assert_eq!(&midi_to_note(104.7), "A7");
    }

    #[test]
    fn note_to_midi_test() {
        assert_eq!(note_to_midi("C").unwrap(), 12.0);
        assert_eq!(note_to_midi("C#3").unwrap(), 49.0);
        assert_eq!(note_to_midi("C♯3").unwrap(), 49.0);
        assert_eq!(note_to_midi("C♭3").unwrap(), 47.0);
        assert_eq!(note_to_midi("f4").unwrap(), 65.0);
        assert_eq!(note_to_midi("Bb-1").unwrap(), 10.0);
        assert_eq!(note_to_midi("G𝄪6").unwrap(), 93.0);
        assert_eq!(note_to_midi("B𝄫6").unwrap(), 93.0);

        assert_eq!(note_to_midi_vec(&["C1", "E1", "G1"]).unwrap(), [24.0, 28.0, 31.0]);
    }

    #[test]
    fn hz_mel_test() {
        assert_eq!(hz_to_mel_slaney(60.0), 0.9);
        assert_eq!(
            hz_to_mel_slaney_vec(&[110.0, 220.0, 440.0]),
            [1.65, 3.3, 6.6]
        );
        assert_eq!(hz_to_mel_slaney(1234.0), 18.058261667852214);
    }

    #[test]
    fn hz_to_octs_test() {
        assert_eq!(hz_to_octs(440.0), 4.0);
        // assert_eq!(hz_to_octs_vec(&[32.0, 64.0, 128.0, 256.0]), [0.219, 1.219, 2.219, 3.219]);
    }

    #[test]
    fn a4_to_tuning_test() {
        assert_eq!(a4_to_tuning(440.0, None), 0.0);
        // assert_eq!(a4_to_tuning(432.0, None), -0.318);
        // assert_eq!(a4_to_tuning_vec(&[440.0, 444.0], Some(24)), [0., 0.313]);
    }

    #[test]
    fn tuning_to_a4_test() {
        assert_eq!(tuning_to_a4(0.0, None), 440.0);
        // assert_eq!(tuning_to_a4(-0.318, None), 431.992);
        // assert_eq!(tuning_to_a4_vec(&[0.1, 0.2, -0.1], Some(36)), [440.848, 441.698, 439.154]);
    }
}
