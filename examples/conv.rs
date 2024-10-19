use peony::conv;

fn main() {
    let freqs = [440.0, 660.0, 125.0];

    let midi_numbers = conv::hz_to_midi_vec(&freqs);
    println!("Midi numbers:");
    for number in midi_numbers.iter() {
        print!("{} ", number);
    }
    println!();

    let notes = conv::midi_to_note_vec(&midi_numbers);
    println!("Notes:");
    for note in notes.iter() {
        print!("{} ", note);
    }
    println!();

    let notes = conv::hz_to_note_vec(&freqs);

    let freqs = conv::note_to_hz_vec(&notes).unwrap();
    println!("Freqs:");
    for freq in freqs.iter() {
        print!("{} ", freq);
    }
    println!();

    let mel_freqs = conv::hz_to_mel_slaney_vec(&freqs);
    println!("Mel freqs:");
    for freq in mel_freqs.iter() {
        print!("{} ", freq);
    }
    println!();
}