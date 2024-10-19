use std::time::Duration;

use peony::signal::Signal;
use rodio::{OutputStream, Sink};

fn main() {
    let signal: Signal<f32> = Signal::load(
        "Shape of You.wav",
        Duration::ZERO,
        Some(Duration::from_secs(10)),
    )
    .unwrap()
    .into_mono();

    let spectrum = signal.stft(
        2048,
        Some(512),
        None,
        peony::spectrum::WindowType::Hamming,
        true,
    );

    for channel in spectrum.freqs.iter() {
        for freqs in channel.iter().take(5) {
            for freq in freqs.iter().take(5) {
                print!("{:.2} ", freq);
            }
            println!();
        }
        println!();
    }

    let signal = spectrum.istft(
        2048,
        Some(512),
        None,
        peony::spectrum::WindowType::Hamming,
        true,
    );

    let source = signal.into_rodio_source::<f32>();

    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();
    sink.append(source);
    sink.sleep_until_end()
}
