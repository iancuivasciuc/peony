use peony::signal::Signal;
use std::time::Duration;
use std::time::Instant;

fn main() {
    let signal = Signal::tone(440.0, 44100, Duration::from_secs(1000));

    let start = Instant::now();

    let answer = signal.stft_default();

    let duration = start.elapsed();

    println!("Rust!");
    println!("Time elapsed: {:.4}", duration.as_secs_f64());
}
